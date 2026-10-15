"""Tests for the VLM aux_reward sandboxed loader.

The sandbox is the only line of defense between an LLM-authored snippet
and the training loop, so every attack vector and every numeric edge case
is exercised explicitly here.
"""

from __future__ import annotations

import math
import os
import sys
import textwrap

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sota", "MujocoRL"))

import aux_reward_sandbox as sb  # noqa: E402


def _dedent(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


# --------------------------------------------------------------------------- #
# Shape and signature
# --------------------------------------------------------------------------- #
class TestShape:
    def test_minimal_valid_snippet(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return 0.0
        """)
        ok, reason = sb.validate_aux_reward_code(code)
        assert ok, reason

    def test_rejects_empty_string(self):
        ok, reason = sb.validate_aux_reward_code("")
        assert not ok and "empty" in reason

    def test_rejects_non_string(self):
        ok, _ = sb.validate_aux_reward_code(None)  # type: ignore[arg-type]
        assert not ok

    def test_rejects_syntax_error(self):
        ok, reason = sb.validate_aux_reward_code("def aux_reward(:")
        assert not ok and "syntax" in reason

    def test_rejects_wrong_function_name(self):
        code = _dedent("""
            def shape(obs, action, info):
                return 0.0
        """)
        ok, reason = sb.validate_aux_reward_code(code)
        assert not ok and "aux_reward" in reason

    def test_rejects_wrong_signature(self):
        code = _dedent("""
            def aux_reward(state, act):
                return 0.0
        """)
        ok, reason = sb.validate_aux_reward_code(code)
        assert not ok and "signature" in reason

    def test_rejects_varargs(self):
        code = _dedent("""
            def aux_reward(obs, action, info, *args):
                return 0.0
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_kwargs(self):
        code = _dedent("""
            def aux_reward(obs, action, info, **kw):
                return 0.0
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_decorator(self):
        code = _dedent("""
            @staticmethod
            def aux_reward(obs, action, info):
                return 0.0
        """)
        ok, reason = sb.validate_aux_reward_code(code)
        assert not ok and "decorator" in reason

    def test_rejects_multiple_top_level_definitions(self):
        code = _dedent("""
            def helper(x):
                return x
            def aux_reward(obs, action, info):
                return 0.0
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_missing_return(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                x = 1.0
        """)
        ok, reason = sb.validate_aux_reward_code(code)
        assert not ok and "return" in reason


# --------------------------------------------------------------------------- #
# Attack vectors on the AST whitelist
# --------------------------------------------------------------------------- #
class TestAttacks:
    @pytest.mark.parametrize("hostile", [
        "import os",
        "from os import system",
        "exec('x=1')",
        "eval('1+1')",
        "open('/etc/passwd').read()",
        "__import__('os')",
        "obs.__class__",
        "class Evil: pass",
        "lambda x: x",
        "with open('x') as f: pass",
        "for i in range(5): pass",
        "while True: pass",
        "try: pass\nexcept: pass",
        "assert False",
        "raise RuntimeError('x')",
        "global y",
        "del obs",
        "[i for i in range(5)]",
        "(i for i in range(5))",
        "{i for i in range(5)}",
        "yield 1",
        "await x",
    ])
    def test_rejects_forbidden_body(self, hostile: str):
        # Each hostile line goes inside the function body; indentation matters
        # for try/except so dedent + reindent carefully.
        indented = "\n".join("    " + line for line in hostile.splitlines())
        code = f"def aux_reward(obs, action, info):\n{indented}\n    return 0.0\n"
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok, f"should reject: {hostile!r}"

    def test_rejects_dunder_attribute(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(obs.__class__.__name__ == 'ndarray')
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_attribute_on_non_np_name(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return info.get('reward_forward', 0.0)
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_non_whitelisted_np_attr(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.load('x').sum())
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_deep_np_chain(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.linalg.norm(obs) + np.random.random.seed(1))
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_accepts_np_linalg_norm(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.linalg.norm(action))
        """)
        ok, reason = sb.validate_aux_reward_code(code)
        assert ok, reason

    def test_rejects_nested_function(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                def inner(x):
                    return x
                return 0.0
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_rejects_unknown_name(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return _os.getcwd()
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok


# --------------------------------------------------------------------------- #
# Loader + call semantics
# --------------------------------------------------------------------------- #
def _info(**kw):
    base = {"reward_forward": 0.0, "reward_survive": 1.0, "reward_ctrl": 0.0,
            "x_position": 0.0}
    base.update(kw)
    return base


class TestLoadAndCall:
    def test_load_returns_callable(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.sum(np.square(action)))
        """)
        fn = sb.load_aux_reward(code)
        assert callable(fn)

    def test_simple_call_returns_expected_value(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.sum(np.square(action)))
        """)
        fn = sb.load_aux_reward(code)
        value = sb.safe_call(fn, np.zeros(10), np.array([1.0, 2.0]), _info())
        assert value == pytest.approx(5.0)

    def test_call_with_obs_feature(self):
        # torso height proxy via obs[0] on Walker2d
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(obs[0])
        """)
        fn = sb.load_aux_reward(code)
        obs = np.arange(17, dtype=float)
        obs[0] = 1.2
        value = sb.safe_call(fn, obs, np.zeros(6), _info())
        assert value == pytest.approx(1.2)

    def test_builtins_abs_min_max(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return max(min(abs(obs[0]), 1.0), -1.0)
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.array([2.5]), np.zeros(1), _info()) == 1.0

    def test_subscript_and_slice(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.sum(obs[0:3]))
        """)
        fn = sb.load_aux_reward(code)
        obs = np.array([1.0, 2.0, 3.0, 99.0])
        assert sb.safe_call(fn, obs, np.zeros(1), _info()) == pytest.approx(6.0)

    def test_if_expression(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return 1.0 if obs[0] > 0 else -1.0
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.array([0.5]), np.zeros(1), _info()) == 1.0
        assert sb.safe_call(fn, np.array([-0.5]), np.zeros(1), _info()) == -1.0


class TestNumericSafety:
    def test_nan_result_clipped_to_zero(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.log(0.0) - np.log(0.0))
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.zeros(1), np.zeros(1), _info()) == 0.0

    def test_inf_result_clipped_to_zero(self):
        # +inf and -inf both get mapped to 0 (match NaN handling), so a
        # pathological divergence can't poison the critic.
        code = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.log(0.0))
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.zeros(1), np.zeros(1), _info()) == 0.0

    def test_overflow_clipped_to_bound(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return 10000.0
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.zeros(1), np.zeros(1), _info()) == sb.DEFAULT_CLIP

    def test_underflow_clipped_to_negative_bound(self):
        code = _dedent("""
            def aux_reward(obs, action, info):
                return -10000.0
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.zeros(1), np.zeros(1), _info()) == -sb.DEFAULT_CLIP

    def test_runtime_error_swallowed(self):
        # Division by zero at Python level raises; safe_call swallows.
        code = _dedent("""
            def aux_reward(obs, action, info):
                return 1.0 / 0.0
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.zeros(1), np.zeros(1), _info()) == 0.0

    def test_non_numeric_return_swallowed(self):
        # The snippet returns a numpy array; _clip_result's float() cast
        # raises TypeError on multi-element arrays and we return 0.
        code = _dedent("""
            def aux_reward(obs, action, info):
                return obs
        """)
        fn = sb.load_aux_reward(code)
        assert sb.safe_call(fn, np.zeros(3), np.zeros(1), _info()) == 0.0


class TestSandboxIsolation:
    def test_no_builtins_leak(self):
        # ``print`` is intentionally not in the whitelist; a snippet that
        # tries to use it should fail validation (unknown name).
        code = _dedent("""
            def aux_reward(obs, action, info):
                print('x')
                return 0.0
        """)
        ok, _ = sb.validate_aux_reward_code(code)
        assert not ok

    def test_fresh_globals_per_load(self):
        code_a = _dedent("""
            def aux_reward(obs, action, info):
                a = 1.0
                return a
        """)
        code_b = _dedent("""
            def aux_reward(obs, action, info):
                return float(np.sum(action))
        """)
        fn_a = sb.load_aux_reward(code_a)
        fn_b = sb.load_aux_reward(code_b)
        # Different closures: changing one must not affect the other
        assert fn_a.__globals__ is not fn_b.__globals__


class TestInvalidLoad:
    def test_load_raises_on_invalid(self):
        with pytest.raises(sb.AuxRewardSandboxError):
            sb.load_aux_reward("import os\n")

    def test_load_raises_when_body_empty(self):
        with pytest.raises(sb.AuxRewardSandboxError):
            sb.load_aux_reward("")
