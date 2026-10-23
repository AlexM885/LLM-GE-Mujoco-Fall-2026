"""Sandboxed loader for VLM-authored auxiliary reward snippets.

A VLM observes parent rollouts and emits a short Python snippet with the
exact signature::

    def aux_reward(obs, action, info):
        ...
        return <float>

The snippet shapes the TD3 training reward: ``r' = walker_reward + lam * aux_reward(obs, action, info)``.
Evaluation stays on the raw walker reward, so a bad aux_reward cannot
inflate fitness; the only risks it introduces are correctness bugs that
poison the TD3 critic and arbitrary code execution. This module defends
against both.

Guarantees:

- AST whitelist: a snippet that imports, opens files, exec/evals, defines
  classes/nested functions, or touches dunder names is rejected before
  it is ever compiled.
- Restricted globals: at call time the only names in scope are ``np`` and
  a small set of builtins (``abs``, ``float``, ``int``, ``len``, ``min``,
  ``max``). Attribute access is further filtered to a whitelist of numpy
  operations.
- Numeric clipping: NaN/inf returns and out-of-range values are clipped to
  ``[-DEFAULT_CLIP, DEFAULT_CLIP]``, which protects the critic.
- Per-call timeout via :mod:`signal` (SIGALRM) so a snippet that loops or
  calls into an expensive numpy operation can be bounded.

The sandbox returns a plain Python callable, so training code treats the
VLM-authored reward exactly like any other function.
"""

from __future__ import annotations

import ast
import math
import signal
from typing import Any, Callable, Mapping

import numpy as np


SIGNATURE_ARGS = ("obs", "action", "info")
FUNCTION_NAME = "aux_reward"

#: Clipped output range. Matches the typical scale of Walker2d per-step
#: reward (|r| <= ~5), so an aux signal of similar magnitude is well-formed
#: and far-out-of-scale snippets cannot swamp the training signal.
DEFAULT_CLIP = 10.0

#: Soft limit on how long one call to the snippet may take.
DEFAULT_TIMEOUT_SEC = 0.05

#: Builtins the snippet is allowed to use. Deliberately tiny.
_SAFE_BUILTINS: Mapping[str, Any] = {
    "abs": abs,
    "float": float,
    "int": int,
    "len": len,
    "min": min,
    "max": max,
    "range": range,
    "sum": sum,
}

#: Allowed numpy attributes. Everything else on ``np`` raises at validation
#: time, so unexpected code paths (e.g. ``np.load``, ``np.save``,
#: ``np.linalg.inv``) cannot execute. ``np.linalg.norm`` is reached via the
#: dotted whitelist in :data:`_ALLOWED_DOTTED_ATTRS`.
_ALLOWED_NP_ATTRS = frozenset({
    "abs", "clip", "cos", "dot", "exp", "float32", "float64", "hypot",
    "log", "log1p", "maximum", "mean", "minimum", "negative",
    "pi", "power", "prod", "sign", "sin", "sqrt", "square", "sum",
    "tanh", "where",
})

#: Dotted attribute paths that are explicitly allowed through the
#: ``np.<a>.<b>`` chain. Narrow by design.
_ALLOWED_DOTTED_ATTRS: Mapping[str, frozenset] = {
    "np.linalg": frozenset({"norm"}),
}


class AuxRewardSandboxError(Exception):
    """Raised when an aux_reward snippet fails validation or execution."""


# --------------------------------------------------------------------------- #
# AST whitelist
# --------------------------------------------------------------------------- #
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Assign, ast.AugAssign, ast.Return, ast.If, ast.Expr, ast.Pass,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Load, ast.Store,
    ast.Call, ast.Attribute, ast.Subscript, ast.Name, ast.Constant,
    ast.List, ast.Tuple, ast.Slice,
    ast.IfExp,
)

#: Reject these outright. Explicit list to make audit easier than
#: "anything not in the allow set" because the allow list grew above.
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom,
    ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda,
    ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While,
    ast.Try, ast.TryStar if hasattr(ast, "TryStar") else ast.Try,
    ast.Raise, ast.Assert,
    ast.Global, ast.Nonlocal,
    ast.Delete, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Await, ast.Yield, ast.YieldFrom,
    ast.NamedExpr,
)


def _dotted_name(node: ast.AST) -> str | None:
    """Flatten an attribute chain (``np.linalg.norm``) into its string form.

    Returns ``None`` when the chain is not a pure dotted path (e.g. the
    base is a call result rather than a Name).
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _check_attribute(node: ast.Attribute) -> str | None:
    """Return an error string if an attribute access is not whitelisted."""
    if node.attr.startswith("_"):
        return f"dunder/private attribute access not allowed: {node.attr}"
    dotted = _dotted_name(node)
    if dotted is None:
        return "attribute access on a non-Name expression is not allowed"
    parts = dotted.split(".")
    if parts[0] != "np":
        return f"attribute access on {parts[0]!r} not allowed (only np.*)"
    if len(parts) == 2:
        if parts[1] not in _ALLOWED_NP_ATTRS and dotted not in _ALLOWED_DOTTED_ATTRS:
            return f"np.{parts[1]} is not in the whitelist"
        return None
    if len(parts) == 3:
        prefix = ".".join(parts[:2])
        allowed = _ALLOWED_DOTTED_ATTRS.get(prefix)
        if not allowed or parts[2] not in allowed:
            return f"{dotted} is not in the whitelist"
        return None
    return f"attribute chain too deep: {dotted}"


def _check_name(node: ast.Name, allowed_locals: set[str]) -> str | None:
    if node.id.startswith("_"):
        return f"dunder/private name not allowed: {node.id}"
    if node.id == "np":
        return None
    if node.id in _SAFE_BUILTINS:
        return None
    if node.id in allowed_locals:
        return None
    return f"unknown name {node.id!r} (not in whitelist nor locally bound)"


def _collect_assigned_names(fn: ast.FunctionDef) -> set[str]:
    """Set of local names ever written in the function body.

    Pre-collected so that :func:`_walk_body` can validate ``Name(Load)``
    references without worrying about statement order.
    """
    names: set[str] = {arg.arg for arg in fn.args.args}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _walk_body(fn: ast.FunctionDef) -> None:
    """Walk the function body and raise on any forbidden construct."""
    allowed_locals = _collect_assigned_names(fn)
    for node in ast.walk(fn):
        if isinstance(node, _FORBIDDEN_NODES):
            raise AuxRewardSandboxError(
                f"forbidden syntax {type(node).__name__!r}")
        if not isinstance(node, _ALLOWED_NODES):
            raise AuxRewardSandboxError(
                f"disallowed AST node {type(node).__name__!r}")
        if isinstance(node, ast.Attribute):
            err = _check_attribute(node)
            if err:
                raise AuxRewardSandboxError(err)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            err = _check_name(node, allowed_locals)
            if err:
                raise AuxRewardSandboxError(err)
        elif isinstance(node, ast.FunctionDef) and node is not fn:
            raise AuxRewardSandboxError("nested function definitions not allowed")


def validate_aux_reward_code(code: str) -> tuple[bool, str]:
    """Static-check an aux_reward snippet. Returns ``(ok, reason)``."""
    if not isinstance(code, str) or not code.strip():
        return False, "code is empty or not a string"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "code must contain exactly one top-level function definition"
    fn = tree.body[0]
    if fn.name != FUNCTION_NAME:
        return False, f"function must be named {FUNCTION_NAME!r}, got {fn.name!r}"
    if fn.decorator_list:
        return False, "decorators are not allowed"
    args = [a.arg for a in fn.args.args]
    if tuple(args) != SIGNATURE_ARGS:
        return (False,
                f"signature must be ({', '.join(SIGNATURE_ARGS)}), got ({', '.join(args)})")
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs:
        return False, "aux_reward may not take *args, **kwargs, or kw-only args"
    try:
        _walk_body(fn)
    except AuxRewardSandboxError as exc:
        return False, str(exc)
    # The function must actually return. If every path falls off the end
    # the sandboxed call returns ``None``, which we would then have to
    # treat as zero; make it a validation error instead.
    if not any(isinstance(node, ast.Return) for node in ast.walk(fn)):
        return False, "aux_reward must contain a return statement"
    return True, ""


# --------------------------------------------------------------------------- #
# Loader + safe call wrapper
# --------------------------------------------------------------------------- #
def _build_safe_globals() -> dict[str, Any]:
    """Fresh globals dict for one exec. ``__builtins__`` is scrubbed."""
    return {
        "__builtins__": dict(_SAFE_BUILTINS),
        "np": np,
    }


def load_aux_reward(code: str) -> Callable[[Any, Any, Mapping[str, Any]], float]:
    """Validate, compile, and return the aux_reward callable.

    The returned callable is bound to a restricted globals dict and never
    sees the outer process state.
    """
    ok, reason = validate_aux_reward_code(code)
    if not ok:
        raise AuxRewardSandboxError(reason)
    safe_globals = _build_safe_globals()
    compiled = compile(code, filename="<aux_reward>", mode="exec")
    exec(compiled, safe_globals)
    fn = safe_globals.get(FUNCTION_NAME)
    if not callable(fn):
        raise AuxRewardSandboxError(
            f"loaded code did not expose a callable named {FUNCTION_NAME!r}")
    return fn


class _TimeoutSignal(Exception):
    pass


def _raise_timeout(_signum, _frame):
    raise _TimeoutSignal()


def _clip_result(value: Any, clip: float) -> float:
    """Coerce the snippet's return into a bounded float."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    if v > clip:
        return clip
    if v < -clip:
        return -clip
    return v


def safe_call(
    fn: Callable[..., Any],
    obs: Any,
    action: Any,
    info: Mapping[str, Any],
    *,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    clip: float = DEFAULT_CLIP,
) -> float:
    """Call ``fn(obs, action, info)`` with a timeout and numeric clipping.

    SIGALRM has 1-second resolution on Linux via ``signal.alarm``. For
    sub-second budgets we use ``signal.setitimer`` which supports fractional
    seconds. Timeout and arithmetic errors are swallowed and return 0.0,
    so a bad snippet degrades to "no shaping" rather than crashing training.
    """
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, max(timeout_sec, 1e-4))
    try:
        value = fn(obs, action, info)
    except _TimeoutSignal:
        value = 0.0
    except Exception:
        # The snippet is adversarial; any runtime error is caught here and
        # the step reward is simply unshaped on that step.
        value = 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Mask SIGALRM before restoring the original handler. If the alarm
        # fired just as we're exiting, _raise_timeout could re-trigger inside
        # signal.signal() and leak _TimeoutSignal past this `finally`. Setting
        # SIG_IGN first guarantees any pending delivery is dropped. Wrap the
        # restore itself in a try/except as a belt-and-braces defence.
        try:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.signal(signal.SIGALRM, old_handler)
        except _TimeoutSignal:
            pass
    return _clip_result(value, clip)
