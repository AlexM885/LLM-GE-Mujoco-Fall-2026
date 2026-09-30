"""Code-block extraction and child validation (no LLM, no torch)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mesb_llm_guard import extract_code, is_placeholder, validate_child  # noqa: E402

# The reply Llama-3.3 actually gave on PACE (2026-09-29). The team's
# clean_code_from_llm returned "[modified code]" for it.
PACE_REPLY = '''Here is your task:

1.  **Modify** `learning_rate` to be **0.00025**, and keep all other hyperparameters unchanged.
2.  **Add** a new key-value pair: `"target_kl"` with value **0.03**.

Your response should be in this format: "The modified code is as follows:
```python
[modified code]
```"

The modified code is as follows:
```python
# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    return dict(
        learning_rate=0.00025,
        gamma=0.99,
        target_kl=0.03,
        verbose=0,
    )
```
'''


def test_pace_reply_skips_placeholder():
    code = extract_code(PACE_REPLY)
    assert "[modified code]" not in code
    assert "def get_ppo_kwargs" in code and "target_kl=0.03" in code


def test_single_block_matches_original_behaviour():
    reply = "Sure:\n```python\nHIDDEN = [64]\ndef f():\n    return 1\n```\nDone."
    assert extract_code(reply) == "HIDDEN = [64]\ndef f():\n    return 1"


@pytest.mark.parametrize("fence", ["```python", "```py", "```Python", "```", "``` python"])
def test_language_tags(fence):
    assert extract_code(f"x\n{fence}\ndef g():\n    pass\n```") == "def g():\n    pass"


def test_prefers_definitions_over_trailing_usage_snippet():
    reply = ("```python\ndef get_ppo_kwargs():\n    return {}\n```\n"
             "Use it like this:\n```python\nmodel = PPO('MlpPolicy', env)\n```")
    assert extract_code(reply).startswith("def get_ppo_kwargs")


def test_last_definition_block_wins():
    reply = ("Original:\n```python\ndef get_ppo_kwargs():\n    return {'a': 1}\n```\n"
             "Improved:\n```python\ndef get_ppo_kwargs():\n    return {'a': 2}\n```")
    assert "'a': 2" in extract_code(reply)


def test_unterminated_block_from_token_limit():
    reply = "Here:\n```python\ndef f():\n    return 3\n"
    assert extract_code(reply) == "def f():\n    return 3"


@pytest.mark.parametrize("reply", [None, "", "   ", "no code at all",
                                   "```python\n[modified code]\n```",
                                   "```python\n...\n```\n```\n# TODO\n```"])
def test_unusable_replies_give_error_token(reply):
    assert extract_code(reply) == "ERROR"


def test_unparseable_block_still_returned_when_nothing_better():
    # Behaves like the original (returns code) so the existing FORBIDDEN_PATTERNS /
    # validation stages decide; validate_child will then reject it.
    assert extract_code("```python\ndef broken(:\n```") == "def broken(:"


def test_placeholder_detection():
    assert is_placeholder("[modified code]")
    assert is_placeholder("  ...  \n")
    assert is_placeholder("# your code here")
    assert not is_placeholder("x = 1")


# ------------------------------------------------------------------ validation
SEED = REPO / "sota" / "MujocoRL" / "network.py"


def test_seed_network_is_valid():
    ok, reason, info = validate_child(SEED)
    assert ok, reason
    assert info["duplicate_definitions"] == []


def _write(tmp_path, text):
    p = tmp_path / "network_x.py"
    p.write_text(text)
    return p


def test_placeholder_child_rejected(tmp_path):
    text = SEED.read_text().replace(
        SEED.read_text().split("# --OPTION--")[2], "\n[modified code]\n")
    ok, reason, _ = validate_child(_write(tmp_path, text))
    assert not ok and "SyntaxError" in reason


def test_error_token_child_rejected(tmp_path):
    text = SEED.read_text() + "\nERROR\n"
    ok, reason, _ = validate_child(_write(tmp_path, text))
    assert not ok and "bare expression" in reason


def test_missing_function_rejected(tmp_path):
    text = SEED.read_text().replace("def get_ppo_kwargs", "def get_ppo_kwarg")
    ok, reason, _ = validate_child(_write(tmp_path, text))
    assert not ok and "get_ppo_kwargs" in reason


def test_duplicates_reported_not_rejected(tmp_path):
    text = SEED.read_text() + "\n\ndef get_ppo_kwargs():\n    return {}\n"
    ok, _, info = validate_child(_write(tmp_path, text))
    assert ok and info["duplicate_definitions"] == ["get_ppo_kwargs"]


def test_missing_file(tmp_path):
    ok, reason, _ = validate_child(tmp_path / "nope.py")
    assert not ok and "not written" in reason
