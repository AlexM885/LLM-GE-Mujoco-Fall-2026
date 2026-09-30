"""Safety checks around LLM output for the MESB operators (no torch / LLM imports).

Two jobs:

``extract_code(reply)``
    Replacement for ``llm_utils.clean_code_from_llm``, installed at runtime by
    ``src/mesb_llm_operator.py`` (the team's file is not modified). The
    original always returns the *first* fenced block. On PACE, Llama-3.3
    replied with a made-up format template first::

        Your response should be in this format: ```python
        [modified code]
        ```
        The modified code is as follows: ```python
        def get_ppo_kwargs(): ...
        ```

    and the original saved ``[modified code]`` as the gene. Here every fenced
    block is considered; placeholders are dropped; among blocks that parse as
    Python, the last one that actually defines something wins. If nothing
    usable is found the original's failure token ``"ERROR"`` is returned.
    Preference order: blocks defining a function/class, then blocks with
    only assignments, then any parseable block, then anything non-placeholder;
    within a tier the last block wins (LLMs tend to end with the final code).

``validate_child(path)``
    Static check of a finished child network before it is trained: it must
    parse, define top-level ``get_policy_kwargs`` and ``get_ppo_kwargs``, and
    contain no bare-name statements (``ERROR``, ``modified_code`` ...) that
    would crash at import. Duplicate top-level definitions (the LLM rewrote
    the whole module inside one chunk; Python keeps the last one) are
    reported but allowed.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

REQUIRED_FUNCTIONS = ("get_policy_kwargs", "get_ppo_kwargs")
FAILURE_TOKEN = "ERROR"

_BLOCK_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.S)
_OPEN_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*)$", re.S)
_PLACEHOLDER_RE = re.compile(
    r"^\s*(\[[^\]\n]*\]|<[^>\n]*>|\.\.\.|…|pass|#[^\n]*|your code here|code here|"
    r"modified code|updated code|" + FAILURE_TOKEN + r")\s*$", re.I)


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def is_placeholder(code: str) -> bool:
    """True for template stand-ins such as ``[modified code]`` or ``...``."""
    lines = [ln for ln in code.strip().splitlines() if ln.strip()]
    return not lines or all(_PLACEHOLDER_RE.match(ln) for ln in lines)


def _definition_rank(code: str) -> int:
    """2 = defines a function/class, 1 = only assignments, 0 = neither/unparseable."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return 0
    if any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body):
        return 2
    if any(isinstance(n, (ast.Assign, ast.AnnAssign)) for n in tree.body):
        return 1
    return 0


def code_blocks(reply: str) -> list[str]:
    """All fenced code blocks, plus an unterminated trailing one (token limit)."""
    blocks = [body.strip("\n") for _, body in _BLOCK_RE.findall(reply)]
    tail = reply[reply.rfind("```"):] if reply.count("```") % 2 == 1 else ""
    if tail:
        m = _OPEN_RE.match(tail)
        if m:
            blocks.append(m.group(2).strip("\n"))
    return blocks


def extract_code(reply) -> str:
    """Best code block from an LLM reply, or ``"ERROR"`` (see module docstring)."""
    if not isinstance(reply, str) or not reply.strip():
        return FAILURE_TOKEN
    candidates = [b.strip() for b in code_blocks(reply) if b.strip() and not is_placeholder(b)]
    parseable = [b for b in candidates if _parses(b)]
    with_defs = [b for b in parseable if _definition_rank(b) == 2]
    assigning = [b for b in parseable if _definition_rank(b) == 1]
    for pool in (with_defs, assigning, parseable, candidates):
        if pool:
            return pool[-1]
    return FAILURE_TOKEN


def validate_child(path: str | Path) -> tuple[bool, str, dict]:
    """Return (ok, reason, info) for a generated network file."""
    path = Path(path)
    if not path.exists():
        return False, f"child file not written: {path}", {}
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError line {exc.lineno}: {exc.msg}", {}
    top = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    names = Counter(n.name for n in top)
    missing = [f for f in REQUIRED_FUNCTIONS if f not in names]
    if missing:
        return False, f"missing required function(s) {missing}", {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.List, ast.Set)):
            return False, (f"bare expression on line {node.lineno} "
                           f"({ast.get_source_segment(source, node)!r}) would fail at import"), {}
    duplicates = sorted(n for n, c in names.items() if c > 1)
    return True, "ok", {"duplicate_definitions": duplicates}
