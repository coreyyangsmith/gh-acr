"""Local recovery helpers for structured LLM outputs.

Prefer recovering a usable value from messy model text before spending
tokens on re-invokes. Pure functions — no degradation recording here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# Verdict tokens accepted by the conflict analyzer.
_VERDICT_MAP = {
    "a": "ALL_A",
    "all_a": "ALL_A",
    "parent_a": "ALL_A",
    "b": "ALL_B",
    "all_b": "ALL_B",
    "parent_b": "ALL_B",
    "mix": "MIX",
    "mixed": "MIX",
    "merge": "MIX",
}

# Strip common markdown / punctuation wrappers around a lone verdict token.
_TOKEN_STRIP = "\"'`*_.,;:()[]{}"

# Standalone line that is only a verdict (optional bold/quotes).
_STANDALONE_VERDICT = re.compile(
    r"^\s*(?:\*\*|__|`|'|\")?\s*"
    r"(?P<tok>A|B|Mix|MIX|ALL_A|ALL_B|Parent\s*A|Parent\s*B)"
    r"\s*(?:\*\*|__|`|'|\")?\s*$",
    re.IGNORECASE,
)

# "I choose A" / "I would choose **A**" / "choose Parent B as ..."
_CHOOSE_PHRASE = re.compile(
    r"(?is)(?:I\s+(?:would\s+)?choose|choose|verdict(?:\s+is)?|decision(?:\s+is)?)"
    r"\s*(?::|\s+)?\s*"
    r"(?:Parent\s+)?"
    r"(?:\*\*|__|`|'|\")?\s*"
    r"(?P<tok>A|B|Mix|MIX|ALL_A|ALL_B)"
    r"\s*(?:\*\*|__|`|'|\")?",
)

# Stop scanning standalone lines once rationale sections begin.
_RATIONALE_START = re.compile(
    r"(?i)^\s*(here'?s\s+why|reason(?:ing)?|rationale|evaluation)\b"
    r"|^\s*\d+\.\s+\*?\*?correctness",
)


def _map_token(tok: str) -> Optional[str]:
    key = re.sub(r"\s+", "_", tok.strip().lower())
    return _VERDICT_MAP.get(key)


def _strict_first_line(text: str) -> Optional[str]:
    """Current first-line / first-token behaviour (no degradation)."""
    raw = (text or "").strip()
    if not raw:
        return None
    first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), raw)
    token = first_line.split()[0].strip().strip(_TOKEN_STRIP).lower()
    mapped = _map_token(token)
    if mapped:
        return mapped
    compact = first_line.lower().replace(" ", "_").strip(_TOKEN_STRIP)
    return _map_token(compact)


def _standalone_verdict_line(text: str) -> Optional[str]:
    """First line that is exactly a verdict, before rationale sections."""
    for ln in (text or "").splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if _RATIONALE_START.search(stripped):
            break
        m = _STANDALONE_VERDICT.match(stripped)
        if m:
            mapped = _map_token(m.group("tok"))
            if mapped:
                return mapped
    return None


def _choose_phrase(text: str) -> Optional[str]:
    """Match 'I choose A' / 'choose Parent B' near the start of the output."""
    # Prefer the head of the response to avoid matching criterion prose.
    head = (text or "")[:800]
    m = _CHOOSE_PHRASE.search(head)
    if not m:
        return None
    return _map_token(m.group("tok"))


def extract_analyzer_verdict(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract ALL_A / ALL_B / MIX from analyzer output.

    Returns ``(decision, strategy)`` where strategy is one of
    ``strict_first_line``, ``standalone_line``, ``choose_phrase``, or
    ``(None, None)`` when unrecoverable.
    """
    raw = (text or "").strip()
    if not raw:
        return None, None

    for strategy, fn in (
        ("strict_first_line", _strict_first_line),
        ("standalone_line", _standalone_verdict_line),
        ("choose_phrase", _choose_phrase),
    ):
        decision = fn(raw)
        if decision:
            return decision, strategy
    return None, None


def normalize_decision_standard(text: str) -> str:
    """Normalize analyzer text to ALL_A / ALL_B / MIX (MIX if unrecoverable).

    Soft-degradation recording is the caller's responsibility when this
    returns MIX for empty/unrecognized input.
    """
    decision, _ = extract_analyzer_verdict(text)
    return decision if decision is not None else "MIX"


def _strip_markdown_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    # Drop opening fence (``` or ```json)
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    # Drop closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_balanced_json(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` or ``[...]`` substring, if any."""
    start = None
    opener = None
    closer = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if start is None:
            if ch in "{[":
                start = i
                opener = ch
                closer = "}" if ch == "{" else "]"
                depth = 1
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_lenient(text: str) -> Any:
    """Parse JSON from model text, recovering fenced / embedded objects.

    Raises
    ------
    json.JSONDecodeError | ValueError
        When no usable JSON value can be recovered.
    """
    raw = _strip_markdown_fences(text or "")
    if not raw:
        raise ValueError("empty JSON text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    embedded = _extract_balanced_json(raw)
    if embedded is None:
        raise ValueError("no JSON object/array found in text")
    return json.loads(embedded)


def parse_review_outcome(text: str) -> tuple[Optional[str], str]:
    """Parse reviewer JSON into ``(outcome, rationale)``.

    ``outcome`` is ``ACCEPT`` / ``REJECT`` or ``None`` if unrecoverable.
    ``rationale`` is best-effort text (may be the raw body on failure).
    """
    try:
        data = parse_json_lenient(text)
    except Exception:
        return None, (text or "").strip()

    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None, (text or "").strip()

    raw_outcome = str(data.get("outcome", "")).strip().upper()
    outcome = raw_outcome if raw_outcome in {"ACCEPT", "REJECT"} else None
    rationale = str(data.get("rationale", "")).strip()
    if not rationale and outcome is None:
        rationale = (text or "").strip()
    return outcome, rationale


def parse_plan_json(text: str, expected_paths: set[str] | None = None) -> dict[str, Any]:
    """Parse a per-file merge plan; raises on JSON/schema failure.

    Parameters
    ----------
    text
        Raw LLM output.
    expected_paths
        Conflicted file paths. When provided, the parsed object must share
        at least one key with this set or a ``ValueError`` is raised.
    """
    plan = parse_json_lenient(text)
    if not isinstance(plan, dict):
        raise ValueError(f"plan JSON is not an object: {type(plan).__name__}")
    if expected_paths is not None and expected_paths:
        if not (set(plan.keys()) & expected_paths):
            raise ValueError(
                f"plan JSON schema mismatch: parsed_keys={list(plan.keys())} "
                f"expected={sorted(expected_paths)}"
            )
    return plan


__all__ = [
    "extract_analyzer_verdict",
    "normalize_decision_standard",
    "parse_json_lenient",
    "parse_review_outcome",
    "parse_plan_json",
]
