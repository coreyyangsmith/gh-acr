"""Tests for local LLM output recovery helpers."""

from __future__ import annotations

import pytest

from src.agents.parse_utils import (
    extract_analyzer_verdict,
    normalize_decision_standard,
    parse_json_lenient,
    parse_plan_json,
    parse_review_outcome,
)

# Unique truncated analyzer outputs from the 2026-07-18 Llama-3.1 ablation
# (failures.jsonl unclear_verdict_fallback details).
_ABLATION_SNIPPETS: list[tuple[str, str]] = [
    (
        "Based on the provided evidence, I choose:\n\nA\n\nHere's why:\n\n1. **Correctness and safety**: Parent A adds a wide range of new resource estimation operators, including fermionic and ising models, orbital",
        "ALL_A",
    ),
    (
        "Based on the provided evidence, I would choose:\n\nA\n\nHere's why:\n\n1. **Correctness and safety**: Parent A adds support for a new cache type (CosmosDB) and removes unnecessary imports, which seems to be",
        "ALL_A",
    ),
    (
        "Based on the provided code, I would choose Parent A as the final verdict.\n\nHere's why:\n\n1. **Correctness and safety**: Both Parent A and Parent B have similar logic for handling custom LLMs and OpenAI",
        "ALL_A",
    ),
    (
        "Based on the provided code and the evaluation criteria, I would choose Parent B as the final merge result.\n\nHere's why:\n\n1. **Correctness and safety**: Both parents have similar code structures and mo",
        "ALL_B",
    ),
    (
        "Based on the provided code and the evaluation criteria, I would choose **A** as the verdict for the entire change set.\n\nHere's why:\n\n1. **Correctness and safety**: Parent A introduces a new `llm_modul",
        "ALL_A",
    ),
    (
        "Based on the provided evidence, I choose:\n\nB\n\nHere's why:\n\n1. **Correctness and safety**: Both parents seem to handle the same exceptions and edge cases. However, Parent B has a more robust way of han",
        "ALL_B",
    ),
    (
        "Based on the provided evidence, I choose:\n\nA\n\nHere's why:\n\n1. **Correctness and safety**: Parent A introduces a new class `PolicyHeadCont` that extends `Slicer` for continuous action spaces. This chan",
        "ALL_A",
    ),
]


@pytest.mark.parametrize("raw,expected", _ABLATION_SNIPPETS)
def test_extract_analyzer_verdict_ablation_snippets(raw: str, expected: str):
    decision, strategy = extract_analyzer_verdict(raw)
    assert decision == expected
    assert strategy in {"standalone_line", "choose_phrase", "strict_first_line"}


def test_extract_analyzer_verdict_strict_and_negative():
    assert extract_analyzer_verdict("A") == ("ALL_A", "strict_first_line")
    assert extract_analyzer_verdict("Parent B") == ("ALL_B", "strict_first_line")
    assert extract_analyzer_verdict("Mix") == ("MIX", "strict_first_line")
    assert extract_analyzer_verdict("") == (None, None)
    assert extract_analyzer_verdict("unclear verdict with no token") == (None, None)
    # Rationale-only prose mentioning Parent A must not false-positive
    prose = (
        "Here is my analysis.\n\n"
        "1. **Correctness and safety**: Parent A adds a feature. "
        "Parent B removes tests. No clear winner."
    )
    assert extract_analyzer_verdict(prose) == (None, None)


def test_normalize_decision_standard_compat():
    assert normalize_decision_standard("a") == "ALL_A"
    assert normalize_decision_standard("unclear") == "MIX"


def test_parse_json_lenient_fenced_and_embedded():
    assert parse_json_lenient('```json\n{"a.py": "merge"}\n```') == {"a.py": "merge"}
    assert parse_json_lenient('Sure.\n{"x": 1}\nThanks') == {"x": 1}
    with pytest.raises((ValueError, Exception)):
        parse_json_lenient("not json at all")


def test_parse_plan_json_schema():
    plan = parse_plan_json('{"a.py": "A", "b.py": "merge"}', expected_paths={"a.py", "b.py"})
    assert plan["a.py"] == "A"
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_plan_json('{"strategy": "merge"}', expected_paths={"a.py"})


def test_parse_review_outcome():
    outcome, rationale = parse_review_outcome(
        '{"outcome": "accept", "rationale": "looks fine"}'
    )
    assert outcome == "ACCEPT"
    assert rationale == "looks fine"
    outcome, _ = parse_review_outcome('[{"outcome": "REJECT", "rationale": "bad"}]')
    assert outcome == "REJECT"
    outcome, rationale = parse_review_outcome("not json")
    assert outcome is None
    assert "not json" in rationale
