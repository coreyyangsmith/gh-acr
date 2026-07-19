"""Tests for soft-degradation tracking."""

from __future__ import annotations

from src.utils.degradation import (
    clear_degradations,
    get_degradations,
    has_degradations,
    primary_degradation_category,
    record_degradation,
)


def test_degradation_tracker_clear_record_get():
    clear_degradations()
    assert not has_degradations()
    assert get_degradations() == []
    assert primary_degradation_category() is None

    record_degradation(
        "prompt_truncation",
        "clipped",
        detail="tokens=9000",
        node="truncating_llm_wrapper",
    )
    record_degradation(
        "json_parse_fallback",
        "bad json",
        node="conflict_agent",
        file="a.py",
    )

    assert has_degradations()
    events = get_degradations()
    assert len(events) == 2
    assert events[0]["category"] == "prompt_truncation"
    assert events[0]["detail"] == "tokens=9000"
    assert events[1]["file"] == "a.py"
    assert primary_degradation_category() == "prompt_truncation"
    assert primary_degradation_category(events) == "prompt_truncation"

    # get_degradations returns a copy
    events[0]["category"] = "mutated"
    assert get_degradations()[0]["category"] == "prompt_truncation"

    clear_degradations()
    assert not has_degradations()
    assert get_degradations() == []


def test_record_without_clear_still_works():
    # Fresh context may have None; recording should still succeed.
    clear_degradations()
    clear_degradations()  # ensure empty list
    # Simulate never-cleared by setting via record after clear
    record_degradation("llm_unavailable_heuristic", "stub")
    assert has_degradations()
    clear_degradations()
