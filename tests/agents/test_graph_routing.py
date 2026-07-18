"""Unit tests for multi-agent graph routing and pure node helpers."""

from __future__ import annotations

from src.agents.multi_agent.graph_builder import (
    _prepare_feedback_for_retry,
    _route_after_analyze,
    _route_after_review,
)
from src.agents.multi_agent.nodes import (
    _fallback_summary,
    _init_token_counts,
    _normalize_decision_standard,
)


def test_route_after_review_all_accept():
    state = {
        "_review_iter": 0,
        "review_results": {
            "a.py": {"outcome": "ACCEPT"},
            "b.py": {"outcome": "ACCEPT"},
        },
    }
    assert _route_after_review(state) == "finish"


def test_route_after_review_max_iters():
    state = {
        "_review_iter": 2,
        "review_results": {"a.py": {"outcome": "REJECT"}},
    }
    assert _route_after_review(state) == "finish"


def test_route_after_review_retry():
    state = {
        "_review_iter": 0,
        "review_results": {"a.py": {"outcome": "REJECT"}},
    }
    assert _route_after_review(state) == "retry"


def test_prepare_feedback_for_retry_increments_and_aggregates():
    state = {
        "_review_iter": 0,
        "review_results": {
            "a.py": {"outcome": "REJECT", "rationale": "fix indent"},
            "b.py": {"outcome": "ACCEPT", "rationale": "ok"},
        },
        "reviews": {"a.py": "raw review a"},
    }
    out = _prepare_feedback_for_retry(state)
    assert out["_review_iter"] == 1
    assert "a.py" in out["review_feedback"]
    assert "fix indent" in out["review_feedback"]["a.py"]
    assert "b.py" not in out["review_feedback"]


def test_route_after_analyze():
    assert _route_after_analyze({"bypass_decision": "ALL_A"}) == "all_a"
    assert _route_after_analyze({"bypass_decision": "ALL_B"}) == "all_b"
    assert _route_after_analyze({"bypass_decision": "MIX"}) == "mix"
    assert _route_after_analyze({}) == "mix"


def test_fallback_summary_counts_adds_dels():
    diff = "--- a\n+++ b\n@@\n-old\n+new1\n+new2\n"
    summary = _fallback_summary(diff)
    assert "Adds 2" in summary
    assert "removes 1" in summary


def test_normalize_decision_standard():
    assert _normalize_decision_standard("ALL_A") == "ALL_A"
    assert _normalize_decision_standard("b") == "ALL_B"
    # "mix" contains neither "a" nor "b" as a substring
    assert _normalize_decision_standard("MIX") == "MIX"
    assert _normalize_decision_standard("a") == "ALL_A"
    assert _normalize_decision_standard("") == "MIX"


def test_init_token_counts_creates_nested_defaults():
    state: dict = {}
    counts = _init_token_counts(state, "a.py")
    assert counts == {
        "system_prompt": 0,
        "original": 0,
        "diff_a": 0,
        "diff_b": 0,
        "output": 0,
    }
    # Second call returns same dict
    assert _init_token_counts(state, "a.py") is counts
