"""Agent node tests with mocked LLM backends."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.helpers import FakeEncoder, RecordingLLM


def test_single_agent_uses_llm_output(synthetic_agent_state, mock_get_backend):
    from src.agents.single_agent.merge_agent import resolve_conflict_agent_node

    out = resolve_conflict_agent_node(synthetic_agent_state)
    assert out["status"] == "resolved_agent"
    assert "src/main.py" in out["resolved_contents"]
    assert out["resolved_contents"]["src/main.py"]


def test_single_agent_fallback_without_llm(synthetic_agent_state):
    from src.agents.single_agent.merge_agent import resolve_conflict_agent_node

    with patch(
        "src.agents.single_agent.merge_agent.get_backend",
        return_value=(None, None),
    ):
        out = resolve_conflict_agent_node(synthetic_agent_state)
    assert out["status"] == "resolved_agent_stub"
    assert out["resolved_contents"] == synthetic_agent_state["parent_a_contents"]


def test_summarizer_node_with_llm(synthetic_agent_state, mock_get_backend):
    from src.agents.multi_agent.nodes import create_summarizer_node

    node = create_summarizer_node("bypass7")
    out = node(synthetic_agent_state)
    assert out["status"] == "summarised"
    assert "src/main.py" in out["summaries"]
    assert "summary_a" in out["summaries"]["src/main.py"]


def test_summarizer_fallback_without_llm(synthetic_agent_state):
    from src.agents.multi_agent.nodes import create_summarizer_node

    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(FakeEncoder(), None),
    ):
        node = create_summarizer_node("bypass7")
        out = node(synthetic_agent_state)
    assert out["status"] == "summarised"
    assert "Adds" in out["summaries"]["src/main.py"]["summary_a"]


def test_analyzer_node_sets_bypass_decision(synthetic_agent_state, mock_get_backend):
    from src.agents.multi_agent.nodes import create_conflict_analyzer_node

    # Pre-seed summaries so analyzer has context
    synthetic_agent_state["summaries"] = {
        "src/main.py": {"summary_a": "changed a", "summary_b": "changed b"}
    }
    node = create_conflict_analyzer_node("bypass7")
    out = node(synthetic_agent_state)
    assert out["status"] == "analyzed"
    assert out["bypass_decision"] in {"ALL_A", "ALL_B", "MIX"}


def test_resolution_and_review_nodes(synthetic_agent_state, mock_get_backend):
    from src.agents.multi_agent.nodes import (
        create_conflict_agent_node,
        create_resolution_agent_node,
        create_review_agent_node,
    )

    synthetic_agent_state["summaries"] = {
        "src/main.py": {"summary_a": "a", "summary_b": "b"}
    }
    synthetic_agent_state["bypass_decision"] = "MIX"

    plan_node = create_conflict_agent_node("bypass7")
    state = plan_node(synthetic_agent_state)
    assert state["status"] == "planned"
    assert "conflict_plan" in state

    # Force a simple A choice so resolution is deterministic
    state["conflict_plan"] = {"src/main.py": "A"}
    resolve_node = create_resolution_agent_node("bypass7")
    state = resolve_node(state)
    assert state["resolved_contents"]["src/main.py"] == synthetic_agent_state["parent_a_contents"]["src/main.py"]

    review_node = create_review_agent_node("bypass7")
    state = review_node(state)
    assert "review_results" in state
