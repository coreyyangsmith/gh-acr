"""Unit tests for deterministic baseline resolvers."""

from __future__ import annotations

from src.agents.base_agent.base_a import resolve_conflict_base_a_node
from src.agents.base_agent.base_b import resolve_conflict_base_b_node


def _state():
    return {
        "parent_a_contents": {"a.py": "A"},
        "parent_b_contents": {"a.py": "B"},
        "status": "prepared",
    }


def test_base_a_copies_parent_a():
    state = _state()
    out = resolve_conflict_base_a_node(state)
    assert out["resolved_contents"] == {"a.py": "A"}
    assert out["status"] == "resolved_base_a"
    assert out is state  # mutates in place


def test_base_b_copies_parent_b():
    state = _state()
    out = resolve_conflict_base_b_node(state)
    assert out["resolved_contents"] == {"a.py": "B"}
    assert out["status"] == "resolved_base_b"
