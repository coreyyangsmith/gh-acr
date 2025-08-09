"""Baseline resolver that selects Parent B's content for all files."""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["resolve_conflict_base_b_node"]

FileContents = Dict[str, str]


def resolve_conflict_base_b_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """LangGraph node that picks Parent-B's content for every conflicted file."""
    state["resolved_contents"] = state["parent_b_contents"]  # type: ignore[index]
    state["status"] = "resolved_base_b"
    return state

