"""Baseline (Parent-A) merge-conflict resolver.

This agent does **no intelligent merging** – it simply takes the file version
from *Parent A* and returns it as the "resolved" content.  It acts as a lower
bound for evaluation and incurs zero external cost.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["resolve_conflict_base_node"]

FileContents = Dict[str, str]

def resolve_conflict_base_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """LangGraph node that picks Parent-A's content for every conflicted file."""
    state["resolved_contents"] = state["parent_a_contents"]  # type: ignore[index]
    state["status"] = "resolved_base"
    return state
