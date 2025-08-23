"""Dynamic multi-agent merge resolver package.

Flow (LangGraph nodes):
1. summarizer_agent_node  – summaries of A/B diffs
2. conflict_agent_node    – high-level merge plan AND per-file dynamic prompts
3. resolution_agent_node  – produces patched file(s) using per-file dynamic prompts
4. review_agent_node      – final approval / comments, with up to 3 repair loops

The entry-point callable `resolve_conflict_dynamic_agent_node` composes the above
nodes into a *mini* LangGraph executed synchronously.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph
import difflib

# Reuse summarizer and review from the multi-agent package
from ..multi.summarizer_agent import summarizer_agent_node  # type: ignore
from ..multi.review_agent import review_agent_node  # type: ignore

# Dynamic variants
from .conflict_agent import conflict_agent_node
from .resolution_agent import resolution_agent_node

__all__ = ["resolve_conflict_dynamic_agent_node"]


def _route_after_review(state: Dict[str, Any]) -> str:
    """Route based on review outcomes and retry budget."""
    max_iters = 2
    iter_no = int(state.get("_review_iter", 0))
    results = state.get("review_results", {}) or {}
    if results and all((v or {}).get("outcome") == "ACCEPT" for v in results.values()):
        return "finish"
    if iter_no >= max_iters:
        return "finish"
    return "retry"


def _prepare_feedback_for_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate reviews across iterations so resolution sees cumulative feedback."""
    iter_no = int(state.get("_review_iter", 0))
    review_results = state.get("review_results", {}) or {}
    raw_reviews = state.get("reviews", {}) or {}

    history: Dict[str, list[str]] = state.setdefault("review_feedback_history", {})  # type: ignore[assignment]

    for path, data in review_results.items():
        if not isinstance(data, dict):
            continue
        if str(data.get("outcome", "")).upper() == "ACCEPT":
            continue
        rationale = str(data.get("rationale", "")).strip()
        full_text = str(raw_reviews.get(path, "")).strip()
        entry_parts = []
        if rationale:
            entry_parts.append(f"Rationale: {rationale}")
        if full_text and full_text != rationale:
            entry_parts.append(f"Raw: {full_text}")
        if not entry_parts:
            entry_parts.append("(no review comments provided)")
        entry = f"Iteration {iter_no}:\n" + "\n".join(entry_parts)
        history.setdefault(path, []).append(entry)

    feedback_map: Dict[str, str] = {}
    for path, entries in history.items():
        feedback_map[path] = "\n\n---\n\n".join(entries)

    state["review_feedback"] = feedback_map
    state["_review_iter"] = iter_no + 1
    state["review_feedback_history"] = history
    return state


def resolve_conflict_dynamic_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Dynamic merge-conflict resolver that generates per-file prompts during planning."""

    sg = StateGraph(dict)

    sg.add_node("summarise", summarizer_agent_node)
    sg.add_node("plan", conflict_agent_node)
    sg.add_node("patch", resolution_agent_node)
    sg.add_node("review", review_agent_node)

    def _finalize_node(s: Dict[str, Any]) -> Dict[str, Any]:
        resolved = s.get("resolved_contents", {}) or {}
        final_diffs = dict(s.get("final_diffs", {}) or {})
        ancestor_contents = s.get("ancestor_contents", {}) or {}
        for path, merged_text in resolved.items():
            if path not in final_diffs:
                a_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
                m_lines = str(merged_text).splitlines(keepends=True)
                final_diffs[path] = "".join(
                    difflib.unified_diff(a_lines, m_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
                )
        s["final_diffs"] = final_diffs
        s["status"] = "review_finalized"
        return s

    sg.add_node("finalize", _finalize_node)

    def _feedback_node(s: Dict[str, Any]) -> Dict[str, Any]:
        return _prepare_feedback_for_retry(s)

    sg.add_node("feedback", _feedback_node)

    sg.set_entry_point("summarise")
    sg.add_edge("summarise", "plan")
    sg.add_edge("plan", "patch")
    sg.add_edge("patch", "review")
    sg.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "finish": "finalize",
            "retry": "feedback",
        },
    )
    sg.add_edge("finalize", END)
    sg.add_edge("feedback", "patch")

    sub_app = sg.compile()
    if "_review_iter" not in state:
        state["_review_iter"] = 0
    return sub_app.invoke(state)


