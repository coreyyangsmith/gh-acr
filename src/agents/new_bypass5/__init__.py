"""new_bypass multi-agent merge resolver package (clone of bypass7)."""

from __future__ import annotations

from typing import Any, Dict
from langgraph.graph import END, StateGraph
import difflib

from .summarizer_agent import summarizer_agent_node
from .conflict_analyzer import conflict_analyzer_node
from .conflict_agent import conflict_agent_node
from .resolution_agent import resolution_agent_node
from .review_agent import review_agent_node

__all__ = [
    "resolve_conflict_new_bypass_multi_agent_node",
    "resolve_conflict_new_bypass5_multi_agent_node",
]


def _route_after_review(state: Dict[str, Any]) -> str:
    max_iters = 2
    iter_no = int(state.get("_review_iter", 0))
    results = state.get("review_results", {}) or {}
    if results and all((v or {}).get("outcome") == "ACCEPT" for v in results.values()):
        return "finish"
    if iter_no >= max_iters:
        return "finish"
    return "retry"


def _prepare_feedback_for_retry(state: Dict[str, Any]) -> Dict[str, Any]:
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


def resolve_conflict_new_bypass_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    sg = StateGraph(dict)

    # Core nodes
    sg.add_node("summarise", summarizer_agent_node)
    sg.add_node("analyze", conflict_analyzer_node)
    sg.add_node("plan", conflict_agent_node)
    sg.add_node("patch", resolution_agent_node)
    sg.add_node("review", review_agent_node)

    def _bypass_select(s: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(s.get("bypass_decision", "MIX")).upper()
        if decision == "ALL_A":
            s["resolved_contents"] = s.get("parent_a_contents", {}) or {}
        elif decision == "ALL_B":
            s["resolved_contents"] = s.get("parent_b_contents", {}) or {}
        s["status"] = (
            "bypassed" if decision in ("ALL_A", "ALL_B") else s.get("status", "")
        )
        return s

    sg.add_node("bypass", _bypass_select)

    def _finalize_node(s: Dict[str, Any]) -> Dict[str, Any]:
        resolved = s.get("resolved_contents", {}) or {}
        final_diffs = dict(s.get("final_diffs", {}) or {})
        ancestor_contents = s.get("ancestor_contents", {}) or {}
        for path, merged_text in resolved.items():
            if path not in final_diffs:
                a_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
                m_lines = str(merged_text).splitlines(keepends=True)
                final_diffs[path] = "".join(
                    difflib.unified_diff(
                        a_lines, m_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
                    )
                )
        s["final_diffs"] = final_diffs
        s["status"] = "review_finalized"
        return s

    sg.add_node("finalize", _finalize_node)

    def _feedback_node(s: Dict[str, Any]) -> Dict[str, Any]:
        return _prepare_feedback_for_retry(s)

    sg.add_node("feedback", _feedback_node)

    def _route_after_analyze(s: Dict[str, Any]) -> str:
        decision = str(s.get("bypass_decision", "MIX")).upper()
        if decision == "ALL_A":
            return "all_a"
        if decision == "ALL_B":
            return "all_b"
        return "mix"

    sg.set_entry_point("summarise")
    sg.add_edge("summarise", "analyze")
    sg.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"all_a": "bypass", "all_b": "bypass", "mix": "plan"},
    )
    sg.add_edge("bypass", "finalize")
    sg.add_edge("plan", "patch")
    sg.add_edge("patch", "review")
    sg.add_conditional_edges(
        "review",
        _route_after_review,
        {"finish": "finalize", "retry": "feedback"},
    )
    sg.add_edge("feedback", "patch")
    sg.add_edge("finalize", END)

    sub_app = sg.compile()
    if "_review_iter" not in state:
        state["_review_iter"] = 0
    return sub_app.invoke(state)



def resolve_conflict_new_bypass5_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Compatibility wrapper exporting the numbered resolver name for imports.

    Delegates to ``resolve_conflict_new_bypass_multi_agent_node``.
    """
    return resolve_conflict_new_bypass_multi_agent_node(state)


