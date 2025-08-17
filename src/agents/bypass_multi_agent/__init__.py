"""Bypass multi-agent merge resolver package.

Flow (LangGraph nodes):
1. summarizer_agent_node   – summaries of A/B diffs
2. conflict_analyzer_node  – global judgement: All A / All B / Mix A/B
3. conflict_agent_node     – high-level per-file merge plan (only for Mix)
4. resolution_agent_node   – produces patched file(s)
5. review_agent_node       – review with up to 2 repair loops (only for Mix)

Entry point `resolve_conflict_bypass_multi_agent_node` composes the above into
an internal LangGraph executed synchronously.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph
import difflib

from .summarizer_agent import summarizer_agent_node
from .conflict_analyzer import conflict_analyzer_node
from .conflict_agent import conflict_agent_node
from .resolution_agent import resolution_agent_node
from .review_agent import review_agent_node

__all__ = ["resolve_conflict_bypass_multi_agent_node"]


def _route_after_review(state: Dict[str, Any]) -> str:
    """Route based on review outcomes and retry budget.

    - If all files ACCEPT or retries exhausted -> finish
    - Else -> patch again (send review rationale as feedback)
    """
    max_iters = 2
    iter_no = int(state.get("_review_iter", 0))
    results = state.get("review_results", {}) or {}
    if results and all((v or {}).get("outcome") == "ACCEPT" for v in results.values()):
        return "finish"
    if iter_no >= max_iters:
        return "finish"
    return "retry"


def _prepare_feedback_for_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect ALL reviewer comments for REJECT files and increment retry counter.

    Aggregates across iterations so the resolution agent sees the full history
    of reviewer guidance per file.
    """
    iter_no = int(state.get("_review_iter", 0))
    review_results = state.get("review_results", {}) or {}
    raw_reviews = state.get("reviews", {}) or {}

    # Maintain a history of comments per file across iterations
    history: Dict[str, list[str]] = state.setdefault("review_feedback_history", {})  # type: ignore[assignment]

    for path, data in review_results.items():
        if not isinstance(data, dict):
            continue
        if str(data.get("outcome", "")).upper() == "ACCEPT":
            continue
        rationale = str(data.get("rationale", "")).strip()
        full_text = str(raw_reviews.get(path, "")).strip()
        # Compose an entry that includes both structured rationale and raw review body
        entry_parts = []
        if rationale:
            entry_parts.append(f"Rationale: {rationale}")
        if full_text and full_text != rationale:
            entry_parts.append(f"Raw: {full_text}")
        if not entry_parts:
            entry_parts.append("(no review comments provided)")
        entry = f"Iteration {iter_no}:\n" + "\n".join(entry_parts)
        history.setdefault(path, []).append(entry)

    # Build the feedback for the next resolution pass by concatenating history
    feedback_map: Dict[str, str] = {}
    for path, entries in history.items():
        feedback_map[path] = "\n\n---\n\n".join(entries)

    state["review_feedback"] = feedback_map
    state["_review_iter"] = iter_no + 1
    state["review_feedback_history"] = history
    return state


def resolve_conflict_bypass_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Multi-agent resolver with an early bypass analyzer (ALL_A/ALL_B/MIX)."""

    sg = StateGraph(dict)

    # Core nodes
    sg.add_node("summarise", summarizer_agent_node)
    sg.add_node("analyze", conflict_analyzer_node)
    sg.add_node("plan", conflict_agent_node)
    sg.add_node("patch", resolution_agent_node)
    sg.add_node("review", review_agent_node)

    # Bypass selector: if analyzer returned ALL_A or ALL_B, write resolved contents directly
    def _bypass_select(s: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(s.get("bypass_decision", "MIX")).upper()
        if decision == "ALL_A":
            s["resolved_contents"] = s.get("parent_a_contents", {}) or {}
        elif decision == "ALL_B":
            s["resolved_contents"] = s.get("parent_b_contents", {}) or {}
        s["status"] = "bypassed" if decision in ("ALL_A", "ALL_B") else s.get("status", "")
        return s

    sg.add_node("bypass", _bypass_select)

    # Finalize node to ensure diffs are ready
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

    # Feedback bookkeeping for review loops
    def _feedback_node(s: Dict[str, Any]) -> Dict[str, Any]:
        return _prepare_feedback_for_retry(s)

    sg.add_node("feedback", _feedback_node)

    # Routing after analyzer
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
        {
            "all_a": "bypass",
            "all_b": "bypass",
            "mix": "plan",
        },
    )

    # For bypass cases directly finalize after selecting contents
    sg.add_edge("bypass", "finalize")

    # Normal multi-agent path
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
    sg.add_edge("feedback", "patch")
    sg.add_edge("finalize", END)

    sub_app = sg.compile()

    if "_review_iter" not in state:
        state["_review_iter"] = 0

    return sub_app.invoke(state)

