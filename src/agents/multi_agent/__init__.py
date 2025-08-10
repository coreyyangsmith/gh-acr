"""Multi-agent merge resolver package.

Flow (LangGraph nodes):
1. summarizer_agent_node  – summaries of A/B diffs
2. conflict_agent_node    – high-level merge plan
3. resolution_agent_node  – produces patched file(s)
4. review_agent_node      – final approval / comments, with up to 3 repair loops

The entry-point callable `resolve_conflict_multi_agent_node` composes the above
nodes into a *mini* LangGraph executed synchronously.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .summarizer_agent import summarizer_agent_node
from .conflict_agent import conflict_agent_node
from .resolution_agent import resolution_agent_node
from .review_agent import review_agent_node

__all__ = ["resolve_conflict_multi_agent_node"]


def _route_after_review(state: Dict[str, Any]) -> str:
    """Route based on review outcomes and retry budget.

    - If all files ACCEPT or retries exhausted -> finish
    - Else -> patch again (send review rationale as feedback)
    """
    max_iters = 3
    iter_no = int(state.get("_review_iter", 0))
    results = state.get("review_results", {}) or {}
    if results and all((v or {}).get("outcome") == "ACCEPT" for v in results.values()):
        return "finish"
    if iter_no >= max_iters:
        return "finish"
    return "retry"


def _prepare_feedback_for_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect rationale by file and increment retry counter."""
    review_results = state.get("review_results", {}) or {}
    feedback_map: Dict[str, str] = {}
    for path, data in review_results.items():
        if not isinstance(data, dict):
            continue
        if data.get("outcome") == "ACCEPT":
            continue
        rationale = str(data.get("rationale", "")).strip()
        if rationale:
            feedback_map[path] = rationale
    state["review_feedback"] = feedback_map
    state["_review_iter"] = int(state.get("_review_iter", 0)) + 1
    return state


# We build a small sub-graph and invoke it inline. This keeps the outer
# pipeline code simple: it treats this as just another *resolver node*.

def resolve_conflict_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Merge-conflict resolver that orchestrates four internal agents with review/repair loop."""

    sg = StateGraph(dict)

    sg.add_node("summarise", summarizer_agent_node)
    sg.add_node("plan", conflict_agent_node)
    sg.add_node("patch", resolution_agent_node)
    sg.add_node("review", review_agent_node)

    # Helper node for feedback bookkeeping
    def _feedback_node(s: Dict[str, Any]) -> Dict[str, Any]:
        return _prepare_feedback_for_retry(s)

    sg.add_node("feedback", _feedback_node)

    sg.set_entry_point("summarise")

    sg.add_edge("summarise", "plan")
    sg.add_edge("plan", "patch")
    sg.add_edge("patch", "review")

    # Conditional after review: either finish or retry patch with feedback
    sg.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "finish": END,
            "retry": "feedback",
        },
    )
    sg.add_edge("feedback", "patch")

    sub_app = sg.compile()

    # Initialize counter
    if "_review_iter" not in state:
        state["_review_iter"] = 0

    return sub_app.invoke(state)

