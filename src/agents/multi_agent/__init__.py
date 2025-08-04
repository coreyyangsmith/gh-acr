"""Multi-agent merge resolver package.

Flow (LangGraph nodes):
1. summarizer_agent_node  – summaries of A/B diffs
2. conflict_agent_node    – high-level merge plan
3. resolution_agent_node  – produces patched file(s)
4. review_agent_node      – final approval / comments

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


# We build a small sub-graph and invoke it inline. This keeps the outer
# pipeline code simple: it treats this as just another *resolver node*.

def resolve_conflict_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Merge-conflict resolver that orchestrates four internal agents."""

    sg = StateGraph(dict)

    sg.add_node("summarise", summarizer_agent_node)
    sg.add_node("plan", conflict_agent_node)
    sg.add_node("patch", resolution_agent_node)
    sg.add_node("review", review_agent_node)

    sg.set_entry_point("summarise")

    sg.add_edge("summarise", "plan")
    sg.add_edge("plan", "patch")
    sg.add_edge("patch", "review")
    sg.add_edge("review", END)

    sub_app = sg.compile()

    # Run synchronously – small number of steps.
    return sub_app.invoke(state)

