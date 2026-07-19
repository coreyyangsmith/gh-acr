"""bj_no_summary: better_judge ablation without the LLM summarizer.

Same LangGraph as better_judge (analyze → A/B bypass or Mix plan → patch →
review), but summaries are seeded from raw diffs instead of an LLM call.
Uses ``better_judge`` prompt templates.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bj_no_summary_node"]

_resolver = create_resolver("bj_no_summary")


def resolve_conflict_bj_no_summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """better_judge ablation: skip summarizer; seed summaries from raw diffs."""
    return _resolver(state)
