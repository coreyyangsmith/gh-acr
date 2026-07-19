"""bj_no_plan: better_judge ablation without planner or reviewer.

Summarise → analyze → (A/B bypass | seed all-merge plan → patch) → finalize.
Uses ``better_judge`` prompt templates. On MIX, every file is merged with no
plan LLM call and no review loop.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bj_no_plan_node"]

_resolver = create_resolver("bj_no_plan")


def resolve_conflict_bj_no_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """better_judge ablation: no planner or reviewer; MIX always merges."""
    return _resolver(state)
