"""bj_no_review: better_judge ablation without the review loop.

Summarise → analyze → (A/B bypass | plan → patch) → finalize.
Uses ``better_judge`` prompt templates. Plan and patch run once with no
repair retries.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bj_no_review_node"]

_resolver = create_resolver("bj_no_review")


def resolve_conflict_bj_no_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """better_judge ablation: plan then execute; no review/feedback loop."""
    return _resolver(state)
