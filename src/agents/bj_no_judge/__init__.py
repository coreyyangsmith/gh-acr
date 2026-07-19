"""bj_no_judge: better_judge ablation without the conflict analyzer.

Force-mix topology (summarise → MIX marker → plan → patch → review) using
``better_judge`` prompt templates. ``bypass_decision`` is hard-set to MIX.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bj_no_judge_node"]

_resolver = create_resolver("bj_no_judge")


def resolve_conflict_bj_no_judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """better_judge ablation: skip analyzer; always take the mix path."""
    return _resolver(state)
