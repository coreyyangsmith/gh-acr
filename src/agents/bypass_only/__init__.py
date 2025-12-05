"""Bypass-only multi-agent merge resolver package.

This package provides a lightweight bypass workflow that only summarizes and
analyzes conflicts, then selects the best parent without performing an actual
merge. It's useful when you want quick parent selection without LLM-based merging.

Flow (LangGraph nodes)
----------------------
1. **summarizer_agent_node**: Generates summaries of A/B diffs
2. **conflict_analyzer_node**: Makes global judgement (All A / All B)
3. **bypass**: Selects the parent contents based on the decision
4. **finalize**: Computes final diffs

Unlike the full bypass workflow, this variant:
- Does NOT create a merge plan
- Does NOT run resolution or review loops
- Forces a strict A/B decision (no Mix)

Entry Point
-----------
The main entry point `resolve_conflict_bypass_only_multi_agent_node` composes
the above nodes into an internal LangGraph executed synchronously.

Implementation Note
-------------------
This package uses the consolidated implementation from `agents.multi_agent`.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bypass_only_multi_agent_node"]


# Create the resolver using the consolidated implementation
_resolver = create_resolver("bypass_only")


def resolve_conflict_bypass_only_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Bypass-only resolver that selects a parent without merging.

    This is a simplified workflow that:
    1. Summarizes the diffs from both parents
    2. Analyzes to pick either A or B (strict, no Mix)
    3. Selects the parent contents directly
    4. Finalizes with computed diffs

    Parameters
    ----------
    state
        The pipeline state dict containing:
        - ancestor_contents: Dict[str, str]
        - parent_a_contents: Dict[str, str]
        - parent_b_contents: Dict[str, str]
        - diffs_a: Dict[str, str]
        - diffs_b: Dict[str, str]
        - sample_row: Dict with scenario metadata
        - model_name: Optional[str]

    Returns
    -------
    Dict[str, Any]
        Updated state with:
        - resolved_contents: Dict[str, str] (from selected parent)
        - final_diffs: Dict[str, str]
        - bypass_decision: "ALL_A" | "ALL_B"
        - bypass_method: "A" | "B"
        - bypass_only_defaulted: True if analyzer failed to decide
    """
    return _resolver(state)


# Re-export individual nodes for backwards compatibility
from .summarizer_agent import summarizer_agent_node
from .conflict_analyzer import conflict_analyzer_node
