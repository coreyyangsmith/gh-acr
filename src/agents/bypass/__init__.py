"""Bypass multi-agent merge resolver package.

This package provides a multi-agent workflow for resolving Git merge conflicts
with an early bypass analyzer that can select All A / All B / Mix resolution.

Flow (LangGraph nodes)
----------------------
1. **summarizer_agent_node**: Generates summaries of A/B diffs
2. **conflict_analyzer_node**: Makes global judgement (All A / All B / Mix)
3. **conflict_agent_node**: Creates per-file merge plan (only for Mix cases)
4. **resolution_agent_node**: Produces patched file(s)
5. **review_agent_node**: Reviews with up to 2 repair loops (only for Mix)

Entry Point
-----------
The main entry point `resolve_conflict_bypass_multi_agent_node` composes
the above nodes into an internal LangGraph executed synchronously.

Implementation Note
-------------------
This package uses the consolidated implementation from `agents.multi_agent`.
The individual agent files (`summarizer_agent.py`, `conflict_analyzer.py`, etc.)
are maintained for backwards compatibility but delegate to the shared implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bypass_multi_agent_node"]


# Create the resolver using the consolidated implementation
_resolver = create_resolver("bypass")


def resolve_conflict_bypass_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-agent resolver with an early bypass analyzer (ALL_A/ALL_B/MIX).

    This is the main entry point for the bypass multi-agent workflow. It:
    1. Summarizes the diffs from both parents
    2. Analyzes whether to bypass (select one parent) or merge
    3. For Mix cases: plans, patches, and reviews with retry loop
    4. Finalizes the resolved contents and diffs

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
        - resolved_contents: Dict[str, str]
        - final_diffs: Dict[str, str]
        - bypass_decision: "ALL_A" | "ALL_B" | "MIX"
        - bypass_method: "A" | "B" | "MIX"
        - summaries, reviews, etc.
    """
    return _resolver(state)


# Re-export individual nodes for backwards compatibility
# These are imported by external code that expects them in this package
from .summarizer_agent import summarizer_agent_node
from .conflict_analyzer import conflict_analyzer_node
from .conflict_agent import conflict_agent_node
from .resolution_agent import resolution_agent_node
from .review_agent import review_agent_node
