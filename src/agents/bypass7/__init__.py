"""Bypass7 multi-agent merge resolver package.

This package provides the same multi-agent workflow as the standard bypass
package, but uses the bypass7 prompt templates for potentially different
behavior.

Flow (LangGraph nodes)
----------------------
1. **summarizer_agent_node**: Generates summaries of A/B diffs
2. **conflict_analyzer_node**: Makes global judgement (All A / All B / Mix)
3. **conflict_agent_node**: Creates per-file merge plan (only for Mix cases)
4. **resolution_agent_node**: Produces patched file(s)
5. **review_agent_node**: Reviews with up to 2 repair loops (only for Mix)

Entry Point
-----------
The main entry point `resolve_conflict_bypass7_multi_agent_node` composes
the above nodes into an internal LangGraph executed synchronously.

Differences from Standard Bypass
--------------------------------
- Uses prompts from `prompts/bypass7/` instead of `prompts/bypass/`
- May produce different analysis and merge behaviors based on prompt tuning

Implementation Note
-------------------
This package uses the consolidated implementation from `agents.multi_agent`.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_bypass7_multi_agent_node"]


# Create the resolver using the consolidated implementation with bypass7 prompts
_resolver = create_resolver("bypass7")


def resolve_conflict_bypass7_multi_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-agent resolver using bypass7 prompts.

    Same workflow as the standard bypass multi-agent, but uses the bypass7
    prompt templates which may be tuned for different behavior.

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
        - resolution_history: Dict[str, list] (per-file resolution attempts)
        - review_history: Dict[str, list] (per-file review iterations)
    """
    return _resolver(state)


# Re-export individual nodes for backwards compatibility
from .summarizer_agent import summarizer_agent_node
from .conflict_analyzer import conflict_analyzer_node
from .conflict_agent import conflict_agent_node
from .resolution_agent import resolution_agent_node
from .review_agent import review_agent_node
