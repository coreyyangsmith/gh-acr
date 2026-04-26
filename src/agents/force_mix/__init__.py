"""Force Mix multi-agent merge resolver package.

This package provides a multi-agent merge resolution workflow that **always**
takes the mix (plan → patch → review) path.  The conflict analyzer / LLM-as-judge
step is skipped entirely: ``bypass_decision`` and ``bypass_method`` are
hard-set to ``"MIX"`` before the planner runs, so downstream reporting is
fully compatible with the bypass7 schema.

Flow (LangGraph nodes)
----------------------
1. **summarizer_agent_node**: Generates summaries of A/B diffs
2. **force_mix_marker**: Sets bypass_decision=MIX without LLM call (no analyzer)
3. **conflict_agent_node**: Creates per-file merge plan
4. **resolution_agent_node**: Produces patched file(s)
5. **review_agent_node**: Reviews with up to 2 repair loops

Entry Point
-----------
The main entry point ``resolve_conflict_force_mix_node`` composes the above
nodes into an internal LangGraph executed synchronously.

Differences from bypass7
-------------------------
- The ``analyze`` (conflict analyzer) node is completely absent.
- The ``bypass`` (wholesale parent selection) node is completely absent.
- Uses prompts from ``prompts/force_mix/`` instead of ``prompts/bypass7/``.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_force_mix_node"]


# Build the resolver once at import time using the force_mix prompt variant
_resolver = create_resolver("force_mix")


def resolve_conflict_force_mix_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-agent resolver that always uses the mix (plan/patch/review) path.

    The conflict analyzer step is skipped; ``bypass_decision`` is hard-set to
    ``"MIX"`` so all scenarios go through the full planning and patching loop.

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
        - bypass_decision: "MIX" (always)
        - bypass_method: "MIX" (always)
        - bypass_analyzer_output: "[force_mix] conflict analyzer skipped"
        - resolution_history: Dict[str, list] (per-file resolution attempts)
        - review_history: Dict[str, list] (per-file review iterations)
    """
    return _resolver(state)
