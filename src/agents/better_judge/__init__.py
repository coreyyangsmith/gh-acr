"""Better-judge multi-agent merge resolver package.

Same LangGraph as bypass7 (summarise → analyze → A/B bypass or Mix
plan → patch → review), using the ``better_judge`` prompt templates.
The conflict analyzer / LLM-as-judge step uses a stricter merge-judge prompt.

Flow (LangGraph nodes)
----------------------
1. **summarizer_agent_node**: Generates summaries of A/B diffs
2. **conflict_analyzer_node**: Makes global judgement (All A / All B / Mix)
3. **conflict_agent_node**: Creates per-file merge plan (only for Mix cases)
4. **resolution_agent_node**: Produces patched file(s)
5. **review_agent_node**: Reviews with up to 2 repair loops (only for Mix)

Entry Point
-----------
``resolve_conflict_better_judge_node`` composes the above nodes into an
internal LangGraph executed synchronously.
"""

from __future__ import annotations

from typing import Any, Dict

from ..multi_agent import create_resolver

__all__ = ["resolve_conflict_better_judge_node"]


_resolver = create_resolver("better_judge")


def resolve_conflict_better_judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-agent resolver using better_judge prompts.

    Same workflow as bypass7, with a different conflict-judge prompt.

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
