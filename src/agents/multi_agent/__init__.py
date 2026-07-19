"""Unified multi-agent merge resolver package.

This module provides the consolidated implementation for the multi-agent
merge resolution workflow, using a factory pattern to create the agent
with the correct prompt variant and graph topology.

Workflow
--------
- **bypass7**: Full multi-agent with summary → analyze → plan → patch → review loop
- **better_judge**: Same graph as bypass7 with a stricter conflict-judge prompt
- **force_mix**: Skips analyzer; always MIX (plan → patch → review)
- **bj_no_summary / bj_no_judge / bj_no_plan / bj_no_review**: better_judge ablations

Usage
-----
>>> from src.agents.multi_agent import create_resolver
>>> resolver = create_resolver("bypass7")
>>> result = resolver(state)

Architecture
------------
All variants share the same core node implementations:
- `summarizer_agent_node`: Generates summaries of A/B diffs
- `conflict_analyzer_node`: Makes global judgement (All A / All B / Mix)
- `conflict_agent_node`: Creates per-file merge plan (for Mix cases)
- `resolution_agent_node`: Produces merged files
- `review_agent_node`: Reviews with optional repair loops

The only difference between variants is:
1. Which prompt templates are used (determined by prompt_variant)
2. Which nodes are included in the workflow graph
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal

from .nodes import (
    create_summarizer_node,
    create_conflict_analyzer_node,
    create_conflict_agent_node,
    create_resolution_agent_node,
    create_review_agent_node,
)
from .graph_builder import (
    BypassGraphConfig,
    build_bypass_graph,
    build_configured_graph,
    build_force_mix_graph,
)


# Type alias for prompt directories (used by _load_prompt)
PromptVariant = Literal["bypass7", "better_judge", "force_mix"]

# Type alias for all create_resolver variant names (including ablations)
ResolverVariant = Literal[
    "bypass7",
    "better_judge",
    "force_mix",
    "bj_no_summary",
    "bj_no_judge",
    "bj_no_plan",
    "bj_no_review",
]

# Type alias for resolver functions
ResolverFunc = Callable[[Dict[str, Any]], Dict[str, Any]]

# Ablation configs: always use better_judge prompts; topology flags differ
_BJ_ABLATION_CONFIGS: Dict[str, BypassGraphConfig] = {
    "bj_no_summary": BypassGraphConfig(
        prompt_variant="better_judge",
        include_summarizer=False,
        include_analyzer=True,
        include_planner=True,
        include_reviewer=True,
    ),
    "bj_no_judge": BypassGraphConfig(
        prompt_variant="better_judge",
        include_summarizer=True,
        include_analyzer=False,
        include_planner=True,
        include_reviewer=True,
    ),
    "bj_no_plan": BypassGraphConfig(
        prompt_variant="better_judge",
        include_summarizer=True,
        include_analyzer=True,
        include_planner=False,
        include_reviewer=False,
    ),
    "bj_no_review": BypassGraphConfig(
        prompt_variant="better_judge",
        include_summarizer=True,
        include_analyzer=True,
        include_planner=True,
        include_reviewer=False,
    ),
}


def create_resolver(variant: ResolverVariant = "bypass7") -> ResolverFunc:
    """Create a multi-agent resolver function for the given variant.

    Parameters
    ----------
    variant
        The resolver variant to use. Supported values:
        - "bypass7": Full multi-agent with summarise → analyze → plan/bypass → finalize.
        - "better_judge": Same graph as bypass7 with a stricter conflict-judge prompt.
        - "force_mix": Skips the conflict analyzer; always routes through the mix
          (plan → patch → review) path. ``bypass_decision`` is hard-set to ``MIX``.
        - "bj_no_summary": better_judge ablation; raw diffs seed summaries (no LLM summarizer).
        - "bj_no_judge": better_judge ablation; force MIX (no analyzer), like force_mix.
        - "bj_no_plan": better_judge ablation; no planner or reviewer; MIX always merges.
        - "bj_no_review": better_judge ablation; plan → patch with no review loop.

    Returns
    -------
    ResolverFunc
        A callable that takes a state dict and returns the updated state.

    Examples
    --------
    >>> resolver = create_resolver("bypass7")
    >>> result = resolver({"scenario_id": 123, ...})
    >>> force_resolver = create_resolver("force_mix")
    >>> result = force_resolver({"scenario_id": 456, ...})
    """
    if variant in _BJ_ABLATION_CONFIGS:
        return build_configured_graph(_BJ_ABLATION_CONFIGS[variant])
    if variant == "force_mix":
        return build_force_mix_graph(prompt_variant="force_mix")
    if variant in ("bypass7", "better_judge"):
        return build_bypass_graph(prompt_variant=variant)
    raise ValueError(
        f"Unknown resolver variant {variant!r}; expected one of "
        f"bypass7, better_judge, force_mix, bj_no_summary, bj_no_judge, "
        f"bj_no_plan, bj_no_review"
    )


# Re-export individual node factories for direct use if needed
__all__ = [
    "create_resolver",
    "create_summarizer_node",
    "create_conflict_analyzer_node",
    "create_conflict_agent_node",
    "create_resolution_agent_node",
    "create_review_agent_node",
    "PromptVariant",
    "ResolverVariant",
    "ResolverFunc",
    "BypassGraphConfig",
    "build_configured_graph",
    "build_force_mix_graph",
    "build_bypass_graph",
]
