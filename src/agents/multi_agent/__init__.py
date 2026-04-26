"""Unified multi-agent merge resolver package.

This module provides the consolidated implementation for the bypass7
multi-agent merge resolution workflow, using a factory pattern to create
the agent with the correct prompt variant.

Workflow
--------
- **bypass7**: Full multi-agent with summary → analyze → plan → patch → review loop

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
from .graph_builder import build_bypass_graph, build_force_mix_graph


# Type alias for supported prompt variants
PromptVariant = Literal["bypass7", "force_mix"]

# Type alias for resolver functions
ResolverFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def create_resolver(variant: PromptVariant = "bypass7") -> ResolverFunc:
    """Create a multi-agent resolver function for the given variant.

    Parameters
    ----------
    variant
        The prompt variant to use. Supported values:
        - "bypass7": Full multi-agent with summarise → analyze → plan/bypass → finalize.
        - "force_mix": Skips the conflict analyzer; always routes through the mix
          (plan → patch → review) path. ``bypass_decision`` is hard-set to ``MIX``.

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
    if variant == "force_mix":
        return build_force_mix_graph(prompt_variant="force_mix")
    return build_bypass_graph(prompt_variant=variant)


# Re-export individual node factories for direct use if needed
__all__ = [
    "create_resolver",
    "create_summarizer_node",
    "create_conflict_analyzer_node",
    "create_conflict_agent_node",
    "create_resolution_agent_node",
    "create_review_agent_node",
    "PromptVariant",
    "ResolverFunc",
    "build_force_mix_graph",
]



