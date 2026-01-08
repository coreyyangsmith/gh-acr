"""Unified multi-agent merge resolver package.

This module provides a consolidated implementation for all bypass-style
multi-agent merge resolution workflows. Instead of duplicating code across
`bypass/`, `bypass_only/`, and `bypass7/` directories, this module uses a
factory pattern to create agents with different prompt variants.

Workflow Variants
-----------------
- **bypass**: Full multi-agent with summary → analyze → plan → patch → review loop
- **bypass_only**: Lightweight version that only summarizes and analyzes, then bypasses
- **bypass7**: Same as bypass but uses bypass7 prompt templates

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
from .graph_builder import build_bypass_graph, build_bypass_only_graph


# Type alias for supported prompt variants
PromptVariant = Literal["bypass", "bypass_only", "bypass7"]

# Type alias for resolver functions
ResolverFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def create_resolver(variant: PromptVariant = "bypass") -> ResolverFunc:
    """Create a multi-agent resolver function for the specified variant.

    Parameters
    ----------
    variant
        The prompt variant to use:
        - "bypass": Full multi-agent with review loop
        - "bypass_only": Summarize + analyze only, no merge
        - "bypass7": Same as bypass with bypass7 prompts

    Returns
    -------
    ResolverFunc
        A callable that takes a state dict and returns the updated state.

    Examples
    --------
    >>> resolver = create_resolver("bypass7")
    >>> result = resolver({"scenario_id": 123, ...})
    """
    if variant == "bypass_only":
        return build_bypass_only_graph(prompt_variant=variant)
    else:
        # Both "bypass" and "bypass7" use the full graph, just different prompts
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
]



