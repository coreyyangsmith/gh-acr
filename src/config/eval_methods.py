"""Centralized definitions for evaluation methods.

This module defines the canonical set of evaluation method identifiers used
throughout the application. It provides type definitions, validation, and
standard orderings for consistent reporting.

Evaluation Methods
------------------
- **base_a**: Baseline that always selects Parent A's version
- **base_b**: Baseline that always selects Parent B's version
- **agent**: Single-turn LLM-based merge resolver
- **bypass7**: Multi-agent with analyzer that can bypass (All A/B) or merge
- **force_mix**: Multi-agent that skips the conflict analyzer and always uses
  the mix (plan → patch → review) path

Usage Patterns
--------------
1. Type checking with `EvalMethod`::

    def process(method: EvalMethod) -> None:
        if method == "bypass7":
            ...

2. Iterating all methods::

    for method in ALL_EVAL_METHODS:
        run_evaluation(method)

3. Consistent ordering in reports::

    df = df.sort_values("method", key=lambda x: [
        DEFAULT_METHOD_ORDER.index(m) for m in x
    ])

Example Usage
-------------
>>> from src.config.eval_methods import EvalMethod, ALL_EVAL_METHODS
>>> def run_method(method: EvalMethod) -> dict:
...     if method not in ALL_EVAL_METHODS:
...         raise ValueError(f"Unknown method: {method}")
...     return {"method": method, "score": 0.95}
"""

from __future__ import annotations

from typing import Final, Literal, get_args


# Type alias describing the allowed evaluation method strings
EvalMethod = Literal[
    "base_a",
    "base_b",
    "agent",
    "bypass7",
    "force_mix",
]
"""Type alias for valid evaluation method identifiers.

Use this for type annotations to ensure only valid methods are passed:

>>> def process(method: EvalMethod) -> None:
...     pass  # Type checker validates method is one of the allowed values
"""


# Canonical list of all supported evaluation methods
ALL_EVAL_METHODS: Final[list[EvalMethod]] = [
    "base_a",
    "base_b",
    "agent",
    "bypass7",
    "force_mix",
]
"""Complete list of supported evaluation methods.

Use this for iteration when you need to process all methods:

>>> for method in ALL_EVAL_METHODS:
...     results[method] = evaluate(method)
"""


# Default ordering to use in charts/reports
DEFAULT_METHOD_ORDER: Final[list[EvalMethod]] = [
    "base_a",
    "base_b",
    "agent",
    "bypass7",
    "force_mix",
]
"""Standard ordering for methods in reports and visualizations.

This ordering places baselines first, then simple agent, then
the multi-agent variant.
"""


def is_valid_method(method: str) -> bool:
    """Check if a string is a valid evaluation method.

    Parameters
    ----------
    method
        The method name to validate.

    Returns
    -------
    bool
        True if method is in ALL_EVAL_METHODS.

    Examples
    --------
    >>> is_valid_method("bypass7")
    True
    >>> is_valid_method("unknown")
    False
    """
    return method in ALL_EVAL_METHODS


def get_method_index(method: EvalMethod) -> int:
    """Get the standard ordering index for a method.

    Parameters
    ----------
    method
        A valid evaluation method.

    Returns
    -------
    int
        The index in DEFAULT_METHOD_ORDER.

    Raises
    ------
    ValueError
        If method is not in the ordering.
    """
    return DEFAULT_METHOD_ORDER.index(method)


__all__ = [
    "EvalMethod",
    "ALL_EVAL_METHODS",
    "DEFAULT_METHOD_ORDER",
    "is_valid_method",
    "get_method_index",
]
