from __future__ import annotations

"""Centralized definitions for evaluation methods.

This module defines the canonical set of evaluation method identifiers used
throughout the application, along with common orderings for reporting.
"""

from typing import Final, Literal


# Public type alias describing the allowed evaluation method strings
EvalMethod = Literal[
    "base_a",
    "base_b",
    "agent",
    "multi",
    "bypass",
    "bypass2",
    "bypass3",
    "bypass4",
    "bypass5",
    "bypass6",
    "bypass7",
    "bypass8",
    "bypass_only",
    "bypass_only2",
    "dynamic",
]


# Canonical list of all supported evaluation methods
ALL_EVAL_METHODS: Final[list[EvalMethod]] = [
    "base_a",
    "base_b",
    "agent",
    "multi",
    "bypass",
    "bypass2",
    "bypass3",
    "bypass4",
    "bypass5",
    "bypass6",
    "bypass7",
    "bypass8",
    "bypass_only",
    "bypass_only2",
    "dynamic",
]


# Default ordering to use in charts/reports
DEFAULT_METHOD_ORDER: Final[list[EvalMethod]] = [
    "base_a",
    "base_b",
    "agent",
    "multi",
    "bypass",
    "bypass2",
    "bypass3",
    "bypass4",
    "bypass5",
    "bypass6",
    "bypass7",
    "bypass8",
    "bypass_only",
    "bypass_only2",
    "dynamic",
]
