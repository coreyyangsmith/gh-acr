"""Lightweight LLM-powered merge resolver (simple agent).

This package wraps the existing *merge_agent* implementation, but places it
under the *agents.simple_agent* namespace to keep the repository structure
organised:

    agents/
      ├─ base_agent/    – deterministic baseline
      └─ simple_agent/  – LLM-based resolver
"""
from __future__ import annotations

from .merge_agent import resolve_conflict_agent_node as resolve_conflict_agent2_node  # noqa: F401

__all__ = ["resolve_conflict_agent2_node"]
