"""Single-turn LLM-powered merge resolver.

This package wraps the merge_agent implementation under the
``agents.single_agent`` namespace:

    agents/
      ├─ base_agent/     – deterministic baselines
      └─ single_agent/   – LLM-based single-turn resolver
"""
from __future__ import annotations

from .merge_agent import resolve_conflict_agent_node  # noqa: F401

__all__ = ["resolve_conflict_agent_node"]
