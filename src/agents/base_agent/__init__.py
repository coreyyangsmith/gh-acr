"""Baseline merge resolvers.

Exports two simple nodes:
 - resolve_conflict_base_a_node: picks Parent A for all files
 - resolve_conflict_base_b_node: picks Parent B for all files
"""
from __future__ import annotations

from .base_a import resolve_conflict_base_a_node
from .base_b import resolve_conflict_base_b_node

__all__ = [
    "resolve_conflict_base_a_node",
    "resolve_conflict_base_b_node",
]

