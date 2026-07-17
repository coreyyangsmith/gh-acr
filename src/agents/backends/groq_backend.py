"""Groq backend factory (thin re-export of ``GroqHandler``)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..handlers.groq_handler import GroqHandler


def create_groq_backend(model_name: str) -> Tuple[Optional[Any], Any]:
    """Initialize Groq chat backend for model_name (groq:<model>)."""
    return GroqHandler().create(model_name)


__all__ = ["create_groq_backend"]
