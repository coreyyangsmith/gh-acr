"""OpenAI backend factory (thin re-export of ``OpenAIHandler``)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..handlers.openai_handler import OpenAIHandler


def create_openai_backend(model_name: str) -> Tuple[Optional[Any], Any]:
    """Initialize OpenAI chat backend for model_name (openai/<model>)."""
    return OpenAIHandler().create(model_name)


__all__ = ["create_openai_backend"]
