"""Registry mapping model-name schemes to LLM handlers."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .base import BaseLLMHandler
from .groq_handler import GroqHandler
from .local_handler import LocalHandler
from .openai_handler import OpenAIHandler
from .openrouter_handler import OpenRouterHandler

# Order matters only for overlapping prefixes (none today).
_HANDLERS: List[BaseLLMHandler] = [
    OpenAIHandler(),
    OpenRouterHandler(),
    GroqHandler(),
    LocalHandler(),
]


def get_handlers() -> List[BaseLLMHandler]:
    """Return the registered handler instances."""
    return list(_HANDLERS)


def resolve_handler(model_name: str) -> BaseLLMHandler:
    """Pick the handler whose scheme matches *model_name*.

    Raises
    ------
    ValueError
        If no registered handler matches.
    """
    for handler in _HANDLERS:
        if handler.matches(model_name):
            return handler
    raise ValueError(
        f"Unknown model_name scheme: {model_name!r}. "
        "Expected openai/, openrouter/, groq:, or local:"
    )


def create_backend(model_name: str) -> Tuple[Optional[Any], Any]:
    """Resolve handler and return ``(encoder, raw_llm)``."""
    return resolve_handler(model_name).create(model_name)


__all__ = [
    "get_handlers",
    "resolve_handler",
    "create_backend",
]
