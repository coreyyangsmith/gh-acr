"""LLM inference handlers (provider-specific backends).

Use :func:`resolve_handler` / :func:`create_backend` or the facade
``src.agents.llm_base.get_backend``.
"""

from __future__ import annotations

from typing import Any

from .base import BaseLLMHandler
from .groq_handler import GroqHandler, create_groq_backend
from .local_handler import LocalHandler
from .openai_handler import OpenAIHandler, create_openai_backend
from .openrouter_handler import OpenRouterHandler
from .registry import create_backend, get_handlers, resolve_handler

__all__ = [
    "BaseLLMHandler",
    "OpenAIHandler",
    "GroqHandler",
    "OpenRouterHandler",
    "LocalHandler",
    "resolve_handler",
    "create_backend",
    "get_handlers",
    "create_openai_backend",
    "create_groq_backend",
    "create_local_backend",
]


def __getattr__(name: str) -> Any:
    """Lazy-load local backend helpers (avoids importing torch at package import)."""
    if name == "create_local_backend":
        from .local_backend import create_local_backend

        return create_local_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
