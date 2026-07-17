"""LLM inference handlers (provider-specific backends).

Use :func:`resolve_handler` / :func:`create_backend` or the facade
``src.agents.llm_base.get_backend``.
"""

from __future__ import annotations

from .base import BaseLLMHandler
from .groq_handler import GroqHandler
from .local_handler import LocalHandler
from .openai_handler import OpenAIHandler
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
]
