"""Tests for handler scheme registry."""

from __future__ import annotations

import pytest

from src.agents.handlers import (
    GroqHandler,
    LocalHandler,
    OpenAIHandler,
    OpenRouterHandler,
    resolve_handler,
)
from src.agents.handlers.registry import create_backend


@pytest.mark.parametrize(
    "model_name,handler_cls",
    [
        ("openai/gpt-4o-mini", OpenAIHandler),
        ("openrouter/anthropic/claude-sonnet-4.5", OpenRouterHandler),
        ("groq:qwen/qwen3-32b", GroqHandler),
        ("local:meta-llama/Llama-3.1-8B-Instruct", LocalHandler),
    ],
)
def test_resolve_handler_schemes(model_name, handler_cls):
    handler = resolve_handler(model_name)
    assert isinstance(handler, handler_cls)


def test_resolve_handler_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        resolve_handler("azure/gpt-4")


def test_create_backend_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        create_backend("foobar")
