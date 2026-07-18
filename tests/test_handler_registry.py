"""Tests for handler scheme registry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.handlers import (
    GroqHandler,
    LocalHandler,
    OpenAIHandler,
    OpenRouterHandler,
    get_handlers,
    resolve_handler,
)
from src.agents.handlers.registry import create_backend


@pytest.mark.parametrize(
    "model_name,handler_cls",
    [
        ("openai/gpt-4o-mini", OpenAIHandler),
        ("openrouter/anthropic/claude-sonnet-4.5", OpenRouterHandler),
        ("openrouter/openai/gpt-4o-mini", OpenRouterHandler),
        ("groq:qwen/qwen3-32b", GroqHandler),
        ("local:meta-llama/Llama-3.1-8B-Instruct", LocalHandler),
    ],
)
def test_resolve_handler_schemes(model_name, handler_cls):
    handler = resolve_handler(model_name)
    assert isinstance(handler, handler_cls)


def test_openrouter_not_confused_with_openai():
    """openrouter/openai/... must not resolve to OpenAIHandler."""
    assert isinstance(
        resolve_handler("openrouter/openai/gpt-4o-mini"), OpenRouterHandler
    )


def test_get_handlers_covers_all_providers():
    classes = {type(h) for h in get_handlers()}
    assert classes == {
        OpenAIHandler,
        OpenRouterHandler,
        GroqHandler,
        LocalHandler,
    }


def test_resolve_handler_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        resolve_handler("azure/gpt-4")


@pytest.mark.parametrize(
    "bad",
    ["", "gpt-4o-mini", "http://x", "OPENAI/gpt-4o-mini", "OpenRouter/x"],
)
def test_resolve_handler_rejects_misc_schemes(bad):
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        resolve_handler(bad)


def test_create_backend_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        create_backend("foobar")


def test_create_backend_delegates_to_handler_create():
    sentinel = (None, object())
    with patch.object(OpenAIHandler, "create", return_value=sentinel) as mock:
        result = create_backend("openai/gpt-4o-mini")
    mock.assert_called_once_with("openai/gpt-4o-mini")
    assert result is sentinel
