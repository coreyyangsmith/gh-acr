"""Tests for OpenRouterHandler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.handlers.openrouter_handler import (
    DEFAULT_OPENROUTER_BASE_URL,
    OpenRouterHandler,
)


def test_matches_and_parse():
    h = OpenRouterHandler()
    assert h.matches("openrouter/anthropic/claude-sonnet-4.5")
    assert not h.matches("openai/gpt-4o-mini")
    assert (
        h.parse_model_id("openrouter/anthropic/claude-sonnet-4.5")
        == "anthropic/claude-sonnet-4.5"
    )
    assert (
        h.parse_model_id("openrouter/openai/gpt-4o-mini")
        == "openai/gpt-4o-mini"
    )


def test_missing_api_key(clear_api_keys):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterHandler().create("openrouter/anthropic/claude-sonnet-4.5")


def test_create_default_base_url(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.tiktoken_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/anthropic/claude-sonnet-4.5")

    assert captured.get("model") == "anthropic/claude-sonnet-4.5"
    assert captured.get("base_url") == DEFAULT_OPENROUTER_BASE_URL
    assert captured.get("api_key") == "or-test"
    assert captured.get("temperature") == 0
    assert "default_headers" not in captured


def test_create_respects_base_url_and_headers(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://gh-acr.example")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "GH-ACR")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.tiktoken_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/openai/gpt-4o-mini")

    assert captured.get("base_url") == "https://example.test/api/v1"
    assert captured.get("model") == "openai/gpt-4o-mini"
    headers = captured.get("default_headers") or {}
    assert headers.get("HTTP-Referer") == "https://gh-acr.example"
    assert headers.get("X-OpenRouter-Title") == "GH-ACR"
