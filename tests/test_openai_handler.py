"""Tests for OpenAIHandler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.handlers.openai_handler import OpenAIHandler


def test_matches_and_parse():
    h = OpenAIHandler()
    assert h.matches("openai/gpt-4o-mini")
    assert not h.matches("openrouter/openai/gpt-4o-mini")
    assert h.parse_model_id("openai/gpt-4o-mini") == "gpt-4o-mini"


def test_missing_api_key(clear_api_keys):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIHandler().create("openai/gpt-4o-mini")


def test_create_passes_model_and_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openai_handler.tiktoken_encoder",
            return_value=None,
        ),
    ):
        OpenAIHandler().create("openai/gpt-4o-mini")

    assert captured.get("model") == "gpt-4o-mini"
    assert captured.get("temperature") == 0
    assert captured.get("api_key") == "sk-test"


def test_gpt5_skips_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openai_handler.tiktoken_encoder",
            return_value=None,
        ),
    ):
        OpenAIHandler().create("openai/gpt-5-nano")

    assert captured.get("model") == "gpt-5-nano"
    assert "temperature" not in captured


def test_create_passes_max_tokens_from_model_costs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openai_handler.tiktoken_encoder",
            return_value="enc",
        ),
    ):
        enc, llm = OpenAIHandler().create("openai/gpt-4o-mini")

    assert enc == "enc"
    assert captured.get("max_tokens") == 16_000


def test_parse_rejects_non_openai_scheme():
    with pytest.raises(ValueError, match="does not match scheme"):
        OpenAIHandler().parse_model_id("openrouter/openai/gpt-4o-mini")
