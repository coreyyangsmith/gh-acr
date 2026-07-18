"""Unit tests for BaseLLMHandler shared behaviour."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import pytest

from src.agents.handlers.base import BaseLLMHandler
from src.agents.handlers.openai_handler import OpenAIHandler


class _StubHandler(BaseLLMHandler):
    scheme = "stub"
    separator = "/"
    api_key_env = "STUB_API_KEY"

    def create(self, model_name: str) -> Tuple[Optional[Any], Any]:
        return None, object()


def test_matches_rejects_empty_and_partial_prefix():
    h = OpenAIHandler()
    assert not h.matches("")
    assert not h.matches("openai")
    assert not h.matches("openai:")
    assert h.matches("openai/gpt-4o-mini")


def test_parse_model_id_wrong_scheme():
    h = OpenAIHandler()
    with pytest.raises(ValueError, match="does not match scheme"):
        h.parse_model_id("groq:llama")


def test_parse_model_id_empty():
    h = OpenAIHandler()
    with pytest.raises(ValueError, match="requires a model id"):
        h.parse_model_id("openai/")


def test_require_api_key_success(monkeypatch):
    monkeypatch.setenv("STUB_API_KEY", "secret")
    assert _StubHandler().require_api_key() == "secret"


def test_require_api_key_missing(monkeypatch):
    monkeypatch.delenv("STUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="STUB_API_KEY"):
        _StubHandler().require_api_key()


def test_require_api_key_without_env_attr():
    class NoKeyHandler(BaseLLMHandler):
        scheme = "x"
        separator = "/"
        api_key_env = ""

        def create(self, model_name: str):
            return None, None

    with pytest.raises(RuntimeError, match="does not define api_key_env"):
        NoKeyHandler().require_api_key()
