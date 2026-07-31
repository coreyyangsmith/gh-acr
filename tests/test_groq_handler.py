"""Tests for GroqHandler."""

from __future__ import annotations

import pytest

from src.agents.handlers.groq_handler import GroqHandler


def test_matches_and_parse():
    h = GroqHandler()
    assert h.matches("groq:qwen/qwen3-32b")
    assert not h.matches("openai/gpt-4o-mini")
    assert h.parse_model_id("groq:qwen/qwen3-32b") == "qwen/qwen3-32b"
    assert h.parse_model_id("groq:llama-3.1-8b-instant") == "llama-3.1-8b-instant"


def test_missing_api_key(clear_api_keys):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqHandler().create("groq:llama-3.1-8b-instant")


def test_create_passes_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("GROQ_REQUEST_TIMEOUT", raising=False)
    captured: dict = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    from types import ModuleType

    fake_mod = ModuleType("langchain_groq")
    fake_mod.ChatGroq = FakeChatGroq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_groq", fake_mod)

    enc, llm = GroqHandler().create("groq:llama-3.1-8b-instant")

    assert enc is None
    assert llm is not None
    assert captured.get("model") == "llama-3.1-8b-instant"
    assert captured.get("temperature") == 0
    assert captured.get("groq_api_key") == "gsk-test"
    # MODEL_COSTS entry has output_limit 128_000
    assert captured.get("max_tokens") == 128_000
    # Default bounded timeout so hung calls become retryable
    assert captured.get("request_timeout") == 120.0
    # Groq API has no precision / quantization / dtype parameter to set.
    assert "dtype" not in captured
    assert "quantization" not in captured
    assert "precision" not in captured
    assert "model_kwargs" not in captured


def test_create_honors_request_timeout_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_REQUEST_TIMEOUT", "45.5")
    captured: dict = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    from types import ModuleType

    fake_mod = ModuleType("langchain_groq")
    fake_mod.ChatGroq = FakeChatGroq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_groq", fake_mod)

    GroqHandler().create("groq:llama-3.1-8b-instant")
    assert captured.get("request_timeout") == 45.5


def test_parse_rejects_wrong_scheme():
    with pytest.raises(ValueError, match="does not match scheme"):
        GroqHandler().parse_model_id("openai/gpt-4o-mini")


def test_empty_model_id_raises():
    with pytest.raises(ValueError, match="requires a model id"):
        GroqHandler().parse_model_id("groq:")
