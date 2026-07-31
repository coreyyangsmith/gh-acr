"""Tests for OpenRouterHandler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.handlers.openrouter_handler import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_PROVIDER_ORDER,
    DEFAULT_OPENROUTER_QUANTIZATIONS,
    OpenRouterHandler,
    openrouter_model_family,
    resolve_provider_preferences,
    resolve_provider_routing,
    resolve_quantizations,
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
    monkeypatch.delenv("OPENROUTER_QUANTIZATIONS", raising=False)
    monkeypatch.delenv("OPENROUTER_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("GHACR_LLM_REQUEST_TIMEOUT", raising=False)
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/anthropic/claude-sonnet-4.5")

    assert captured.get("model") == "anthropic/claude-sonnet-4.5"
    assert captured.get("base_url") == DEFAULT_OPENROUTER_BASE_URL
    assert captured.get("api_key") == "or-test"
    assert captured.get("temperature") == 0
    assert captured.get("timeout") == 600.0
    assert "default_headers" not in captured
    assert captured.get("extra_body") == {
        "provider": {
            "order": list(DEFAULT_OPENROUTER_PROVIDER_ORDER),
            "allow_fallbacks": True,
            "quantizations": list(DEFAULT_OPENROUTER_QUANTIZATIONS),
        }
    }


def test_create_honors_request_timeout_env(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_REQUEST_TIMEOUT", "42")
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "off")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/openai/gpt-4o-mini")

    assert captured.get("timeout") == 42.0


def test_create_respects_base_url_and_headers(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://gh-acr.example")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "GH-ACR")
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "off")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/openai/gpt-4o-mini")

    assert captured.get("base_url") == "https://example.test/api/v1"
    assert captured.get("model") == "openai/gpt-4o-mini"
    headers = captured.get("default_headers") or {}
    assert headers.get("HTTP-Referer") == "https://gh-acr.example"
    assert headers.get("X-OpenRouter-Title") == "GH-ACR"
    assert "extra_body" not in captured


def test_http_referer_fallback_env(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("HTTP_REFERER", "https://via-http-referer.example")
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "off")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/anthropic/claude-sonnet-4.5")

    headers = captured.get("default_headers") or {}
    assert headers.get("HTTP-Referer") == "https://via-http-referer.example"


def test_empty_base_url_falls_back_to_default(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "   ")
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "off")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/openai/gpt-4o-mini")

    assert captured.get("base_url") == DEFAULT_OPENROUTER_BASE_URL


def test_passes_max_tokens_from_model_costs(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "off")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create("openrouter/qwen/qwen3-32b")

    assert captured.get("max_tokens") == 16_384


def test_create_uses_resolve_encoder_for_qwen(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "off")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    fake_enc = object()

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=fake_enc,
        ) as resolve,
    ):
        enc, _llm = OpenRouterHandler().create("openrouter/qwen/qwen3-32b")

    resolve.assert_called_once_with("openrouter/qwen/qwen3-32b")
    assert enc is fake_enc


def test_empty_model_id_raises():
    with pytest.raises(ValueError, match="requires a model id"):
        OpenRouterHandler().parse_model_id("openrouter/")


@pytest.mark.parametrize(
    "model,family",
    [
        ("openrouter/openai/gpt-5-nano", "gpt5nano"),
        ("openrouter/meta-llama/llama-3.1-8b-instruct", "llama3"),
        ("openrouter/qwen/qwen3-32b", "qwen3"),
        ("openrouter/anthropic/claude-sonnet-4.5", None),
    ],
)
def test_openrouter_model_family(model, family):
    assert openrouter_model_family(model) == family


def test_resolve_provider_routing_family_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.setenv("OPENROUTER_PROVIDER_LLAMA3", "together,fireworks")
    routing = resolve_provider_routing(
        "openrouter/meta-llama/llama-3.1-8b-instruct"
    )
    assert routing == {
        "order": ["together", "fireworks"],
        "allow_fallbacks": False,
    }


def test_resolve_provider_routing_global_fallback(monkeypatch):
    monkeypatch.delenv("OPENROUTER_PROVIDER_LLAMA3", raising=False)
    monkeypatch.setenv("OPENROUTER_PROVIDER", "deepinfra")
    routing = resolve_provider_routing(
        "openrouter/meta-llama/llama-3.1-8b-instruct"
    )
    assert routing == {"order": ["deepinfra"], "allow_fallbacks": False}


def test_resolve_provider_routing_allow_fallbacks_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_GPT5NANO", "openai")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS_GPT5NANO", "1")
    routing = resolve_provider_routing("openrouter/openai/gpt-5-nano")
    assert routing == {"order": ["openai"], "allow_fallbacks": True}


def test_resolve_provider_routing_defaults_to_groq(monkeypatch):
    for key in (
        "OPENROUTER_PROVIDER",
        "OPENROUTER_PROVIDER_LLAMA3",
        "OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_ALLOW_FALLBACKS_LLAMA3",
    ):
        monkeypatch.delenv(key, raising=False)
    assert resolve_provider_routing(
        "openrouter/meta-llama/llama-3.1-8b-instruct"
    ) == {
        "order": list(DEFAULT_OPENROUTER_PROVIDER_ORDER),
        "allow_fallbacks": True,
    }


def test_resolve_provider_routing_off_disables_order(monkeypatch):
    monkeypatch.delenv("OPENROUTER_PROVIDER_LLAMA3", raising=False)
    monkeypatch.setenv("OPENROUTER_PROVIDER", "off")
    assert (
        resolve_provider_routing("openrouter/meta-llama/llama-3.1-8b-instruct")
        is None
    )


def test_resolve_quantizations_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_QUANTIZATIONS", raising=False)
    assert resolve_quantizations() == list(DEFAULT_OPENROUTER_QUANTIZATIONS)
    assert "fp8" not in resolve_quantizations()


@pytest.mark.parametrize("off_value", ["off", "none", "any", "*", ""])
def test_resolve_quantizations_disabled(monkeypatch, off_value):
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", off_value)
    assert resolve_quantizations() is None


def test_resolve_quantizations_custom(monkeypatch):
    monkeypatch.setenv("OPENROUTER_QUANTIZATIONS", "bf16,fp32")
    assert resolve_quantizations() == ["bf16", "fp32"]


def test_resolve_provider_preferences_merges(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_LLAMA3", "together")
    monkeypatch.delenv("OPENROUTER_QUANTIZATIONS", raising=False)
    prefs = resolve_provider_preferences(
        "openrouter/meta-llama/llama-3.1-8b-instruct"
    )
    assert prefs == {
        "order": ["together"],
        "allow_fallbacks": False,
        "quantizations": list(DEFAULT_OPENROUTER_QUANTIZATIONS),
    }


def test_resolve_provider_preferences_default_groq(monkeypatch):
    for key in (
        "OPENROUTER_PROVIDER",
        "OPENROUTER_PROVIDER_LLAMA3",
        "OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_QUANTIZATIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    prefs = resolve_provider_preferences(
        "openrouter/meta-llama/llama-3.1-8b-instruct"
    )
    assert prefs == {
        "order": ["groq"],
        "allow_fallbacks": True,
        "quantizations": list(DEFAULT_OPENROUTER_QUANTIZATIONS),
    }
    assert "fp8" not in prefs["quantizations"]


def test_create_passes_provider_extra_body(monkeypatch, clear_api_keys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_PROVIDER_LLAMA3", "together")
    monkeypatch.delenv("OPENROUTER_QUANTIZATIONS", raising=False)
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        patch(
            "src.agents.handlers.openrouter_handler.resolve_encoder",
            return_value=None,
        ),
    ):
        OpenRouterHandler().create(
            "openrouter/meta-llama/llama-3.1-8b-instruct"
        )

    assert captured.get("extra_body") == {
        "provider": {
            "order": ["together"],
            "allow_fallbacks": False,
            "quantizations": list(DEFAULT_OPENROUTER_QUANTIZATIONS),
        }
    }
