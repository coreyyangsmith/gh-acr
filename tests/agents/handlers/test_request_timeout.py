"""Tests for shared LLM HTTP request-timeout resolution."""

from __future__ import annotations

from src.agents.handlers.request_timeout import (
    DEFAULT_LLM_REQUEST_TIMEOUT_S,
    resolve_llm_request_timeout,
)


def test_default_timeout(monkeypatch):
    monkeypatch.delenv("OPENROUTER_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("GHACR_LLM_REQUEST_TIMEOUT", raising=False)
    assert (
        resolve_llm_request_timeout(specific_env="OPENROUTER_REQUEST_TIMEOUT")
        == DEFAULT_LLM_REQUEST_TIMEOUT_S
    )


def test_specific_env_wins(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REQUEST_TIMEOUT", "123.5")
    monkeypatch.setenv("GHACR_LLM_REQUEST_TIMEOUT", "999")
    assert (
        resolve_llm_request_timeout(specific_env="OPENROUTER_REQUEST_TIMEOUT")
        == 123.5
    )


def test_shared_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_REQUEST_TIMEOUT", raising=False)
    monkeypatch.setenv("GHACR_LLM_REQUEST_TIMEOUT", "450")
    assert resolve_llm_request_timeout(specific_env="OPENAI_REQUEST_TIMEOUT") == 450.0


def test_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REQUEST_TIMEOUT", "nope")
    monkeypatch.delenv("GHACR_LLM_REQUEST_TIMEOUT", raising=False)
    assert (
        resolve_llm_request_timeout(specific_env="OPENROUTER_REQUEST_TIMEOUT")
        == DEFAULT_LLM_REQUEST_TIMEOUT_S
    )


def test_non_positive_falls_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REQUEST_TIMEOUT", "0")
    monkeypatch.setenv("GHACR_LLM_REQUEST_TIMEOUT", "-1")
    assert (
        resolve_llm_request_timeout(specific_env="OPENROUTER_REQUEST_TIMEOUT")
        == DEFAULT_LLM_REQUEST_TIMEOUT_S
    )
