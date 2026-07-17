"""Tests for get_backend facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_base import get_backend


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        get_backend("unknown-provider/model")


def test_get_backend_wraps_runnable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeLLM:
        def __init__(self):
            self._config = None

        def with_config(self, config=None):
            self._config = config
            return self

        def invoke(self, prompt, config=None):
            return MagicMock(content="ok")

    fake = FakeLLM()

    with patch(
        "src.agents.llm_base.create_backend",
        return_value=(None, fake),
    ):
        get_backend.cache_clear()
        enc, llm = get_backend("openai/gpt-4o-mini-test-cache-key")

    assert hasattr(llm, "invoke")
    result = llm.invoke("hello")
    assert getattr(result, "content", None) == "ok"
