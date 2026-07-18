"""Tests for get_backend facade."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_base import _ThreadSafeLLMWrapper, get_backend
from src.agents.observability import LangfuseLLMWrapper, clear_run_context, set_run_context
from src.agents.truncation_wrapper import TruncatingLLMWrapper


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown model_name scheme"):
        get_backend("unknown-provider/model")


def test_get_backend_wraps_runnable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

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


def test_get_backend_includes_langfuse_wrapper(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    class FakeLLM:
        def __init__(self):
            self.last_config = None

        def with_config(self, config=None):
            return self

        def invoke(self, prompt, config=None):
            self.last_config = config
            return MagicMock(content="ok")

    fake = FakeLLM()
    fake_handler = object()
    fake_cls = MagicMock(return_value=fake_handler)

    with (
        patch("src.agents.llm_base.create_backend", return_value=(None, fake)),
        patch(
            "src.agents.observability.langfuse_tracing._import_callback_handler",
            return_value=fake_cls,
        ),
    ):
        get_backend.cache_clear()
        _, llm = get_backend("openai/gpt-4o-mini-langfuse-cache-key")

        # Unwrap ThreadSafe -> LangfuseLLMWrapper
        assert isinstance(llm._inner, LangfuseLLMWrapper)

        set_run_context(
            eval_method="agent",
            scenario_id="55",
            model_name="openai/gpt-4o-mini",
        )
        try:
            llm.invoke("trace me")
        finally:
            clear_run_context()

    assert fake.last_config is not None
    assert fake.last_config["run_name"] == "agent-scenario-55"
    assert fake_handler in fake.last_config["callbacks"]


def test_get_backend_raises_when_create_returns_none_llm():
    with patch(
        "src.agents.llm_base.create_backend",
        return_value=(None, None),
    ):
        get_backend.cache_clear()
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            get_backend("openai/null-llm-cache-key")


def test_get_backend_caches_by_model_name(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    class FakeLLM:
        def with_config(self, config=None):
            return self

        def invoke(self, prompt, config=None):
            return MagicMock(content="ok")

    calls = {"n": 0}

    def _create(name):
        calls["n"] += 1
        return None, FakeLLM()

    with patch("src.agents.llm_base.create_backend", side_effect=_create):
        get_backend.cache_clear()
        a = get_backend("openai/cache-me-once")
        b = get_backend("openai/cache-me-once")
        c = get_backend("openai/cache-me-twice")

    assert a is b
    assert a is not c
    assert calls["n"] == 2


def test_get_backend_wrap_stack_includes_truncation(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    class FakeLLM:
        def with_config(self, config=None):
            return self

        def invoke(self, prompt, config=None):
            return MagicMock(content="ok")

    with patch(
        "src.agents.llm_base.create_backend",
        return_value=(None, FakeLLM()),
    ):
        get_backend.cache_clear()
        _, llm = get_backend("openai/wrap-order-key")

    assert isinstance(llm, _ThreadSafeLLMWrapper)
    seen = []
    cur = llm
    for _ in range(6):
        seen.append(type(cur).__name__)
        if isinstance(cur, TruncatingLLMWrapper):
            break
        cur = getattr(cur, "_inner", None)
        if cur is None:
            break
    assert "TruncatingLLMWrapper" in seen
    assert "_ThreadSafeLLMWrapper" in seen


def test_thread_safe_wrapper_serializes_invoke():
    class SlowLLM:
        def __init__(self):
            self.depth = 0
            self.max_depth = 0

        def invoke(self, prompt, config=None):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            self.depth -= 1
            return MagicMock(content=str(prompt))

    slow = SlowLLM()
    wrapped = _ThreadSafeLLMWrapper(slow)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: wrapped.invoke(f"p{i}"), range(40)))
    assert slow.max_depth == 1
