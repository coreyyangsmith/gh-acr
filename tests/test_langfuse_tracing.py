"""Extensive unit tests for LangFuse observability helpers."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.agents.observability import (
    LangfuseLLMWrapper,
    build_langfuse_invoke_config,
    clear_run_context,
    flush_langfuse,
    get_run_context,
    is_langfuse_enabled,
    make_trace_name,
    set_run_context,
)
from src.agents.observability import langfuse_tracing as lt
from src.config.eval_methods import ALL_EVAL_METHODS


@pytest.fixture(autouse=True)
def _reset_context():
    clear_run_context()
    yield
    clear_run_context()


@pytest.fixture
def langfuse_keys(monkeypatch: pytest.MonkeyPatch):
    """Enable LangFuse via credentials (tracing flag left default-on)."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)


@pytest.fixture
def no_langfuse_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)


# ---------------------------------------------------------------------------
# is_langfuse_enabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "public,secret,flag,expected",
    [
        ("pk", "sk", None, True),
        ("pk", "sk", "1", True),
        ("pk", "sk", "true", True),
        ("pk", "sk", "YES", True),
        ("pk", "sk", "0", False),
        ("pk", "sk", "false", False),
        ("pk", "sk", "no", False),
        ("pk", "sk", "off", False),
        ("pk", "sk", "", False),
        ("pk", "", None, False),
        ("", "sk", None, False),
        ("  ", "sk", None, False),
        ("pk", "  ", None, False),
        (None, None, None, False),
    ],
)
def test_is_langfuse_enabled_matrix(
    monkeypatch: pytest.MonkeyPatch,
    public: str | None,
    secret: str | None,
    flag: str | None,
    expected: bool,
):
    if public is None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public)
    if secret is None:
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret)
    if flag is None:
        monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", flag)

    assert is_langfuse_enabled() is expected


# ---------------------------------------------------------------------------
# Trace naming + context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(ALL_EVAL_METHODS))
def test_make_trace_name_for_all_eval_methods(method: str):
    assert make_trace_name(method, "123") == f"{method}-scenario-123"


def test_make_trace_name_uses_context_when_args_omitted():
    set_run_context(eval_method="force_mix", scenario_id="9", model_name="m")
    assert make_trace_name() == "force_mix-scenario-9"


def test_make_trace_name_defaults_to_unknown_without_context():
    assert make_trace_name() == "unknown-scenario-unknown"


def test_make_trace_name_explicit_none_falls_back_to_context():
    set_run_context(eval_method="agent", scenario_id="5")
    assert make_trace_name(None, None) == "agent-scenario-5"


def test_set_run_context_coerces_none_model_to_empty():
    set_run_context(eval_method="agent", scenario_id="1", model_name=None)
    assert get_run_context()["model_name"] == ""


def test_set_run_context_coerces_non_strings():
    set_run_context(eval_method="agent", scenario_id=42, model_name=None)  # type: ignore[arg-type]
    assert get_run_context()["scenario_id"] == "42"


# ---------------------------------------------------------------------------
# build_langfuse_invoke_config
# ---------------------------------------------------------------------------


def test_build_config_disabled_preserves_none(no_langfuse_keys):
    assert build_langfuse_invoke_config(None) is None


def test_build_config_disabled_is_passthrough(no_langfuse_keys):
    existing = {"callbacks": ["keep-me"], "run_name": "original"}
    assert build_langfuse_invoke_config(existing) is existing


def test_build_config_when_handler_import_fails_returns_existing(langfuse_keys):
    existing = {"run_name": "keep"}
    with patch.object(lt, "_import_callback_handler", return_value=None):
        assert build_langfuse_invoke_config(existing) is existing


def test_build_config_when_handler_ctor_raises_returns_existing(langfuse_keys):
    existing = {"run_name": "keep"}

    def _boom():
        raise RuntimeError("ctor failed")

    with patch.object(lt, "_import_callback_handler", return_value=_boom):
        assert build_langfuse_invoke_config(existing) is existing


def test_build_config_enabled_sets_method_trace_name(langfuse_keys):
    fake_handler = MagicMock(name="CallbackHandler")
    fake_cls = MagicMock(return_value=fake_handler)

    set_run_context(eval_method="bypass7", scenario_id="7", model_name="openai/gpt-4o-mini")

    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config({"callbacks": []}, model_name="openai/gpt-4o-mini")

    assert config["run_name"] == "bypass7-scenario-7"
    assert config["metadata"]["langfuse_trace_name"] == "bypass7-scenario-7"
    assert config["metadata"]["eval_method"] == "bypass7"
    assert config["metadata"]["scenario_id"] == "7"
    assert config["metadata"]["model_name"] == "openai/gpt-4o-mini"
    assert "bypass7" in config["tags"]
    assert fake_handler in config["callbacks"]
    fake_cls.assert_called_once_with()


def test_build_config_preserves_existing_callbacks_and_metadata(langfuse_keys):
    prior_cb = object()
    set_run_context(eval_method="agent", scenario_id="1")
    fake_handler = object()
    fake_cls = MagicMock(return_value=fake_handler)

    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config(
            {
                "callbacks": [prior_cb],
                "metadata": {"custom": "x"},
                "tags": ["preexisting"],
            },
            model_name="m",
        )

    assert prior_cb in config["callbacks"]
    assert fake_handler in config["callbacks"]
    assert config["metadata"]["custom"] == "x"
    assert config["metadata"]["eval_method"] == "agent"
    assert "preexisting" in config["tags"]
    assert "agent" in config["tags"]


def test_build_config_does_not_duplicate_method_tag(langfuse_keys):
    set_run_context(eval_method="agent", scenario_id="1")
    fake_cls = MagicMock(return_value=object())
    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config({"tags": ["agent"]}, model_name="m")
    assert config["tags"].count("agent") == 1


def test_build_config_uses_context_model_when_arg_omitted(langfuse_keys):
    set_run_context(eval_method="agent", scenario_id="1", model_name="from-context")
    fake_cls = MagicMock(return_value=object())
    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config({})
    assert config["metadata"]["model_name"] == "from-context"


def test_build_config_prefers_explicit_model_over_context(langfuse_keys):
    set_run_context(eval_method="agent", scenario_id="1", model_name="from-context")
    fake_cls = MagicMock(return_value=object())
    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config({}, model_name="explicit")
    assert config["metadata"]["model_name"] == "explicit"


def test_build_config_unknown_without_context(langfuse_keys):
    fake_cls = MagicMock(return_value=object())
    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config({})
    assert config["run_name"] == "unknown-scenario-unknown"
    assert config["metadata"]["eval_method"] == "unknown"


def test_build_config_non_dict_unconvertible_passthrough(langfuse_keys):
    class Weird:
        pass

    weird = Weird()
    with patch.object(lt, "_import_callback_handler", return_value=MagicMock()):
        assert build_langfuse_invoke_config(weird) is weird


@pytest.mark.parametrize("method", ["base_a", "base_b", "agent", "bypass7", "force_mix"])
def test_build_config_trace_name_includes_each_method(langfuse_keys, method: str):
    set_run_context(eval_method=method, scenario_id="sid")
    fake_cls = MagicMock(return_value=object())
    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        config = build_langfuse_invoke_config({})
    assert config["run_name"] == f"{method}-scenario-sid"
    assert config["metadata"]["langfuse_trace_name"] == f"{method}-scenario-sid"


# ---------------------------------------------------------------------------
# LangfuseLLMWrapper
# ---------------------------------------------------------------------------


def test_langfuse_wrapper_forwards_merged_config(langfuse_keys):
    inner = MagicMock()
    inner.invoke.return_value = MagicMock(content="ok")
    fake_handler = object()
    fake_cls = MagicMock(return_value=fake_handler)

    set_run_context(eval_method="agent", scenario_id="1", model_name="openai/gpt-4o-mini")
    wrapper = LangfuseLLMWrapper(inner, model_name="openai/gpt-4o-mini")

    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        result = wrapper.invoke("hello prompt")

    assert getattr(result, "content", None) == "ok"
    _, kwargs = inner.invoke.call_args
    merged = kwargs["config"]
    assert merged["run_name"] == "agent-scenario-1"
    assert fake_handler in merged["callbacks"]


def test_langfuse_wrapper_disabled_forwards_original_config(no_langfuse_keys):
    inner = MagicMock()
    inner.invoke.return_value = MagicMock(content="ok")
    wrapper = LangfuseLLMWrapper(inner, model_name="m")
    original = {"run_name": "caller"}
    wrapper.invoke("p", config=original)
    _, kwargs = inner.invoke.call_args
    assert kwargs["config"] is original


def test_langfuse_wrapper_with_config_delegates():
    inner = MagicMock()
    inner.with_config.return_value = inner
    wrapper = LangfuseLLMWrapper(inner, model_name="m")
    out = wrapper.with_config({"callbacks": []})
    assert out is wrapper
    inner.with_config.assert_called_once_with({"callbacks": []})


def test_langfuse_wrapper_ainvoke_when_disabled(no_langfuse_keys):
    class Inner:
        def __init__(self):
            self.called = False

        def invoke(self, prompt, config=None):
            self.called = True
            return MagicMock(content="sync-fallback")

    inner = Inner()
    wrapper = LangfuseLLMWrapper(inner, model_name="local:test")
    result = asyncio.run(wrapper.ainvoke("prompt"))
    assert result.content == "sync-fallback"
    assert inner.called is True


def test_langfuse_wrapper_ainvoke_uses_inner_ainvoke(langfuse_keys):
    class Inner:
        def __init__(self):
            self.got_config = None

        async def ainvoke(self, prompt, config=None):
            self.got_config = config
            return MagicMock(content="async-ok")

    inner = Inner()
    wrapper = LangfuseLLMWrapper(inner, model_name="m")
    set_run_context(eval_method="bypass7", scenario_id="2")
    fake_cls = MagicMock(return_value=object())

    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        result = asyncio.run(wrapper.ainvoke("prompt", config={"tags": ["t"]}))

    assert result.content == "async-ok"
    assert inner.got_config["run_name"] == "bypass7-scenario-2"
    assert "t" in inner.got_config["tags"]
    assert "bypass7" in inner.got_config["tags"]


def test_wrapper_method_switches_across_invokes_without_rebuild(langfuse_keys):
    """Cached backend reuses wrapper; context alone must change trace names."""
    inner = MagicMock()
    inner.invoke.return_value = MagicMock(content="ok")
    fake_cls = MagicMock(side_effect=lambda: object())
    wrapper = LangfuseLLMWrapper(inner, model_name="m")

    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        set_run_context(eval_method="agent", scenario_id="1")
        wrapper.invoke("a")
        set_run_context(eval_method="bypass7", scenario_id="1")
        wrapper.invoke("b")
        set_run_context(eval_method="force_mix", scenario_id="1")
        wrapper.invoke("c")

    names = [c.kwargs["config"]["run_name"] for c in inner.invoke.call_args_list]
    assert names == [
        "agent-scenario-1",
        "bypass7-scenario-1",
        "force_mix-scenario-1",
    ]


# ---------------------------------------------------------------------------
# flush_langfuse
# ---------------------------------------------------------------------------


def test_flush_noop_when_disabled(no_langfuse_keys):
    flush_langfuse()  # must not raise


def test_flush_uses_get_client_when_available(langfuse_keys):
    client = MagicMock()

    with patch(
        "langfuse.get_client",
        return_value=client,
        create=True,
    ):
        flush_langfuse()

    client.flush.assert_called_once()


def test_flush_falls_back_to_langfuse_ctor(langfuse_keys):
    client = MagicMock()
    lf_cls = MagicMock(return_value=client)

    with patch("langfuse.get_client", side_effect=RuntimeError("no get_client"), create=True):
        with patch("langfuse.Langfuse", lf_cls, create=True):
            flush_langfuse()

    lf_cls.assert_called_once()
    client.flush.assert_called_once()


def test_flush_swallows_total_failure(langfuse_keys):
    with patch("langfuse.get_client", side_effect=RuntimeError("boom"), create=True):
        with patch("langfuse.Langfuse", side_effect=RuntimeError("boom2"), create=True):
            flush_langfuse()  # must not raise


# ---------------------------------------------------------------------------
# Context isolation across threads
# ---------------------------------------------------------------------------


def test_run_context_is_thread_local():
    results: dict[str, str] = {}

    def _worker(method: str, scenario: str):
        set_run_context(eval_method=method, scenario_id=scenario)
        results[method] = get_run_context()["eval_method"]
        clear_run_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_worker, "agent", "1")
        f2 = pool.submit(_worker, "bypass7", "2")
        f1.result()
        f2.result()

    assert results["agent"] == "agent"
    assert results["bypass7"] == "bypass7"
    assert get_run_context()["eval_method"] == ""


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_observability_package_exports():
    import src.agents.observability as obs

    for name in (
        "LangfuseLLMWrapper",
        "build_langfuse_invoke_config",
        "clear_run_context",
        "flush_langfuse",
        "get_run_context",
        "is_langfuse_enabled",
        "make_trace_name",
        "set_run_context",
    ):
        assert hasattr(obs, name)
