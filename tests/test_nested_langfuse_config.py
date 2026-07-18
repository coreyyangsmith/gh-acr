"""Tests that nested multi-agent graph invokes carry method-named LangFuse config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.multi_agent import graph_builder as gb
from src.agents.observability import clear_run_context, set_run_context
from src.agents.observability import langfuse_tracing as lt


@pytest.fixture(autouse=True)
def _reset_context():
    clear_run_context()
    yield
    clear_run_context()


@pytest.mark.parametrize(
    "builder_name,prompt_variant,method",
    [
        ("build_bypass_graph", "bypass7", "bypass7"),
        ("build_force_mix_graph", "force_mix", "force_mix"),
    ],
)
def test_nested_invoke_includes_method_in_langfuse_trace_name(
    builder_name: str,
    prompt_variant: str,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)

    fake_handler = object()
    fake_cls = MagicMock(return_value=fake_handler)
    captured: dict = {}

    mock_sub_app = MagicMock()
    mock_sub_app.invoke.side_effect = lambda state, config=None: (
        captured.update(config or {}),
        state,
    )[1]

    mock_sg = MagicMock()
    mock_sg.add_node = MagicMock()
    mock_sg.add_edge = MagicMock()
    mock_sg.add_conditional_edges = MagicMock()
    mock_sg.set_entry_point = MagicMock()
    mock_sg.compile.return_value = mock_sub_app

    # Stub node factories so building the graph does not load prompts / backends
    noop = lambda state: state  # noqa: E731
    with patch.object(lt, "_import_callback_handler", return_value=fake_cls):
        set_run_context(eval_method=method, scenario_id="42", model_name="m")
        with patch.object(gb, "StateGraph", return_value=mock_sg):
            with patch.object(gb, "create_summarizer_node", return_value=noop):
                with patch.object(gb, "create_conflict_analyzer_node", return_value=noop):
                    with patch.object(gb, "create_conflict_agent_node", return_value=noop):
                        with patch.object(gb, "create_resolution_agent_node", return_value=noop):
                            with patch.object(gb, "create_review_agent_node", return_value=noop):
                                builder = getattr(gb, builder_name)
                                resolver = builder(prompt_variant=prompt_variant)
                                resolver({"scenario_id": "42", "_review_iter": 0})

    expected = f"{method}-scenario-42"
    assert captured["run_name"] == expected
    assert captured["metadata"]["langfuse_trace_name"] == expected
    assert captured["metadata"]["eval_method"] == method
    assert method in captured["tags"]
    assert fake_handler in captured["callbacks"]
    assert fake_cls.call_count == 1
    mock_sub_app.invoke.assert_called_once()


def test_nested_langfuse_config_passthrough_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    set_run_context(eval_method="bypass7", scenario_id="7")
    cfg = gb._nested_langfuse_config()
    assert cfg["run_name"] == "bypass7-scenario-7"
    assert "callbacks" not in cfg or not cfg.get("callbacks")
