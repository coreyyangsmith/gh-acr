"""Tests that RateLimitAndCostHandler records structured llm_calls token counts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents.callbacks import RateLimitAndCostHandler
from src.agents.observability import (
    clear_run_context,
    get_llm_calls,
    set_llm_node,
    set_run_context,
)


@pytest.fixture(autouse=True)
def _reset_context():
    clear_run_context()
    yield
    clear_run_context()


def test_on_llm_end_appends_prompt_and_completion_tokens(monkeypatch: pytest.MonkeyPatch):
    # Avoid real rate-limiter waits
    monkeypatch.setattr(
        "src.agents.callbacks.LimiterRegistry.get",
        lambda **kwargs: MagicMock(acquire=MagicMock(), adjust=MagicMock()),
    )
    monkeypatch.setattr(
        "src.agents.callbacks.count_tokens",
        lambda enc, text: len(str(text).split()) if text else 0,
    )

    set_run_context(eval_method="bypass7", scenario_id="9", model_name="openai/gpt-4o-mini")
    set_llm_node("summarizer_agent")

    handler = RateLimitAndCostHandler(encoder=object(), model_name="openai/gpt-4o-mini")
    run_id = "run-1"
    handler.on_llm_start({}, ["hello world prompt tokens"], run_id=run_id)

    response = SimpleNamespace(
        generations=[[SimpleNamespace(text="one two three four five")]]
    )
    handler.on_llm_end(response, run_id=run_id)

    calls = get_llm_calls()
    assert len(calls) == 1
    assert calls[0]["node"] == "summarizer_agent"
    assert calls[0]["eval_method"] == "bypass7"
    assert calls[0]["scenario_id"] == "9"
    assert calls[0]["prompt_tokens"] == 4  # hello world prompt tokens
    assert calls[0]["completion_tokens"] == 5
    assert calls[0]["total_tokens"] == 9
