"""Tests for RateLimitAndCostHandler callback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.helpers import FakeEncoder

from src.agents.callbacks import RateLimitAndCostHandler, _extract_usage_tokens
from src.utils.rate_limiter import LimiterRegistry


def setup_function():
    LimiterRegistry._instances.clear()


def teardown_function():
    LimiterRegistry._instances.clear()


def test_on_llm_start_acquires_and_reserves():
    handler = RateLimitAndCostHandler(encoder=FakeEncoder(), model_name="openai/gpt-4o-mini")
    with patch.object(handler._limiter, "acquire", return_value=0.0) as acquire:
        handler.on_llm_start({}, ["hello world"], run_id="r1")
    acquire.assert_called_once()
    assert "r1" in handler._reservations
    assert handler._reservations["r1"]["prompt_tokens"] == 2  # FakeEncoder: 2 words


def test_on_llm_end_adjusts_and_clears_reservation():
    handler = RateLimitAndCostHandler(encoder=FakeEncoder(), model_name="openai/gpt-4o-mini")
    handler._reservations["r1"] = {"prompt_tokens": 2, "reserved": 10}

    response = MagicMock()
    gen = MagicMock()
    gen.text = "one two three"
    response.generations = [[gen]]
    response.llm_output = None

    with patch.object(handler._limiter, "adjust") as adjust:
        handler.on_llm_end(response, run_id="r1")
    adjust.assert_called_once()
    assert "r1" not in handler._reservations


def test_on_llm_end_prefers_api_usage_tokens():
    handler = RateLimitAndCostHandler(
        encoder=FakeEncoder(),
        model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    )
    handler._reservations["r1"] = {"prompt_tokens": 999, "reserved": 2000}

    response = MagicMock()
    response.generations = []
    response.llm_output = {
        "token_usage": {"prompt_tokens": 1481, "completion_tokens": 759}
    }

    recorded: list[dict] = []

    def _capture(record):
        recorded.append(record)

    with patch.object(handler._limiter, "adjust") as adjust:
        with patch("src.agents.observability.append_llm_call", _capture):
            with patch("src.agents.observability.get_llm_node", return_value=""):
                with patch("src.agents.observability.get_run_context", return_value={}):
                    with patch(
                        "src.agents.observability.langfuse_tracing.is_langfuse_enabled",
                        return_value=False,
                    ):
                        handler.on_llm_end(response, run_id="r1")

    adjust.assert_called_once()
    assert recorded
    assert recorded[0]["prompt_tokens"] == 1481
    assert recorded[0]["completion_tokens"] == 759
    assert recorded[0]["usage_from_api"] is True
    # $0.02/$0.03 per 1M
    assert abs(recorded[0]["cost_in"] - 1481 / 1000 * 0.00002) < 1e-12
    assert abs(recorded[0]["cost_out"] - 759 / 1000 * 0.00003) < 1e-12


def test_extract_usage_tokens_from_llm_output():
    response = MagicMock()
    response.llm_output = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 4}}
    response.generations = []
    assert _extract_usage_tokens(response) == (10, 4)


def test_on_llm_error_clears_reservation():
    handler = RateLimitAndCostHandler(encoder=FakeEncoder(), model_name="openai/gpt-4o-mini")
    handler._reservations["r1"] = {"prompt_tokens": 2, "reserved": 10}
    with patch.object(handler._limiter, "adjust"):
        handler.on_llm_error(RuntimeError("boom"), run_id="r1")
    assert "r1" not in handler._reservations
