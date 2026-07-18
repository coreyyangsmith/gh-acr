"""Tests for RateLimitAndCostHandler callback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.helpers import FakeEncoder

from src.agents.callbacks import RateLimitAndCostHandler
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

    with patch.object(handler._limiter, "adjust") as adjust:
        handler.on_llm_end(response, run_id="r1")
    adjust.assert_called_once()
    assert "r1" not in handler._reservations


def test_on_llm_error_clears_reservation():
    handler = RateLimitAndCostHandler(encoder=FakeEncoder(), model_name="openai/gpt-4o-mini")
    handler._reservations["r1"] = {"prompt_tokens": 2, "reserved": 10}
    with patch.object(handler._limiter, "adjust"):
        handler.on_llm_error(RuntimeError("boom"), run_id="r1")
    assert "r1" not in handler._reservations
