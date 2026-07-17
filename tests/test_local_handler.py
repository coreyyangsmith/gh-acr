"""Tests for LocalHandler (no weight loading)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.handlers.local_handler import LocalHandler


def test_matches_and_parse():
    h = LocalHandler()
    assert h.matches("local:meta-llama/Llama-3.1-8B-Instruct")
    assert not h.matches("openai/gpt-4o-mini")
    assert (
        h.parse_model_id("local:meta-llama/Llama-3.1-8B-Instruct")
        == "meta-llama/Llama-3.1-8B-Instruct"
    )


def test_empty_model_id_raises():
    with pytest.raises(ValueError):
        LocalHandler().parse_model_id("local:")


def test_require_api_key_not_supported():
    with pytest.raises(RuntimeError, match="does not use a single API key"):
        LocalHandler().require_api_key()


def test_create_delegates_to_local_backend():
    sentinel = (object(), object())
    with patch(
        "src.agents.backends.local_backend.create_local_backend",
        return_value=sentinel,
    ) as mock_create:
        result = LocalHandler().create("local:gpt2")
    mock_create.assert_called_once_with("local:gpt2")
    assert result is sentinel
