"""Tests for thin backend factory shims."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.backends import (
    create_groq_backend,
    create_local_backend,
    create_openai_backend,
)


def test_create_openai_backend_delegates():
    sentinel = (object(), object())
    with patch(
        "src.agents.backends.openai_backend.OpenAIHandler"
    ) as cls:
        cls.return_value.create.return_value = sentinel
        assert create_openai_backend("openai/gpt-4o-mini") is sentinel
        cls.return_value.create.assert_called_once_with("openai/gpt-4o-mini")


def test_create_groq_backend_delegates():
    sentinel = (None, object())
    with patch(
        "src.agents.backends.groq_backend.GroqHandler"
    ) as cls:
        cls.return_value.create.return_value = sentinel
        assert create_groq_backend("groq:llama") is sentinel


def test_create_local_backend_still_callable():
    # Ensure the compatibility export exists and validates empty ids
    with pytest.raises(ValueError, match="local:"):
        create_local_backend("local:")
