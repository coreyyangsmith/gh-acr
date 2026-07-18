"""Tests for handler factory helpers (formerly backend shims)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.handlers import (
    create_groq_backend,
    create_openai_backend,
)


def test_create_openai_backend_delegates():
    sentinel = (object(), object())
    with patch(
        "src.agents.handlers.openai_handler.OpenAIHandler"
    ) as cls:
        cls.return_value.create.return_value = sentinel
        assert create_openai_backend("openai/gpt-4o-mini") is sentinel
        cls.return_value.create.assert_called_once_with("openai/gpt-4o-mini")


def test_create_groq_backend_delegates():
    sentinel = (None, object())
    with patch(
        "src.agents.handlers.groq_handler.GroqHandler"
    ) as cls:
        cls.return_value.create.return_value = sentinel
        assert create_groq_backend("groq:llama") is sentinel


def test_create_local_backend_still_callable():
    # Lazy import: local_backend pulls transformers/torch
    pytest.importorskip("transformers")
    from src.agents.handlers.local_backend import create_local_backend

    with pytest.raises(ValueError, match="local:"):
        create_local_backend("local:")
