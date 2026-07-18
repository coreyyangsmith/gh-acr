"""Tests for token counting helpers."""

from __future__ import annotations

from src.agents.token_utils import count_tokens, tiktoken_encoder
from tests.helpers import FakeEncoder


def test_count_tokens_with_encoder():
    enc = FakeEncoder()
    assert count_tokens(enc, "one two three") == 3
    assert count_tokens(enc, "") == 0


def test_count_tokens_without_encoder_falls_back_to_words():
    assert count_tokens(None, "one two three") == 3
    assert count_tokens(None, "") == 0


def test_tiktoken_encoder_returns_usable_or_none():
    enc = tiktoken_encoder("gpt-4o-mini")
    if enc is None:
        # Environment without tiktoken encodings is acceptable
        assert True
    else:
        assert hasattr(enc, "encode")
        assert len(enc.encode("hello")) >= 1
