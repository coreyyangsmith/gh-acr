"""Unit tests for eval method registry helpers."""

from __future__ import annotations

import pytest

from src.config.eval_methods import (
    ALL_EVAL_METHODS,
    DEFAULT_METHOD_ORDER,
    get_method_index,
    is_valid_method,
)


def test_is_valid_method():
    for m in ALL_EVAL_METHODS:
        assert is_valid_method(m) is True
    assert is_valid_method("unknown") is False
    assert is_valid_method("bypass_multi") is False


def test_get_method_index_order():
    for i, m in enumerate(DEFAULT_METHOD_ORDER):
        assert get_method_index(m) == i


def test_get_method_index_unknown_raises():
    with pytest.raises(ValueError):
        get_method_index("not_a_method")  # type: ignore[arg-type]


def test_all_methods_match_default_order_set():
    assert set(ALL_EVAL_METHODS) == set(DEFAULT_METHOD_ORDER)
    assert list(ALL_EVAL_METHODS) == list(DEFAULT_METHOD_ORDER)
