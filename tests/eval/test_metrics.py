"""Unit tests for eval metrics (exact match, BLEU-3, ROUGE-L)."""

from __future__ import annotations

import pytest

from src.eval.exact_match import is_exact_match, overall as em_overall, per_file as em_per_file
from src.eval.bleu import bleu3_score, overall as bleu_overall, per_file as bleu_per_file
from src.eval.rouge_l import rouge_l_score, overall as rouge_overall, per_file as rouge_per_file


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


def test_exact_match_identical():
    assert is_exact_match("hello\n", "hello\n") is True


def test_exact_match_crlf_vs_lf():
    assert is_exact_match("hello\r\nworld\r\n", "hello\nworld\n") is True


def test_exact_match_trailing_whitespace():
    assert is_exact_match("hello  \n", "hello") is True


def test_exact_match_differing():
    assert is_exact_match("hello", "world") is False


def test_em_per_file_keys_from_truth_missing_pred():
    truth = {"a.py": "x", "b.py": "y"}
    pred = {"a.py": "x", "extra.py": "z"}
    result = em_per_file(pred, truth)
    assert set(result) == {"a.py", "b.py"}
    assert result["a.py"] is True
    assert result["b.py"] is False  # missing pred -> ""


def test_em_overall_all_match():
    m = {"a.py": "x", "b.py": "y"}
    assert em_overall(m, m) is True


def test_em_overall_one_mismatch():
    assert em_overall({"a.py": "x"}, {"a.py": "x", "b.py": "y"}) is False


def test_em_overall_empty_truth_is_true():
    # all([]) == True in Python
    assert em_overall({}, {}) is True


# ---------------------------------------------------------------------------
# BLEU-3
# ---------------------------------------------------------------------------


def test_bleu3_identical_near_one():
    text = "the quick brown fox jumps over the lazy dog"
    score = bleu3_score(text, text)
    assert 0.9 <= score <= 1.0


def test_bleu3_disjoint_zero():
    assert bleu3_score("aaa bbb ccc", "xxx yyy zzz") == 0.0


def test_bleu3_empty_inputs():
    assert bleu3_score("", "hello") == 0.0
    assert bleu3_score("hello", "") == 0.0
    assert bleu3_score("", "") == 0.0


def test_bleu3_partial_overlap_in_unit_interval():
    score = bleu3_score("the quick brown fox", "the quick red fox jumps")
    assert 0.0 < score < 1.0


def test_bleu3_deterministic():
    a, b = "one two three four", "one two three five"
    assert bleu3_score(a, b) == bleu3_score(a, b)


def test_bleu_per_file_and_overall():
    truth = {"a.py": "one two three four", "b.py": "alpha beta gamma"}
    pred = {"a.py": "one two three four", "b.py": "zzz"}
    scores = bleu_per_file(pred, truth)
    assert set(scores) == {"a.py", "b.py"}
    assert scores["a.py"] >= 0.9
    assert scores["b.py"] == 0.0
    overall = bleu_overall(pred, truth)
    assert overall == pytest.approx(sum(scores.values()) / 2)


def test_bleu_overall_empty_truth():
    assert bleu_overall({}, {}) == 0.0


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def test_rouge_l_identical_near_one():
    text = "the quick brown fox"
    score = rouge_l_score(text, text)
    assert 0.9 <= score <= 1.0


def test_rouge_l_disjoint_zero():
    assert rouge_l_score("aaa bbb", "xxx yyy") == 0.0


def test_rouge_l_empty_inputs():
    assert rouge_l_score("", "hello") == 0.0
    assert rouge_l_score("hello", "") == 0.0


def test_rouge_l_partial_overlap():
    score = rouge_l_score("the quick brown fox", "the quick red fox")
    assert 0.0 < score < 1.0


def test_rouge_per_file_and_overall():
    truth = {"a.py": "one two three", "b.py": "alpha beta"}
    pred = {"a.py": "one two three", "b.py": ""}
    scores = rouge_per_file(pred, truth)
    assert scores["a.py"] >= 0.9
    assert scores["b.py"] == 0.0
    assert rouge_overall(pred, truth) == pytest.approx(sum(scores.values()) / 2)


def test_rouge_overall_empty_truth():
    assert rouge_overall({}, {}) == 0.0
