from __future__ import annotations

"""BLEU-3 metric utilities.

Prefers the `sacrebleu` library if available; falls back to a lightweight
manual BLEU-3 implementation otherwise.
"""

from collections import Counter
from math import exp, log
from typing import Dict, List, Tuple

# Optional dependency: sacrebleu
try:  # pragma: no cover - optional performance/quality enhancement
    from sacrebleu.metrics import BLEU as _SacreBLEU  # type: ignore
    _BLEU3 = _SacreBLEU(effective_order=True, max_ngram_order=3)  # type: ignore[call-arg]
except Exception:  # pragma: no cover
    _BLEU3 = None  # type: ignore

__all__ = ["bleu3_score", "per_file", "overall"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:  # noqa: D401 – simple whitespace tokeniser
    """Tokenise *text* by **whitespace** (including newlines)."""
    return text.strip().split()


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:  # noqa: D401
    """Return a list of *n-grams* extracted from *tokens*."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# BLEU-3 core
# ---------------------------------------------------------------------------

def bleu3_score(pred: str, truth: str) -> float:  # noqa: D401
    """Return the BLEU-3 score between *pred* and *truth* \[0, 1\]."""

    # Prefer sacrebleu if available
    if _BLEU3 is not None:
        try:
            score = _BLEU3.sentence_score(pred, [truth]).score  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            # Fallback to corpus scoring for a single pair
            try:
                score = _BLEU3.corpus_score([pred], [[truth]]).score  # type: ignore[attr-defined]
            except Exception:
                score = None  # type: ignore[assignment]
        if score is not None:  # type: ignore[truthy-bool]
            return max(0.0, min(1.0, float(score) / 100.0))

    # Manual BLEU-3 fallback
    pred_toks = _tokenise(pred)
    truth_toks = _tokenise(truth)

    if not pred_toks or not truth_toks:
        return 0.0

    precisions = []
    for n in (1, 2, 3):
        pred_ngrams = Counter(_ngrams(pred_toks, n))
        truth_ngrams = Counter(_ngrams(truth_toks, n))

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        overlap = sum((pred_ngrams & truth_ngrams).values())
        precisions.append(overlap / sum(pred_ngrams.values()))

    if min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        geo_mean = exp(sum(log(p) for p in precisions) / 3)

    len_pred = len(pred_toks)
    len_truth = len(truth_toks)
    if len_pred == 0:
        return 0.0
    bp = 1.0 if len_pred > len_truth else exp(1 - (len_truth / len_pred))
    return bp * geo_mean


def per_file(pred_map: Dict[str, str], truth_map: Dict[str, str]):  # noqa: D401
    """Return a dict *{file_path: BLEU-3}* for each file in *truth_map*."""
    return {path: bleu3_score(pred_map.get(path, ""), truth) for path, truth in truth_map.items()}


def overall(pred_map: Dict[str, str], truth_map: Dict[str, str]) -> float:  # noqa: D401
    """Return the **average** BLEU-3 across all files in *truth_map*."""
    scores = per_file(pred_map, truth_map).values()
    return (sum(scores) / len(truth_map)) if truth_map else 0.0
