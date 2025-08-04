from __future__ import annotations

"""BLEU-3 metric utilities.

The implementation here is a *minimal* re-implementation of the BLEU score that
supports up to tri-gram precision (hence BLEU-3). It deliberately avoids heavy
external dependencies (e.g. *nltk*) so that it can run in lightweight
environments that only have the packages listed in *requirements.txt*.

The formulation follows Papineni *et al.* (2002):
    BLEU = BP * exp( \sum_n w_n * log p_n )
where *p_n* is the modified n-gram precision for n in \[1, 3\] and the *brevity
penalty* (BP) is `exp(1 - len_ref/len_pred)` when `len_pred < len_ref` and 1
otherwise.  Uniform weights \(w_n = 1/3\) are used for BLEU-3.
"""

from collections import Counter
from math import exp, log
from typing import Dict, List, Tuple

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

    # Geometric mean of precisions (log-domain to avoid underflow)
    if min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        geo_mean = exp(sum(log(p) for p in precisions) / 3)

    # Brevity penalty (BP)
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
