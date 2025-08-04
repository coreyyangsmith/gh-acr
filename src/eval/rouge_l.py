from __future__ import annotations

"""ROUGE-L metric utilities.

ROUGE-L measures the **Longest Common Subsequence (LCS)** between the predicted
text and the reference text.  The implementation below follows the original
Lin (2004) definition and computes the F\_1 variant with \(\beta = 1\):

    R\_L = (1 + \beta^2) * P * R / (R + \beta^2 * P)

where *P* is the precision (LCS / |pred|) and *R* is the recall (LCS / |truth|).

The implementation purposefully avoids external dependencies such as
*rouge-score* to keep the project lightweight.
"""

from typing import Dict, List

__all__ = ["rouge_l_score", "per_file", "overall"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:  # noqa: D401 – simple whitespace tokeniser
    """Tokenise *text* by **whitespace** (including newlines)."""
    return text.strip().split()


def _lcs_len(a: List[str], b: List[str]) -> int:  # noqa: D401
    """Return the length of the **Longest Common Subsequence** between *a* and *b*."""
    m, n = len(a), len(b)
    # Using a 2-row DP to reduce memory from O(mn) to O(min(m, n)) could be an
    # optimisation, but the typical file lengths here are small enough that the
    # classic O(mn) DP with a (m+1) x (n+1) table is acceptable.
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m - 1, -1, -1):
        ai = a[i]
        row_i = dp[i]
        row_i1 = dp[i + 1]
        for j in range(n - 1, -1, -1):
            if ai == b[j]:
                row_i[j] = 1 + row_i1[j + 1]
            else:
                row_i[j] = max(row_i[j + 1], row_i1[j])

    return dp[0][0]


# ---------------------------------------------------------------------------
# ROUGE-L core
# ---------------------------------------------------------------------------

def rouge_l_score(pred: str, truth: str) -> float:  # noqa: D401
    """Return the ROUGE-L (F1) score between *pred* and *truth* \[0, 1\]."""

    pred_toks = _tokenise(pred)
    truth_toks = _tokenise(truth)

    if not pred_toks or not truth_toks:
        return 0.0

    lcs = _lcs_len(pred_toks, truth_toks)

    precision = lcs / len(pred_toks)
    recall = lcs / len(truth_toks)

    if precision == 0.0 or recall == 0.0:
        return 0.0

    beta = 1.0
    f_score = (1 + beta ** 2) * precision * recall / (recall + beta ** 2 * precision)
    return f_score


def per_file(pred_map: Dict[str, str], truth_map: Dict[str, str]):  # noqa: D401
    """Return a dict *{file_path: ROUGE-L}* for each file in *truth_map*."""
    return {path: rouge_l_score(pred_map.get(path, ""), truth) for path, truth in truth_map.items()}


def overall(pred_map: Dict[str, str], truth_map: Dict[str, str]) -> float:  # noqa: D401
    """Return the **average** ROUGE-L across all files in *truth_map*."""
    scores = per_file(pred_map, truth_map).values()
    return (sum(scores) / len(truth_map)) if truth_map else 0.0
