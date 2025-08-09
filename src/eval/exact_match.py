from __future__ import annotations

"""Exact-match metric utilities.

Uses a normalisation step to compare predicted text to ground truth. If
`rapidfuzz` is available, it can be leveraged for efficient string operations,
but the equality check is straightforward and does not require a library.
"""

from typing import Dict

try:  # pragma: no cover – optional dependency not strictly required
    from rapidfuzz.utils import default_process as _rf_normalise  # type: ignore
except Exception:  # pragma: no cover
    _rf_normalise = None  # type: ignore

__all__ = ["is_exact_match", "per_file", "overall"]


def _normalise(text: str) -> str:  # noqa: D401
    """Return *text* with CRLF normalised to LF and any trailing whitespace removed."""
    basic = text.replace("\r\n", "\n").rstrip()
    if _rf_normalise is not None:
        try:
            return _rf_normalise(basic)
        except Exception:  # pragma: no cover
            return basic
    return basic


def is_exact_match(pred: str, truth: str) -> bool:  # noqa: D401
    """Return **True** if *pred* matches *truth* exactly after normalisation."""
    return _normalise(pred) == _normalise(truth)


def per_file(pred_map: Dict[str, str], truth_map: Dict[str, str]):  # noqa: D401
    """Return a dict \{file_path: bool\} for each file in *truth_map*."""
    return {path: is_exact_match(pred_map.get(path, ""), truth) for path, truth in truth_map.items()}


def overall(pred_map: Dict[str, str], truth_map: Dict[str, str]) -> bool:  # noqa: D401
    """Return **True** if **all** files match exactly."""
    return all(per_file(pred_map, truth_map).values()) 