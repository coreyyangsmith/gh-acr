from __future__ import annotations

"""Exact-match metric utilities.

An *exact match* is defined as the predicted text being **byte-for-byte** identical to the ground-truth text **after normalising line endings**.
"""

from typing import Dict

__all__ = ["is_exact_match", "per_file", "overall"]


def _normalise(text: str) -> str:  # noqa: D401
    """Return *text* with CRLF normalised to LF and any trailing whitespace removed."""
    return text.replace("\r\n", "\n").rstrip()


def is_exact_match(pred: str, truth: str) -> bool:  # noqa: D401
    """Return **True** if *pred* matches *truth* exactly after normalisation."""
    return _normalise(pred) == _normalise(truth)


def per_file(pred_map: Dict[str, str], truth_map: Dict[str, str]):  # noqa: D401
    """Return a dict \{file_path: bool\} for each file in *truth_map*."""
    return {path: is_exact_match(pred_map.get(path, ""), truth) for path, truth in truth_map.items()}


def overall(pred_map: Dict[str, str], truth_map: Dict[str, str]) -> bool:  # noqa: D401
    """Return **True** if **all** files match exactly."""
    return all(per_file(pred_map, truth_map).values()) 