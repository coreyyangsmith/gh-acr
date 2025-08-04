from __future__ import annotations

"""Utility helpers for slicing the GitGoodBench dataset by *difficulty* level.

Produces separate CSVs (easy/medium/hard) next to the original dataset or
returns individual DataFrames for programmatic use.
"""

from pathlib import Path
import pandas as pd

from .loader import load_benchmark, DATA_PATH

__all__ = ["split_by_difficulty"]


def split_by_difficulty(csv_path: str | Path | None = None, /, *, write_files: bool = False):  # noqa: D401
    """Return (easy_df, medium_df, hard_df).

    If *write_files* is True, CSVs are written as <name>_easy.csv etc.
    """

    df = load_benchmark(csv_path)
    easy_df = df[df["difficulty"] == "easy"].copy()
    med_df = df[df["difficulty"] == "medium"].copy()
    hard_df = df[df["difficulty"] == "hard"].copy()

    if write_files:
        base_path = Path(csv_path or DATA_PATH).with_suffix("")
        easy_df.to_csv(f"{base_path}_easy.csv", index=False)
        med_df.to_csv(f"{base_path}_medium.csv", index=False)
        hard_df.to_csv(f"{base_path}_hard.csv", index=False)

    return easy_df, med_df, hard_df 