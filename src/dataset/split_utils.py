from __future__ import annotations

"""Utility helpers for slicing the GitGoodBench dataset by *difficulty* level.

Produces separate CSVs (easy/medium/hard) next to the original dataset or
returns individual DataFrames for programmatic use.

Usage (PowerShell examples):
    # Use default dataset location
    python -m src.dataset.split_utils

    # Provide an explicit CSV path and write split files next to it
    python -m src.dataset.split_utils C:\data\git_good_bench.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import tyro

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

    # Assign fresh random numeric index in each split; keep slug 'id' column intact
    def _assign_fresh_index(df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng()
        out = df.copy()
        out.index = rng.choice(10**12, size=len(out), replace=False)
        return out

    # Normalize via loader to get scenario_json and canonical columns
    easy_df = load_benchmark(csv_path or DATA_PATH).query('difficulty == "easy"').copy()
    med_df = load_benchmark(csv_path or DATA_PATH).query('difficulty == "medium"').copy()
    hard_df = load_benchmark(csv_path or DATA_PATH).query('difficulty == "hard"').copy()

    easy_df = _assign_fresh_index(easy_df)
    med_df = _assign_fresh_index(med_df)
    hard_df = _assign_fresh_index(hard_df)

    if write_files:
        base_path = Path(csv_path or DATA_PATH).with_suffix("")
        # Drop helper JSON column and keep canonical order if possible
        desired_cols = [
            "id",
            "name",
            "default_branch",
            "license",
            "stargazers",
            "created_at",
            "topics",
            "programming_language",
            "scenario",
            "sample_type",
            "project_size",
            "project_activity",
            "difficulty",
        ]
        def _finalize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.drop(columns=["scenario_json"], errors="ignore")
            return df[desired_cols] if all(c in df.columns for c in desired_cols) else df

        easy_out = _finalize(easy_df)
        med_out = _finalize(med_df)
        hard_out = _finalize(hard_df)

        # Keep the DataFrame index so the CSV has the leading blank id column
        easy_out.to_csv(f"{base_path}_easy.csv", index=True)
        med_out.to_csv(f"{base_path}_medium.csv", index=True)
        hard_out.to_csv(f"{base_path}_hard.csv", index=True)

    return easy_df, med_df, hard_df 


def cli(csv_path: str | Path | None = None, /, *, write_files: bool = True):
    """CLI wrapper. By default, writes split CSVs next to the provided file."""
    split_by_difficulty(csv_path, write_files=write_files)


if __name__ == "__main__":
    tyro.cli(cli)