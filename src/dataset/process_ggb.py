from __future__ import annotations

"""Utility to extract GitGoodBench scenarios that have an explicit merge commit hash.

This script/CLI is intentionally separate from the main pipeline so that the
filtered subset can be inspected or reused independently.

Usage (PowerShell example):
    python -m src.dataset.process_ggb --output-csv data/git_good_bench_merge_commits.csv

If you need to point to a custom GitGoodBench CSV:
    python -m src.dataset.process_ggb --input-csv custom/path.csv --output-csv filtered.csv
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from .loader import load_benchmark, DATA_PATH


def process_ggb(
    *,
    input_csv: Optional[str] = None,
    output_csv: str = "data/git_good_bench_merge_commits.csv",
) -> None:
    """Filter *GitGoodBench* rows that include a merge commit hash and export to CSV.

    Parameters
    ----------
    input_csv
        Optional path to the source ``git_good_bench.csv``. Defaults to the
        canonical dataset location detected via :data:`DATA_PATH`.
    output_csv
        Destination path for the filtered CSV. Parent directories are created
        automatically if they do not exist.
    """

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = load_benchmark(input_csv or DATA_PATH)

    # ------------------------------------------------------------------
    # Identify rows whose *scenario* JSON includes a merge commit hash key
    # ------------------------------------------------------------------
    mask = df["scenario_json"].apply(lambda d: isinstance(d, dict) and "merge_commit_hash" in d)
    filtered_df = df[mask].copy()

    # Ensure the same column order / set as the original CSV (without the helper JSON column)
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

    # Guard: drop helper column if present and reorder
    filtered_df = filtered_df[desired_cols]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the original DataFrame index to preserve the leading blank column
    filtered_df.to_csv(out_path, index=True)

    print(f"Exported {len(filtered_df)} rows with merge_commit_hash → {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    tyro.cli(process_ggb) 