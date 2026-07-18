from __future__ import annotations


r"""Utility to extract GitGoodBench scenarios that have an explicit merge commit hash.

This script/CLI is intentionally separate from the main pipeline so that the
filtered subset can be inspected or reused independently.

Usage (PowerShell examples):
    # Default input (dataset default) → output in data/git_good_bench_merge_commits.csv
    python -m src.dataset.processing.extract_merge_scenario_from_ggb

    # Pass full paths
    python -m src.dataset.processing.extract_merge_scenario_from_ggb --input-csv C:\data\git_good_bench.csv --output-csv C:\out\filtered.csv

    # Pass just names (placed under the data/ directory automatically)
    # Extension is optional; ".csv" will be added if missing
    python -m src.dataset.processing.extract_merge_scenario_from_ggb --input-csv git_good_bench --output-csv git_good_bench_merge_commits
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import tyro

from ..loader import load_benchmark, DATA_PATH


def _resolve_csv_path(value: Optional[str], *, base_dir: Path, default_path: Path, require_exists: bool) -> Path:
    """Resolve a user-provided CSV "name" or path into an absolute ``Path``.

    Rules
    - If ``value`` is None → return ``default_path``.
    - If ``value`` is a bare name (no directory components), place it in ``base_dir``.
    - If ``value`` has no ``.csv`` suffix, add it.
    - Absolute or relative paths with directories are respected as-is.
    - If ``require_exists`` is True, the resulting path must exist.
    """

    if value is None:
        return default_path

    path = Path(value)

    # Ensure .csv suffix if missing
    if path.suffix == "":
        path = path.with_suffix(".csv")

    # If the user passed just a filename (no parent), place it under base_dir
    if not path.is_absolute() and path.parent == Path("."):
        path = base_dir / path.name

    path = path.expanduser().resolve()

    if require_exists and not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    return path


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
    # Resolve paths and load dataset
    # ------------------------------------------------------------------
    data_dir: Path = DATA_PATH.expanduser().resolve().parent

    resolved_input: Path = _resolve_csv_path(
        input_csv,
        base_dir=data_dir,
        default_path=DATA_PATH.expanduser().resolve(),
        require_exists=True,
    )

    # For output, allow names or paths; default already points to data/ by default
    resolved_output: Path = _resolve_csv_path(
        output_csv,
        base_dir=data_dir,
        default_path=Path(output_csv).expanduser().resolve(),
        require_exists=False,
    )

    df = load_benchmark(resolved_input)

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
    filtered_df = filtered_df.drop(columns=["scenario_json"], errors="ignore")

    missing_cols = [c for c in desired_cols if c not in filtered_df.columns]
    if missing_cols:
        present_in_order = [c for c in desired_cols if c in filtered_df.columns]
        extra_cols = [c for c in filtered_df.columns if c not in present_in_order]
        filtered_df = filtered_df[present_in_order + extra_cols]
        print("Note: input CSV missing columns → " + ", ".join(missing_cols))
    else:
        filtered_df = filtered_df[desired_cols]

    # ------------------------------------------------------------------
    # Assign a fresh random numeric index for each row (first blank CSV column)
    # Keep the existing 'id' slug column unchanged.
    # ------------------------------------------------------------------
    rng = np.random.default_rng()
    new_index = rng.choice(10**12, size=len(filtered_df), replace=False)
    filtered_df.index = new_index

    # Finalize: drop helper json and set canonical column order if available
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
    filtered_df = filtered_df.drop(columns=["scenario_json"], errors="ignore")
    if all(c in filtered_df.columns for c in desired_cols):
        filtered_df = filtered_df[desired_cols]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = resolved_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the original DataFrame index to preserve the leading blank column
    filtered_df.to_csv(out_path, index=True)

    try:
        rel = out_path.relative_to(Path.cwd())
    except ValueError:
        rel = out_path
    print(f"Exported {len(filtered_df)} rows with merge_commit_hash → {rel}")


if __name__ == "__main__":
    tyro.cli(process_ggb) 