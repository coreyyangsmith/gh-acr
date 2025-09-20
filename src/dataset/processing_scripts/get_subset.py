from __future__ import annotations

r"""Randomly sample a percentage of a GitGoodBench CSV into a new file.

Usage (PowerShell examples):
    # Required: --percent; optional: --seed and --output-csv
    python -m src.dataset.get_subset C:\data\git_good_bench_merge_commits.csv --percent 10 --seed 42
    
    # Write next to input with an auto-generated filename suffix
    python -m src.dataset.get_subset data/git_good_bench_merge_commits.csv --percent 5
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import tyro
from .loader import load_benchmark


def _normalize_percent_tag(percent: float) -> str:
    """Return a filesystem-friendly tag for a percentage value.

    Examples
    --------
    - 10.0 → ``"10"``
    - 12.5 → ``"12_5"``
    """

    return str(int(percent)) if float(percent).is_integer() else str(percent).replace(".", "_")


def get_subset(
    csv_path: str | Path,
    /,
    *,
    percent: float,
    seed: int = 0,
    output_csv: Optional[str] = None,
) -> Path:
    """Sample ``percent`` percent of rows from ``csv_path`` deterministically.

    The subset is written to ``output_csv`` if provided; otherwise a filename is
    generated next to the input, e.g., ``<name>_subset_10_seed42.csv``.

    Parameters
    ----------
    csv_path
        Path to the source dataset CSV.
    percent
        Percentage of rows to sample (0, 100].
    seed
        Random seed for deterministic sampling.
    output_csv
        Optional destination CSV path; if omitted, an auto-derived filename is
        used next to the input.

    Returns
    -------
    pathlib.Path
        Path to the written subset CSV.

    Raises
    ------
    ValueError
        If ``percent`` is outside the (0, 100] range.
    FileNotFoundError
        If the input CSV cannot be found.
    """

    if not (0 < percent <= 100):
        raise ValueError("percent must be in the range (0, 100]")

    # Load via project loader to normalize (drops leading CSV index; adds scenario_json)
    input_path = Path(csv_path).expanduser().resolve()
    df = load_benchmark(input_path)

    frac = percent / 100.0
    subset_df = df.sample(frac=frac, random_state=seed).copy()

    # Prepare output path
    if output_csv is not None:
        out_path = Path(output_csv).expanduser().resolve()
    else:
        percent_tag = _normalize_percent_tag(percent)
        out_path = input_path.with_name(f"{input_path.stem}_subset_{percent_tag}_seed{seed}.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Assign a fresh random numeric index for the subset; keep existing 'id' slug intact
    rng = np.random.default_rng()
    subset_df.index = rng.choice(10**12, size=len(subset_df), replace=False)

    # Drop helper JSON column and optionally reorder to canonical column order if present
    subset_df = subset_df.drop(columns=["scenario_json"], errors="ignore")
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
    if all(c in subset_df.columns for c in desired_cols):
        subset_df = subset_df[desired_cols]

    # Write with index to preserve the leading blank first column
    subset_df.to_csv(out_path, index=True)

    return out_path


def cli(
    csv_path: str | Path,
    /,
    *,
    percent: float,
    seed: int = 0,
    output_csv: Optional[str] = None,
) -> None:
    """CLI wrapper. Required: ``--percent``. Optional: ``--seed``, ``--output-csv``."""
    out_path = get_subset(csv_path, percent=percent, seed=seed, output_csv=output_csv)
    # Friendly print with relative fallback
    try:
        rel = out_path.relative_to(Path.cwd())
    except ValueError:
        rel = out_path
    print(f"Exported subset ({percent}% ; seed={seed}) → {rel}")


if __name__ == "__main__":
    tyro.cli(cli)


