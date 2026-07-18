"""Print unique values in a specified column (default: `model_name`) from a CSV.

Usage (from repo root, PowerShell one-liners):

- Module form (recommended):
  python -m src.analysis.processing.find_unique_models data\2025_10_18_ALL_RESULTS.csv --column model_name

- Direct script:
  python src\results\processing\find_unique_models.py data\input.csv --column model_name

This prints one unique value per line (sorted ascending), followed by a total count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import tyro


def _detect_column(df: pd.DataFrame, requested: str) -> str:
    """Return the column to use, falling back to common model-name variants."""
    if requested in df.columns:
        return requested
    candidates = [
        c for c in df.columns if c.lower() in {"model", "model_name"} or "model" in c.lower()
    ]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Column '{requested}' not found and no 'model*' column detected in input CSV."
    )


def find_unique(input_file: Path, *, column: str = "model_name") -> list[str]:
    df = pd.read_csv(input_file)
    col = _detect_column(df, column)

    series = df[col].dropna()
    # Normalize to strings and trim whitespace; drop empty strings post-trim
    series = series.astype(str).str.strip()
    series = series[series != ""]

    return sorted(series.unique())


def main(input_file: Path, *, column: str = "model_name") -> None:
    uniques = find_unique(input_file, column=column)
    for value in uniques:
        print(value)
    print(f"Total unique: {len(uniques)}")


if __name__ == "__main__":
    # Accept: input_file [--column model_name]
    args = tyro.cli(tuple[Path, Optional[str]])
    in_path, col = args
    main(in_path, column=col or "model_name")

