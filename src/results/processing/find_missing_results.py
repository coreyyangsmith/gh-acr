from __future__ import annotations

"""CLI to identify missing and fully processed instances by `id` in results.

How to run (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.results.processing.find_missing_results --results-per-instance 6 --results-csv data\results.csv

- Direct script:
  python src\results\processing\find_missing_results.py --results-per-instance 6 --results-csv data\results.csv

Example:
  python -m src.results.processing.find_missing_results --results-per-instance 6 --results-csv data\2025_09_20_results_gemma_all.csv --remove-prep true
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from ..data_loader import load_results


@dataclass
class Flags:
    """Parameters controlling input, expectations, and outputs.

    - results_csv: Path to results CSV; if None, picks latest matching data/*_results_all.csv
    - results_per_instance: Expected number of rows per unique id (including prep rows for counting)
    - remove_prep: If True, exclude rows with eval_method == "prep" from processed_instances output
    - output_dir: Where to write outputs; defaults to the input CSV's parent directory
    """

    results_csv: Optional[Path] = None
    results_per_instance: int = 0
    remove_prep: bool = False
    output_dir: Optional[Path] = None


def _resolve_results_path(path: Optional[Path]) -> Path:
    """Resolve optional path to an existing CSV, allowing extension-less input."""
    if path is None:
        return load_results(None).path
    # Allow passing a path without extension; assume .csv
    if path.suffix == "":
        candidate = path.with_suffix(".csv")
        if candidate.exists():
            return candidate
    return path


def main(flags: Flags) -> None:
    """Compute and export CSVs for missing and processed instance IDs."""
    if flags.results_per_instance <= 0:
        raise ValueError("results_per_instance must be > 0")

    resolved_csv = _resolve_results_path(flags.results_csv)
    data = load_results(resolved_csv)
    df = data.dataframe

    # Validate required columns
    required_cols = {"id", "eval_method"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {sorted(missing)}")

    # Count rows per unique id (include all rows, including prep)
    counts = df.groupby("id").size().reset_index(name="count")

    ids_missing = set(counts.loc[counts["count"] < flags.results_per_instance, "id"].tolist())
    ids_processed = set(counts.loc[counts["count"] == flags.results_per_instance, "id"].tolist())

    # Prepare outputs
    # missed_instances: include all rows for missing IDs (including prep)
    missed_df = df[df["id"].isin(ids_missing)].copy()

    # processed_instances: include all rows for processed IDs; optionally drop prep rows
    processed_df = df[df["id"].isin(ids_processed)].copy()
    if flags.remove_prep and not processed_df.empty:
        processed_df = processed_df[processed_df["eval_method"].astype(str).str.lower() != "prep"].copy()

    # Determine output directory
    output_dir = flags.output_dir or resolved_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    missed_path = output_dir / "missed_instances.csv"
    processed_path = output_dir / "processed_instances.csv"

    # Write CSVs
    missed_df.to_csv(missed_path, index=False)
    processed_df.to_csv(processed_path, index=False)

    # Print concise summary
    unique_ids_total = counts.shape[0]
    print(
        {
            "input": str(resolved_csv),
            "output_dir": str(output_dir),
            "expected_per_id": int(flags.results_per_instance),
            "unique_ids_total": int(unique_ids_total),
            "unique_ids_missing": int(len(ids_missing)),
            "unique_ids_processed": int(len(ids_processed)),
            "rows_written_missed": int(len(missed_df)),
            "rows_written_processed": int(len(processed_df)),
            "remove_prep_in_processed": bool(flags.remove_prep),
            # Note: IDs with count > expected are not exported per spec
            "unique_ids_overfilled": int(
                counts.loc[counts["count"] > flags.results_per_instance].shape[0]
            ),
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


