from __future__ import annotations

"""CLI to remove duplicate runs per (id, file_name, eval_method) in results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tyro

from .data_loader import load_results


@dataclass
class Flags:
    """Options for deduplicating results.

    Keeps the row with the highest similarity per group. If all rows in a group
    have missing similarity, the first occurrence is kept.

    - results_csv: Path to consolidated results CSV; if None, picks latest data/*_results_all.csv
    - output_csv: Optional output path; defaults to <input_dir>/<input_stem>_dedup.csv
    - drop_missing_file_name: If True, ignore rows with empty/NA file_name before deduping
    """

    results_csv: Optional[Path] = None
    output_csv: Optional[Path] = None
    drop_missing_file_name: bool = True


def _resolve_results_path(path: Optional[Path]) -> Path:
    if path is None:
        return load_results(None).path
    if path.suffix == "":
        candidate = path.with_suffix(".csv")
        if candidate.exists():
            return candidate
    return path


def _coerce_similarity(series: pd.Series) -> pd.Series:
    """Convert to numeric; non-convertible become NaN (safe for ranking)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def main(flags: Flags) -> None:
    resolved_csv = _resolve_results_path(flags.results_csv)
    data = load_results(resolved_csv)
    df = data.dataframe

    required_cols = {"id", "file_name", "eval_method", "similarity"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {sorted(missing)}")

    original_rows = len(df)

    # Optionally drop rows without a concrete file_name (commonly prep rows)
    if flags.drop_missing_file_name:
        df = df[~df["file_name"].isna()].copy()
        # Treat empty strings as missing
        df = df[df["file_name"].astype(str).str.len() > 0].copy()

    considered_rows = len(df)

    # Prepare similarity for ranking
    df["_similarity_numeric"] = _coerce_similarity(df["similarity"])  # NaN where not parseable
    df["_similarity_rank"] = df["_similarity_numeric"].fillna(-np.inf)

    # Group and keep the index of the max-similarity row per (id, file_name, eval_method)
    group_keys = ["id", "file_name", "eval_method"]
    if df.empty:
        deduped = df.copy()
    else:
        idx = df.groupby(group_keys)["_similarity_rank"].idxmax()
        deduped = df.loc[idx].copy()

    # Clean up helper columns
    for col in ["_similarity_numeric", "_similarity_rank"]:
        if col in deduped.columns:
            deduped.drop(columns=[col], inplace=True)

    # Sort for stable output
    sort_cols = [c for c in ["id", "file_name", "eval_method", "similarity"] if c in deduped.columns]
    if sort_cols:
        deduped.sort_values(sort_cols, inplace=True)

    # Determine output path
    out_path = flags.output_csv or (resolved_csv.parent / f"{resolved_csv.stem}_dedup.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(out_path, index=False)

    print(
        {
            "input": str(resolved_csv),
            "output": str(out_path),
            "rows_in": int(original_rows),
            "rows_considered": int(considered_rows),
            "rows_out": int(len(deduped)),
            "groups_kept": int(len(deduped)),
            "duplicates_removed": int(max(0, considered_rows - len(deduped))),
            "group_by": group_keys,
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


