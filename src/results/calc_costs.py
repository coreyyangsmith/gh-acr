from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from .data_loader import load_results


@dataclass
class Flags:
    """Compute input, output, and total costs per instance and overall totals.

    - results_csv: Path to results CSV; if None, picks latest matching data/*_results_all.csv
    - output_dir: Where to write outputs; defaults to the input CSV's parent directory
    - instance_col: Column used to identify instances (default: 'id')
    """

    results_csv: Optional[Path] = None
    output_dir: Optional[Path] = None
    instance_col: str = "id"


def _resolve_results_path(path: Optional[Path]) -> Path:
    if path is None:
        return load_results(None).path
    if path.suffix == "":
        candidate = path.with_suffix(".csv")
        if candidate.exists():
            return candidate
    return path


def _ensure_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize cost columns and coerce to numeric
    cols = {"cost_in", "cost_out", "total_cost"}
    missing = cols.difference(df.columns)
    # Derive total_cost if missing
    if "total_cost" in missing and {"cost_in", "cost_out"}.issubset(df.columns):
        # will add after coercion
        missing = missing.difference({"total_cost"})

    # Coerce existing columns to numeric
    for c in [c for c in cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Derive total_cost if not present
    if "total_cost" not in df.columns and {"cost_in", "cost_out"}.issubset(df.columns):
        df["total_cost"] = pd.to_numeric(df["cost_in"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["cost_out"], errors="coerce"
        ).fillna(0.0)

    # Ensure all three columns exist (fill missing with zeros)
    for c in ["cost_in", "cost_out", "total_cost"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def main(flags: Flags) -> None:
    resolved_csv = _resolve_results_path(flags.results_csv)
    data = load_results(resolved_csv)
    df = _ensure_cost_columns(data.dataframe.copy())

    if flags.instance_col not in df.columns:
        raise ValueError(f"Missing instance column '{flags.instance_col}' in results")

    # Ensure token columns exist and are numeric
    for c in ["tokens_in", "tokens_out"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Aggregate per instance
    group_cols = [flags.instance_col]
    per_instance = (
        df.groupby(group_cols)[["cost_in", "cost_out", "total_cost", "tokens_in", "tokens_out"]]
        .sum()
        .reset_index()
        .rename(columns={
            "cost_in": "cost_in_sum",
            "cost_out": "cost_out_sum",
            "total_cost": "total_cost_sum",
            "tokens_in": "tokens_in_sum",
            "tokens_out": "tokens_out_sum",
        })
    )

    # Totals across all instances
    totals = pd.DataFrame(
        [
            {
                "num_rows": int(len(df)),
                "num_instances": int(per_instance.shape[0]),
                "cost_in_total": float(df["cost_in"].sum()),
                "cost_out_total": float(df["cost_out"].sum()),
                "total_cost_total": float(df["total_cost"].sum()),
                "tokens_in_total": float(df["tokens_in"].sum()),
                "tokens_out_total": float(df["tokens_out"].sum()),
            }
        ]
    )

    # Output paths
    output_dir = flags.output_dir or resolved_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    per_instance_path = output_dir / "costs_per_instance.csv"
    totals_path = output_dir / "costs_totals.csv"

    per_instance.to_csv(per_instance_path, index=False)
    totals.to_csv(totals_path, index=False)

    print(
        {
            "input": str(resolved_csv),
            "output_dir": str(output_dir),
            "per_instance_rows": int(len(per_instance)),
            "totals_rows": int(len(totals)),
            "cost_in_total": float(totals.loc[0, "cost_in_total"]),
            "cost_out_total": float(totals.loc[0, "cost_out_total"]),
            "total_cost_total": float(totals.loc[0, "total_cost_total"]),
            "tokens_in_total": float(totals.loc[0, "tokens_in_total"]),
            "tokens_out_total": float(totals.loc[0, "tokens_out_total"]),
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


