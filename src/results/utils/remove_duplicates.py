from __future__ import annotations

"""CLI to drop duplicates using a unique merge_commit_hash parsed from `scenario`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import ast
import pandas as pd
import tyro


def _extract_merge_commit_hash(scenario_str: Any) -> Optional[str]:
    """Best-effort parser to extract `merge_commit_hash` from a scenario value."""
    if scenario_str is None or (isinstance(scenario_str, float) and pd.isna(scenario_str)):
        return None
    if isinstance(scenario_str, dict):
        return scenario_str.get("merge_commit_hash")
    text = str(scenario_str).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            value = parsed.get("merge_commit_hash")
            return str(value) if value is not None else None
    except Exception:
        return None
    return None


@dataclass
class Flags:
    """Arguments for deduplicating by `merge_commit_hash`.

    - input_csv: Path to input CSV file
    - output_csv: Optional output path; defaults to <input_dir>/uniques.csv
    - scenario_col: Column containing the scenario dict-like string (default: 'scenario')
    """

    input_csv: Path
    output_csv: Optional[Path] = None
    scenario_col: str = "scenario"


def main(flags: Flags) -> None:
    input_path = Path(flags.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if flags.scenario_col not in df.columns:
        raise ValueError(f"Column '{flags.scenario_col}' not found in CSV")

    # Parse merge_commit_hash into a new column for deduping
    df["merge_commit_hash"] = df[flags.scenario_col].apply(_extract_merge_commit_hash)

    # Keep only rows with a valid hash
    valid = df[df["merge_commit_hash"].notna()].copy()

    # Drop duplicates by merge_commit_hash, keeping first occurrence
    uniques = valid.drop_duplicates(subset=["merge_commit_hash"], keep="first")

    # Determine output path
    out_path = flags.output_csv or (input_path.parent / "uniques.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uniques.to_csv(out_path, index=False)

    print(
        {
            "input": str(input_path),
            "output": str(out_path),
            "rows_in": int(len(df)),
            "rows_with_hash": int(len(valid)),
            "rows_unique": int(len(uniques)),
            "unique_hashes": int(uniques["merge_commit_hash"].nunique()),
        }
    )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


