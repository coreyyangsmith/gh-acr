from __future__ import annotations

"""Filter a results CSV to only rows with exact_match == True.

Usage:
    python -m src.results.find_em_dataset --input-csv data/2025_10_07_ALL_RESULTS.csv
    # writes data/2025_10_07_ALL_RESULTS_exact_matches.csv by default

Or specify an explicit output path:
    python -m src.results.find_em_dataset --input-csv data/2025_10_07_ALL_RESULTS.csv --output-csv results/em_subset.csv
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from ..utils.logger import logger


def _coerce_exact_match(series: pd.Series) -> pd.Series:
    """Coerce a heterogeneous `exact_match` column to nullable boolean.

    Accepts booleans and common string encodings. Non-recognized values become <NA>.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    s = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f"}

    result = pd.Series(pd.NA, index=s.index, dtype="boolean")
    result.loc[s.isin(true_vals)] = True
    result.loc[s.isin(false_vals)] = False
    return result


def filter_exact_matches(input_csv: Path, output_csv: Optional[Path] = None) -> Path:
    """Read `input_csv`, filter rows where `exact_match` is True, write to `output_csv`.

    Returns the output path used.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    logger.info("Reading CSV: %s", input_csv)
    df = pd.read_csv(input_csv)

    if "exact_match" not in df.columns:
        raise KeyError("Column 'exact_match' not found in input CSV")

    em = _coerce_exact_match(df["exact_match"])  # nullable boolean
    mask = em.fillna(False).astype(bool)
    filtered = df.loc[mask].copy()

    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_exact_matches.csv")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d rows to: %s", len(filtered), output_csv)
    filtered.to_csv(output_csv, index=False)
    return output_csv


@dataclass
class Args:
    """CLI arguments for extracting exact-match rows."""

    input_csv: Path
    output_csv: Optional[Path] = None


def main(args: Args) -> None:
    out = filter_exact_matches(args.input_csv, args.output_csv)
    logger.info("Done. Output saved at: %s", out)


if __name__ == "__main__":
    main(tyro.cli(Args))


