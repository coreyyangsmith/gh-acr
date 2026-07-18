"""Count unique ID values in a CSV file.

Usage examples (from repo root):
- Module form (recommended):
    python -m src.analysis.processing.count_unique_ids data/2025_10_18_ALL_RESULTS.csv

- Direct script:
    python src/analysis/processing/count_unique_ids.py data/input.csv --id-column id

Behavior:
- Detects the ID column (default: "id") with forgiving fallbacks
- Streams the file in chunks, counting non-empty and unique ID values
- Prints a concise summary dictionary to stdout
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro


try:
    # Prefer project logger when available (works when executed as a module)
    from utils.logger import logger as app_logger  # type: ignore
    logger = app_logger
except Exception:  # pragma: no cover - fallback when run directly as a script
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


def _detect_id_column(df: pd.DataFrame, requested: str) -> str:
    """Return the column to use as the ID, with forgiving fallbacks.

    Preference order:
    1) exact match of requested
    2) case-insensitive match of "id"
    3) common variants (contains "id", or equals "sample_id")
    """
    if requested in df.columns:
        return requested
    lowered = {c.lower(): c for c in df.columns}
    if "id" in lowered:
        return lowered["id"]
    candidates = [c for c in df.columns if c.lower() in {"sample_id"} or "id" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"ID column '{requested}' not found and no '*id*' column detected in input CSV."
    )


@dataclass
class Flags:
    input_csv: Path
    id_column: str = "id"
    delimiter: Optional[str] = None  # Auto-detect when None
    encoding: str = "utf-8"
    chunksize: int = 200_000


def main(flags: Flags) -> None:
    input_csv = flags.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Detect the actual ID column name using header only
    header_df = pd.read_csv(
        input_csv,
        sep=flags.delimiter if flags.delimiter is not None else None,
        engine="python" if flags.delimiter is None else "c",
        encoding=flags.encoding,
        nrows=0,
    )
    detected_id_col = _detect_id_column(header_df, flags.id_column)

    logger.info(
        "Counting unique IDs: input=%s, id_column=%s",
        str(input_csv),
        detected_id_col,
    )

    engine = "python" if flags.delimiter is None else "c"
    read_kwargs = {
        "sep": flags.delimiter if flags.delimiter is not None else None,
        "engine": engine,
        "encoding": flags.encoding,
        "chunksize": flags.chunksize,
        "usecols": [detected_id_col],
        "dtype": {detected_id_col: str},
    }

    total_rows = 0
    empty_ids = 0
    non_empty_total = 0
    unique_ids: set[str] = set()

    for chunk in pd.read_csv(input_csv, **read_kwargs):
        total_rows += len(chunk)
        series = chunk[detected_id_col].astype(str).fillna("").str.strip()
        empties = (series == "").sum()
        empty_ids += int(empties)
        non_empty = series[series != ""]
        non_empty_total += int(len(non_empty))
        if not non_empty.empty:
            unique_ids.update(non_empty.tolist())

    unique_count = len(unique_ids)
    duplicates = max(0, non_empty_total - unique_count)

    summary = {
        "input": str(input_csv),
        "id_column": detected_id_col,
        "rows": int(total_rows),
        "non_empty_ids": int(non_empty_total),
        "unique_ids": int(unique_count),
        "duplicate_ids": int(duplicates),
        "empty_ids": int(empty_ids),
    }
    print(summary)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)



