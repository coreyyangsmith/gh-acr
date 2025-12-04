"""Split an input CSV into multiple CSVs by unique values of a column.

Usage (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.results.processing.split_final_dataset_by_model data\2025_10_18_ALL_RESULTS.csv --column model_name

- Direct script:
  python src\results\processing\split_final_dataset_by_model.py data\input.csv --column model_name

By default, outputs are written under a subdirectory named
<input_stem>_by_<column> next to the input file, with filenames of the form
<input_stem>__<sanitized_value>.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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


def _sanitize_value_for_filename(value: str) -> str:
    """Return a filesystem-friendly representation of a value.

    - Trims whitespace
    - Replaces whitespace runs with underscores
    - Keeps alphanumerics, dash, underscore, and dot; drops other chars
    - Collapses repeated underscores
    """
    import re

    s = str(value).strip()
    if s == "":
        return "empty"
    # Normalize whitespace -> underscore
    s = re.sub(r"\s+", "_", s)
    # Keep a safe subset of characters
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    # Avoid accidental empties after stripping
    if s == "":
        return "unknown"
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s


@dataclass
class Flags:
    """Parameters controlling input, grouping, and outputs."""

    input_csv: Path
    output_dir: Optional[Path] = None
    column: str = "model_name"
    chunksize: int = 200_000
    encoding: str = "utf-8"
    delimiter: Optional[str] = None
    drop_empty: bool = True


def _resolve_output_dir(input_csv: Path, output_dir: Optional[Path], group_column: str) -> Path:
    if output_dir is not None:
        return output_dir
    # Default: <input_parent>/<input_stem>_by_<column>
    return input_csv.parent / f"{input_csv.stem}_by_{_sanitize_value_for_filename(group_column)}"


def split_by_column(flags: Flags) -> Dict[str, int]:
    """Split CSV rows into per-value files under the output directory.

    Returns a dict mapping sanitized group value -> rows written.
    """
    input_csv = flags.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir = _resolve_output_dir(input_csv, flags.output_dir, flags.column)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare reading options
    engine = "python" if flags.delimiter is None else "c"
    read_kwargs = {
        "sep": flags.delimiter if flags.delimiter is not None else None,
        "engine": engine,
        "encoding": flags.encoding,
        "chunksize": flags.chunksize,
        "dtype": None,
    }

    logger.info(
        "Splitting CSV by column: input=%s, column=%s, output_dir=%s",
        str(input_csv),
        flags.column,
        str(output_dir),
    )

    header_written: Dict[str, bool] = {}
    rows_per_value: Dict[str, int] = {}
    detected_column: Optional[str] = None

    total_rows = 0
    for chunk in pd.read_csv(input_csv, **read_kwargs):
        # Determine column on first chunk (with fallback detection)
        if detected_column is None:
            detected_column = _detect_column(chunk, flags.column)

        # Filter out empty/NA values if requested
        work = chunk.copy()
        if flags.drop_empty:
            non_na = ~work[detected_column].isna()
            work = work[non_na]
            if not work.empty:
                non_empty = work[detected_column].astype(str).str.strip() != ""
                work = work[non_empty]

        if work.empty:
            continue

        # Group by the detected column and append to respective files
        for raw_value, group_df in work.groupby(detected_column):
            safe_value = _sanitize_value_for_filename(str(raw_value))
            out_path = output_dir / f"{input_csv.stem}__{safe_value}.csv"

            write_header = not header_written.get(safe_value, False) and not out_path.exists()
            group_df.to_csv(
                out_path,
                index=False,
                mode="a",
                header=write_header,
                encoding=flags.encoding,
            )

            header_written[safe_value] = True
            rows_per_value[safe_value] = rows_per_value.get(safe_value, 0) + int(len(group_df))
            total_rows += int(len(group_df))

    logger.info(
        "Done. Wrote %d rows into %d files under %s",
        total_rows,
        len(rows_per_value),
        str(output_dir),
    )

    return rows_per_value


def main(flags: Flags) -> None:
    rows_per_value = split_by_column(flags)
    # Print concise summary suitable for quick inspection
    summary = {
        "input": str(flags.input_csv),
        "output_dir": str(_resolve_output_dir(flags.input_csv, flags.output_dir, flags.column)),
        "column": flags.column,
        "files": len(rows_per_value),
        "rows_per_file": rows_per_value,
    }
    print(summary)


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


