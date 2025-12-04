"""CLI tool to keep only the first N columns of a tabular CSV dataset.

Usage examples:
  - From repo root:
      python -m src.results.processing.remove_extra_columns --input data/2025_10_18_ALL_RESULTS.csv --max-columns 20
  - With explicit output path:
      python -m src.results.processing.remove_extra_columns -i data/in.csv -n 12 -o data/in_first12cols.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import csv

import pandas as pd

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


def infer_default_output_path(input_path: Path, max_columns: int) -> Path:
    """Return a sensible default output path based on the input path and N.

    Example: data/in.csv -> data/in_first20cols.csv
    """
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"
    return input_path.with_name(f"{stem}_first{max_columns}cols{suffix}")


def trim_columns(
    input_file: Path,
    max_columns: int,
    output_file: Optional[Path] = None,
    delimiter: Optional[str] = None,
    chunksize: int = 200_000,
    encoding: str = "utf-8",
) -> Path:
    """Trim a CSV to the first `max_columns` columns and write to `output_file`.

    Args:
        input_file: Path to the input CSV file.
        max_columns: Maximum number of columns to keep (must be >= 1).
        output_file: Optional explicit output path. If None, a default is derived.
        delimiter: Optional field delimiter for reading; if None, let pandas infer.
        chunksize: Number of rows per chunk for streaming processing.
        encoding: File encoding used for reading and writing.

    Returns:
        Path to the written output file.
    """
    if max_columns < 1:
        raise ValueError("max_columns must be >= 1")

    if output_file is None:
        output_file = infer_default_output_path(input_file, max_columns)

    engine = "python" if delimiter is None else "c"
    read_kwargs = {
        "sep": delimiter if delimiter is not None else None,
        "engine": engine,
        "encoding": encoding,
        "chunksize": chunksize,
        "dtype": None,
    }

    logger.info(
        "Trimming columns: input=%s, max_columns=%d, output=%s",
        str(input_file),
        max_columns,
        str(output_file),
    )

    # Stream through the file and write trimmed columns incrementally
    first_chunk = True
    total_rows = 0

    try:
        for chunk in pd.read_csv(input_file, **read_kwargs):
            # Select first N columns by position to preserve order
            selected_columns = list(chunk.columns[:max_columns])
            trimmed = chunk[selected_columns]

            trimmed.to_csv(
                output_file,
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                encoding=encoding,
            )

            first_chunk = False
            total_rows += len(trimmed)

        logger.info("Done. Wrote %d rows to %s", total_rows, str(output_file))
        return output_file
    except Exception as exc:  # Fallback for ragged/irregular rows
        logger.warning(
            "Pandas streaming failed (likely irregular column counts). Falling back to CSV reader. Error: %s",
            exc,
        )

        return _trim_columns_ragged(
            input_file=input_file,
            max_columns=max_columns,
            output_file=output_file,
            delimiter=delimiter,
            encoding=encoding,
        )


def _trim_columns_ragged(
    input_file: Path,
    max_columns: int,
    output_file: Path,
    delimiter: Optional[str],
    encoding: str,
) -> Path:
    """Robust fallback for CSVs with uneven column counts per row.

    This implementation uses Python's csv module, slicing rows to the first
    N fields and padding shorter rows with empty strings to ensure a stable
    column count. It detects the delimiter when not explicitly provided.
    """
    # Detect delimiter when not provided
    detected_dialect: Optional[csv.Dialect] = None
    detected_delimiter: str = ","

    with input_file.open("r", encoding=encoding, newline="") as in_fp:
        # Decide dialect/delimiter for reader and writer
        chosen_delimiter: Optional[str] = delimiter
        if chosen_delimiter is None:
            try:
                sample = in_fp.read(64 * 1024)
                in_fp.seek(0)
                detected_dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                detected_delimiter = detected_dialect.delimiter  # type: ignore[attr-defined]
            except Exception:
                detected_dialect = None
                detected_delimiter = ","
            finally:
                if chosen_delimiter is None:
                    chosen_delimiter = detected_delimiter

        if detected_dialect is not None and delimiter is None:
            reader = csv.reader(in_fp, dialect=detected_dialect)
        else:
            reader = csv.reader(in_fp, delimiter=chosen_delimiter or ",")

        # Prepare writer and header (mirror reader settings)
        with output_file.open("w", encoding=encoding, newline="") as out_fp:
            if detected_dialect is not None and delimiter is None:
                writer = csv.writer(out_fp, dialect=detected_dialect, lineterminator="\n")
            else:
                writer = csv.writer(out_fp, delimiter=chosen_delimiter or ",", lineterminator="\n")

            try:
                header_row = next(reader)
            except StopIteration:
                # Empty file: write nothing and return
                logger.info("Input appears empty. Wrote 0 rows to %s", str(output_file))
                return output_file

            # Trim/pad header to exactly N columns
            if len(header_row) >= max_columns:
                header_out = header_row[:max_columns]
            else:
                header_out = header_row + [""] * (max_columns - len(header_row))
            writer.writerow(header_out)

            total_rows = 0
            for row in reader:
                if len(row) >= max_columns:
                    row_out = row[:max_columns]
                else:
                    row_out = row + [""] * (max_columns - len(row))
                writer.writerow(row_out)
                total_rows += 1

    logger.info("Done (ragged fallback). Wrote %d rows to %s", total_rows, str(output_file))
    return output_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove columns beyond N (keep only first N columns) from a CSV file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "-n",
        "--max-columns",
        dest="max_columns",
        type=int,
        required=True,
        help="Maximum number of columns to keep (>=1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="Optional output CSV path. Defaults to <input>_firstNcols.csv",
    )
    parser.add_argument(
        "--delimiter",
        dest="delimiter",
        default=None,
        help="Optional field delimiter (e.g. ',' or '\t'). If omitted, infer.",
    )
    parser.add_argument(
        "--chunksize",
        dest="chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk for streaming (default: 200k)",
    )
    parser.add_argument(
        "--encoding",
        dest="encoding",
        default="utf-8",
        help="File encoding (default: utf-8)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path: Optional[Path] = Path(args.output_path) if args.output_path else None

    trim_columns(
        input_file=input_path,
        max_columns=args.max_columns,
        output_file=output_path,
        delimiter=args.delimiter,
        chunksize=args.chunksize,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()


