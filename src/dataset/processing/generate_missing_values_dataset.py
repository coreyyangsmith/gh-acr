#!/usr/bin/env python3
"""
Generate a CSV of rows from original-data whose first column value does not
appear in the 'id' column of processed-results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export rows from original-data whose first column is not present "
            "in the processed-results 'id' column."
        )
    )
    parser.add_argument(
        "--original-data",
        required=True,
        help="Path to the original CSV file. The first column is used as the key.",
    )
    parser.add_argument(
        "--processed-results",
        required=True,
        help="Path to the processed results CSV with an 'id' column.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the filtered CSV.",
    )
    return parser.parse_args()


def sniff_csv_format(file_path: Path) -> Tuple[csv.Dialect, bool]:
    sample_size = 65536
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(sample_size)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            except Exception:
                dialect = csv.excel
            try:
                has_header = csv.Sniffer().has_header(sample) if sample else False
            except Exception:
                has_header = False
    except FileNotFoundError:
        raise
    return dialect, has_header


def detect_id_fieldname(fieldnames: Optional[list[str]]) -> Optional[str]:
    if not fieldnames:
        return None
    for name in fieldnames:
        if name is None:
            continue
        if name.strip().lower() == "id":
            return name
    return None


def load_processed_ids(processed_results_path: Path) -> Set[str]:
    dialect, _ = sniff_csv_format(processed_results_path)
    ids: Set[str] = set()
    with open(processed_results_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(
                "processed-results CSV appears to be missing a header row."
            )
        id_fieldname = detect_id_fieldname(reader.fieldnames)
        if id_fieldname is None:
            raise ValueError(
                f"processed-results CSV does not contain an 'id' column. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            if not row:
                continue
            raw_value = row.get(id_fieldname)
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if value:
                ids.add(value)
    return ids


def filter_original_and_write(
    original_data_path: Path, output_csv_path: Path, processed_ids: Set[str]
) -> Tuple[int, int, bool]:
    dialect, has_header = sniff_csv_format(original_data_path)
    total_rows = 0
    written_rows = 0
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(original_data_path, "r", encoding="utf-8-sig", newline="") as src, open(
        output_csv_path, "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src, dialect=dialect)
        writer = csv.writer(dst, dialect=dialect)
        header_row: Optional[list[str]] = None
        if has_header:
            try:
                header_row = next(reader)
            except StopIteration:
                header_row = None
        if header_row is not None:
            writer.writerow(header_row)
        for row in reader:
            if not row:
                continue
            total_rows += 1
            first_value = str(row[0]).strip() if row else ""
            if first_value and first_value in processed_ids:
                continue
            writer.writerow(row)
            written_rows += 1
    return total_rows, written_rows, has_header


def main() -> None:
    args = parse_args()
    original_path = Path(args.original_data)
    processed_path = Path(args.processed_results)
    output_path = Path(args.output_csv)

    if not original_path.exists():
        raise FileNotFoundError(f"original-data not found: {original_path}")
    if not processed_path.exists():
        raise FileNotFoundError(f"processed-results not found: {processed_path}")

    processed_ids = load_processed_ids(processed_path)
    scanned_rows, written_rows, had_header = filter_original_and_write(
        original_path, output_path, processed_ids
    )
    print(
        f"Loaded {len(processed_ids)} unique ids from processed-results. "
        f"Scanned {scanned_rows} data rows from original-data"
        f"{'' if not had_header else ' (header detected)'} . "
        f"Wrote {written_rows} rows to {output_path}."
    )


if __name__ == "__main__":
    main()

