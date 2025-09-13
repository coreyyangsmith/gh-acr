import argparse
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add or update a difficulty column for all rows in a CSV file. "
            "By default, modifies the input file in place."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        type=str.lower,
        required=True,
        choices=["easy", "medium", "hard"],
        help="Difficulty value to write to the column",
    )
    parser.add_argument(
        "-c",
        "--column-name",
        default="difficulty",
        help="Name of the column to add/update (default: difficulty)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write the updated CSV. If omitted, the input file is modified in place."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for reading/writing the CSV (default: utf-8)",
    )
    parser.add_argument(
        "-D",
        "--delimiter",
        default=None,
        help=(
            "Optional delimiter to force (e.g., , ; \t |). If omitted, the script tries to sniff it."
        ),
    )
    return parser.parse_args()


def _sniff_dialect(sample: str) -> Optional[csv.Dialect]:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return None


def add_difficulty_to_csv(
    input_path: Path,
    difficulty: str,
    column_name: str = "difficulty",
    output_path: Optional[Path] = None,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Prepare output destination
    in_place: bool = output_path is None

    # Open input and determine dialect/delimiter
    with input_path.open("r", encoding=encoding, newline="") as infile:
        # Try to sniff the dialect if no delimiter is provided
        sniffed_dialect: Optional[csv.Dialect] = None
        if delimiter is None:
            sample: str = infile.read(4096)
            sniffed_dialect = _sniff_dialect(sample)
            infile.seek(0)

        # Configure CSV reader
        reader_kwargs = {"fieldnames": None}
        if sniffed_dialect is not None:
            reader_kwargs["dialect"] = sniffed_dialect
        elif delimiter is not None:
            reader_kwargs["delimiter"] = delimiter

        reader = csv.DictReader(infile, **reader_kwargs)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("CSV appears to have no header row.")

        # Prepare writer destination: temp file if in-place, else direct to output
        if in_place:
            temp_dir = input_path.parent
            with NamedTemporaryFile(
                mode="w",
                encoding=encoding,
                newline="",
                delete=False,
                dir=temp_dir,
                prefix=input_path.stem + ".",
                suffix=".tmp",
            ) as tmpfile:
                tmp_path = Path(tmpfile.name)

                # Configure writer
                if column_name not in fieldnames:
                    fieldnames.append(column_name)

                writer_kwargs = {"fieldnames": fieldnames}
                if sniffed_dialect is not None:
                    writer_kwargs["dialect"] = sniffed_dialect
                elif delimiter is not None:
                    writer_kwargs["delimiter"] = delimiter

                writer = csv.DictWriter(tmpfile, **writer_kwargs)
                writer.writeheader()

                for row in reader:
                    row[column_name] = difficulty
                    writer.writerow(row)

            # Replace original file atomically
            os.replace(str(tmp_path), str(input_path))
            return input_path
        else:
            assert output_path is not None
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding=encoding, newline="") as outfile:
                # Configure writer
                if column_name not in fieldnames:
                    fieldnames.append(column_name)

                writer_kwargs = {"fieldnames": fieldnames}
                if sniffed_dialect is not None:
                    writer_kwargs["dialect"] = sniffed_dialect
                elif delimiter is not None:
                    writer_kwargs["delimiter"] = delimiter

                writer = csv.DictWriter(outfile, **writer_kwargs)
                writer.writeheader()

                for row in reader:
                    row[column_name] = difficulty
                    writer.writerow(row)

            return output_path


def main() -> None:
    args = parse_args()
    result_path = add_difficulty_to_csv(
        input_path=args.input_csv,
        difficulty=args.difficulty,
        column_name=args.column_name,
        output_path=args.output,
        encoding=args.encoding,
        delimiter=args.delimiter,
    )
    print(f"Wrote difficulty='{args.difficulty}' to column '{args.column_name}' in: {result_path}")


if __name__ == "__main__":
    main()


