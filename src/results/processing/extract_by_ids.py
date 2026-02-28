"""Extract rows from a CSV that match IDs from a text file.

How to run (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.results.processing.extract_by_ids data\invalid_ids.txt data\main.csv data\extracted.csv

- With custom ID column:
  python -m src.results.processing.extract_by_ids data\invalid_ids.txt data\main.csv data\extracted.csv --id-column "Unnamed: 0"

Example:
  python -m src.results.processing.extract_by_ids data\invalid_ids.txt data\git_good_bench_merge_commits_all_working.csv data\invalid_rows.csv
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    ids_file: Path  # Text file with one ID per line
    input_csv: Path  # Main CSV file to extract from
    output_csv: Path  # Output CSV file with matching rows
    id_column: Optional[str] = None  # Column name to match IDs against (auto-detected if not specified)


def main(args: Args) -> None:
    """Extract rows from CSV that match IDs from a text file."""
    # Read IDs from text file
    with open(args.ids_file, 'r') as f:
        ids_to_match = set(line.strip() for line in f if line.strip())
    
    # Convert to integers for matching (IDs in CSV are typically integers)
    ids_to_match_int = set()
    for id_val in ids_to_match:
        try:
            ids_to_match_int.add(int(id_val))
        except ValueError:
            # Keep as string if not convertible
            ids_to_match_int.add(id_val)
    
    # Read CSV
    df = pd.read_csv(args.input_csv)
    
    # Determine which column to use for ID matching
    id_col = args.id_column
    if id_col is None:
        # Auto-detect: check if first column is unnamed (contains numeric IDs)
        first_col = df.columns[0]
        if first_col.startswith('Unnamed'):
            id_col = first_col
        elif 'id' in df.columns:
            # Check if 'id' column contains numeric values matching our IDs
            sample_id = next(iter(ids_to_match_int))
            if df['id'].dtype in ['int64', 'float64'] or df['id'].iloc[0] == sample_id:
                id_col = 'id'
            else:
                id_col = first_col
        else:
            id_col = first_col
    
    print(f"Using column '{id_col}' for ID matching")
    
    # Filter rows where id matches
    extracted_df = df[df[id_col].isin(ids_to_match_int)]
    
    # Save output
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    extracted_df.to_csv(args.output_csv, index=False)
    
    # Print results
    print({
        "ids_file": str(args.ids_file),
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "id_column": id_col,
        "ids_in_file": len(ids_to_match),
        "rows_in_csv": len(df),
        "rows_extracted": len(extracted_df),
        "unique_ids_extracted": int(extracted_df[id_col].nunique()),
    })


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
