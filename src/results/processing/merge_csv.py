"""Merge rows from CSV B into CSV A, overwriting matching rows by id/file_name.

How to run (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.results.processing.merge_csv data\base.csv data\override.csv data\merged.csv

- Direct script:
  python src\results\processing\merge_csv.py data\base.csv data\override.csv data\merged.csv

Example:
  python -m src.results.processing.merge_csv data\2026_01_20_llama_1.csv data\new_results.csv data\merged_results.csv
"""

import pandas as pd
from pathlib import Path
import tyro


def main(file_a: Path, file_b: Path, output_file: Path) -> None:
    """Merge CSV files, overwriting rows in A with matching rows from B.
    
    Args:
        file_a: Base CSV file (rows will be overwritten if matching)
        file_b: Override CSV file (rows from here will replace matching rows in A)
        output_file: Output merged CSV file
    """
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    
    # Create composite key from id and file_name
    df_a['_merge_key'] = df_a['id'].astype(str) + '|' + df_a['file_name'].astype(str)
    df_b['_merge_key'] = df_b['id'].astype(str) + '|' + df_b['file_name'].astype(str)
    
    # Get unique keys from B
    keys_in_b = set(df_b['_merge_key'])
    
    # Count how many rows in A will be replaced
    rows_to_replace = df_a['_merge_key'].isin(keys_in_b).sum()
    
    # Remove rows from A that exist in B
    df_a_filtered = df_a[~df_a['_merge_key'].isin(keys_in_b)]
    
    # Concatenate: A (without matching rows) + B (all rows)
    df_merged = pd.concat([df_a_filtered, df_b], ignore_index=True)
    
    # Remove the temporary merge key column
    df_merged = df_merged.drop(columns=['_merge_key'])
    
    # Sort by id for consistency
    df_merged = df_merged.sort_values(by=['id', 'file_name']).reset_index(drop=True)
    
    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_file, index=False)
    
    # Print results
    print({
        "file_a": str(file_a),
        "file_b": str(file_b),
        "output": str(output_file),
        "rows_in_a": int(len(df_a)),
        "rows_in_b": int(len(df_b)),
        "rows_replaced_from_a": int(rows_to_replace),
        "rows_copied_from_b": int(len(df_b)),
        "total_rows_in_output": int(len(df_merged)),
    })


if __name__ == "__main__":
    args = tyro.cli(tuple[Path, Path, Path])
    main(*args)
