"""TABLE I: Dataset summary — difficulty × project-size cross-tabulation.

Produces a LaTeX-ready CSV and prints a formatted table.

Usage::

    python -m src.results.final_paper_figs.table_1_dataset_summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running as a standalone script or as a module
try:
    from src.results.final_paper_figs.shared import (
        DATASET_CSV, DIFF_ORDER, OUTPUT_DIR, SIZE_ORDER, load_dataset, logger,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.final_paper_figs.shared import (
        DATASET_CSV, DIFF_ORDER, OUTPUT_DIR, SIZE_ORDER, load_dataset, logger,
    )


def generate_table_1(dataset_csv: Path | None = None,
                      output_dir: Path | None = None) -> pd.DataFrame:
    """Generate TABLE I: difficulty × project_size cross-tabulation."""
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("TABLE I: Dataset summary")
    ds = load_dataset(dataset_csv)

    # Normalize to lowercase for safe matching
    ds["difficulty"] = ds["difficulty"].str.lower().str.strip()
    ds["project_size"] = ds["project_size"].str.lower().str.strip()

    # Cross-tabulation
    ct = pd.crosstab(
        ds["difficulty"],
        ds["project_size"],
        margins=True,
        margins_name="Total",
    )

    # Reorder rows and columns
    row_order = [d for d in DIFF_ORDER if d in ct.index] + ["Total"]
    col_order = [s for s in SIZE_ORDER if s in ct.columns] + ["Total"]
    ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

    # Capitalize labels for display
    ct.index = [d.capitalize() for d in ct.index]
    ct.columns = [c.capitalize() for c in ct.columns]
    ct.index.name = "Difficulty"

    # Save CSV
    csv_path = out / "Table_I_dataset_summary.csv"
    ct.to_csv(csv_path)
    logger.info(f"  Saved {csv_path.name}")

    # Print formatted table
    print("\n" + "=" * 60)
    print("TABLE I: Dataset Summary")
    print("=" * 60)
    print(ct.to_string())
    print("=" * 60 + "\n")

    return ct


if __name__ == "__main__":
    generate_table_1()
