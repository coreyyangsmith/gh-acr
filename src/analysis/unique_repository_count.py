"""Count unique repositories from a results CSV and print the total.

This small utility expects a `repo` column and can be extended as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import tyro


def main(path: Optional[Path] = None) -> None:
    if path is None:
        raise SystemExit("Provide --path to a results CSV containing a 'repo' column.")
    df = pd.read_csv(path)
    if "repo" not in df.columns:
        raise SystemExit("Missing 'repo' column in input CSV")
    n = int(df["repo"].astype(str).nunique())
    print({"input": str(path), "unique_repos": n})


if __name__ == "__main__":
    p = tyro.cli(Optional[Path])
    main(p)

