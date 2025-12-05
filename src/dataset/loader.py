"""Utilities for loading the GitGoodBench benchmark CSV.

This module provides functions for loading and preprocessing the benchmark
dataset used for evaluating merge conflict resolution methods. It handles
the specific format of GitGoodBench CSV files and normalizes the data for
use throughout the pipeline.

Dataset Format
--------------
The GitGoodBench CSV has the following structure:
- First column: Unnamed index (from pandas export)
- `id`: Unique scenario identifier
- `name`: Repository slug (e.g., "owner/repo")
- `scenario`: JSON string with conflict metadata
- `difficulty`: Optional difficulty rating (easy/medium/hard)
- Additional columns may be present depending on the dataset version

The `scenario` column contains a JSON object with:
- `files_in_merge_conflict`: List of conflicting file paths
- `parents`: List of two parent commit SHAs [parent_a, parent_b]
- `merge_commit_hash`: SHA of the ground-truth merge commit

Example Usage
-------------
>>> from src.dataset.loader import load_benchmark, DATA_PATH
>>> 
>>> # Load default dataset
>>> df = load_benchmark()
>>> print(f"Loaded {len(df)} scenarios")
>>> 
>>> # Load a specific dataset
>>> df = load_benchmark("data/custom_subset.csv")
>>> 
>>> # Access scenario metadata
>>> row = df.iloc[0]
>>> files = row["scenario_json"]["files_in_merge_conflict"]
>>> parents = row["scenario_json"]["parents"]

Notes
-----
- The loader is designed for data loading only - no side effects
- Scenario JSON is automatically parsed into a dict column
- ID column is ensured to exist and be string type
- The module is safe to import and use without network access
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.config.settings import DATA_PATH


__all__ = [
    "DATA_PATH",
    "load_benchmark",
]


def load_benchmark(csv_path: str | Path | None = None, /) -> DataFrame:
    """Load the GitGoodBench dataset into a pandas DataFrame.

    This function loads the benchmark CSV and performs necessary preprocessing:
    1. Drops the unnamed index column (pandas export artifact)
    2. Parses the scenario JSON string into a dict
    3. Ensures an ID column exists (creates from index if missing)

    Parameters
    ----------
    csv_path
        Optional explicit path to the CSV file. If None (default), falls back
        to DATA_PATH defined in src.config.settings.

    Returns
    -------
    DataFrame
        Preprocessed dataframe with columns:
        - id: str - Unique scenario identifier
        - name: str - Repository slug (owner/repo)
        - scenario: str - Original JSON string
        - scenario_json: dict - Parsed scenario metadata
        - difficulty: str (optional) - Difficulty rating

    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist at the specified path.
    ValueError
        If the scenario column contains malformed JSON.

    Examples
    --------
    >>> # Load default dataset
    >>> df = load_benchmark()
    >>> print(df.columns.tolist())
    ['id', 'name', 'scenario', 'scenario_json', 'difficulty', ...]

    >>> # Access a specific scenario
    >>> scenario = df.iloc[0]["scenario_json"]
    >>> print(scenario["files_in_merge_conflict"])
    ['src/main.py', 'src/utils.py']

    >>> # Filter by difficulty
    >>> easy_df = df[df["difficulty"] == "easy"]

    Notes
    -----
    - The function uses pandas' index_col=0 to handle the unnamed index
    - Scenario parsing uses ast.literal_eval for safe evaluation
    - All IDs are converted to strings for consistent handling
    """
    path: Path = Path(csv_path or DATA_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"GitGoodBench CSV not found: {path}")

    # Load with index_col=0 to handle pandas-exported CSVs
    df = pd.read_csv(path, index_col=0)

    # Parse the scenario column if present
    if "scenario" in df.columns:
        df["scenario_json"] = df["scenario"].map(_parse_scenario)

    # Ensure ID column exists and is string type
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    else:
        df["id"] = df["id"].astype(str)

    return df


def _parse_scenario(raw: str) -> dict:
    """Parse a scenario JSON string into a dictionary.

    The scenario column in GitGoodBench uses Python dict syntax
    (single quotes) rather than strict JSON (double quotes).
    We use ast.literal_eval for safe parsing.

    Parameters
    ----------
    raw
        The raw scenario string from the CSV.

    Returns
    -------
    dict
        Parsed scenario metadata.

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Unable to parse 'scenario' JSON: {raw[:100]}...") from exc
