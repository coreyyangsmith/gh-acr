from __future__ import annotations

"""Utilities for loading the GitGoodBench benchmark CSV.

This module focuses on **data loading only** – it has **no external side

effects**, making it easy to unit-test and reuse throughout the codebase.
"""

from pathlib import Path
import ast
import pandas as pd
from pandas import DataFrame

__all__ = [
    "DATA_PATH",
    "load_benchmark",
]

from src.config.settings import DATA_PATH

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_benchmark(csv_path: str | Path | None = None, /) -> DataFrame:  # noqa: D401 – imperative mood is fine here
    """Load *GitGoodBench* into a ``pandas`` DataFrame.

    Parameters
    ----------
    csv_path
        Optional explicit path to the CSV.  If *None* (default) the function
        falls back to :data:`DATA_PATH` defined in `src.config.settings` (can be
        overridden with the DATASET_CSV environment variable).

    Notes
    -----
    1. The CSV was exported with *pandas* which prepends an unnamed index
       column.  We drop that column to avoid confusion.
    2. The *scenario* column is a stringified ``dict`` that uses single quotes.
       We convert it into a real ``dict`` and expose it via a new column
       *scenario_json* while keeping the original string unchanged for any
       downstream code that may rely on it.

    Returns
    -------
    pandas.DataFrame
        Fully-typed dataframe ready for further processing.
    """

    path: Path = Path(csv_path or DATA_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"GitGoodBench CSV not found: {path}")

    df = pd.read_csv(path, index_col=0)

    # Normalise the scenario column ------------------------------------------------
    def _parse_scenario(raw: str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:  # pragma: no cover – helpful error
            raise ValueError(f"Unable to parse 'scenario' JSON for row: {raw}") from exc

    # Parse scenario column if present
    if "scenario" in df.columns:
        df["scenario_json"] = df["scenario"].map(_parse_scenario)
    
    # Ensure an 'id' column exists. If absent, use the CSV index as id.
    # Always cast to string for robustness across numeric/string ids.
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    else:
        df["id"] = df["id"].astype(str)

    return df
