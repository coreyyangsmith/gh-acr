from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_RESULTS_GLOB = "data/*_results_all.csv"


@dataclass
class ResultsData:
    dataframe: pd.DataFrame
    path: Path


def load_results(path: str | Path | None = None) -> ResultsData:
    """Load consolidated results CSV.

    If path is None, pick the most recent file matching DEFAULT_RESULTS_GLOB.
    Ensures expected dtypes and derived columns exist.
    """
    path_obj: Path
    if path is None:
        candidates: list[Path] = sorted(
            Path.cwd().glob(DEFAULT_RESULTS_GLOB), key=lambda p: p.name, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(
                "No results CSV found. Expected something like data/YYYY_MM_DD_results_all.csv"
            )
        path_obj = candidates[0]
    else:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Results CSV not found: {path_obj}")

    df = pd.read_csv(path_obj)

    # Normalize/ensure columns
    expected_bool = ["exact_match"]
    for col in expected_bool:
        if col in df.columns:
            # Cast robustly from strings like 'True'/'False' or 1/0
            if df[col].dtype != bool:
                df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    # Derivations
    if "eval_method" not in df.columns:
        raise ValueError("Results file must include 'eval_method' column")

    if "tokens_in" not in df.columns and {"tokens_total_input"}.issubset(df.columns):
        df["tokens_in"] = df["tokens_total_input"]
    if "tokens_out" not in df.columns and {"tokens_output"}.issubset(df.columns):
        df["tokens_out"] = df["tokens_output"]

    # Derived metrics (row-level; aggregate-level variants computed in tables/stats)
    if {"tokens_in", "tokens_out"}.issubset(df.columns):
        df["tokens_per_sec"] = _safe_ratio(df["tokens_in"] + df["tokens_out"], df["processing_time_s"])
    else:
        df["tokens_per_sec"] = pd.NA
    if {"similarity", "total_cost"}.issubset(df.columns):
        df["quality_per_dollar"] = _safe_ratio(df["similarity"], df["total_cost"].replace(0, pd.NA))
    else:
        df["quality_per_dollar"] = pd.NA

    return ResultsData(dataframe=df, path=path_obj)


def _safe_ratio(numer: Iterable, denom: Iterable) -> pd.Series:
    import numpy as np

    numer_s = pd.Series(numer)
    denom_s = pd.Series(denom)
    
    # Handle NAType values before casting to float
    numer_s = numer_s.fillna(pd.NA)
    denom_s = denom_s.fillna(pd.NA)
    
    # Convert to float, handling pd.NA properly
    try:
        numer_float = pd.to_numeric(numer_s, errors='coerce')
        denom_float = pd.to_numeric(denom_s, errors='coerce')
    except Exception:
        # Fallback: mask NA values before conversion
        numer_mask = pd.isna(numer_s)
        denom_mask = pd.isna(denom_s)
        
        numer_float = numer_s.copy().astype(object)
        denom_float = denom_s.copy().astype(object)
        
        numer_float[~numer_mask] = pd.to_numeric(numer_s[~numer_mask], errors='coerce')
        denom_float[~denom_mask] = pd.to_numeric(denom_s[~denom_mask], errors='coerce')
        
        numer_float = pd.to_numeric(numer_float, errors='coerce')
        denom_float = pd.to_numeric(denom_float, errors='coerce')
    
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numer_float / denom_float
    return result.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], pd.NA)
