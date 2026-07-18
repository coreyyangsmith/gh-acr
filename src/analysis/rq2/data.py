"""Data preparation utilities for RQ2 analyses.

Handles pairing single-agent and multi-agent results, computing improvement deltas,
creating buckets for stratification, and preparing data for regression analysis.

Supports two granularity levels:
- Per-file (default): Each row in the CSV is treated independently
- Per-instance: Files are grouped by 'id', with EM requiring all files to match
  and soft metrics averaged across files
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from .config import (
    RQ2Config,
    DEFAULT_CONFIG,
    FILE_TYPE_CATEGORIES,
    DEFAULT_CONFLICT_SIZE_BUCKETS,
    DEFAULT_CONTEXT_SIZE_BUCKETS,
)

# Type alias for granularity
GranularityType = Literal["file", "instance"]


@dataclass
class ImprovementData:
    """Container for paired improvement data.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Dataframe with improvement delta columns
    n_pairs : int
        Number of paired scenarios
    single_method : str
        Single-agent method identifier
    multi_method : str
        Multi-agent method identifier
    """

    dataframe: pd.DataFrame
    n_pairs: int
    single_method: str
    multi_method: str


@dataclass
class StratifiedMetrics:
    """Container for stratified metric results.

    Attributes
    ----------
    characteristic : str
        The characteristic used for stratification
    metric : str
        The metric being measured
    data : pd.DataFrame
        Dataframe with bucket, mean_delta, ci_low, ci_high, n, win_rate, etc.
    """

    characteristic: str
    metric: str
    data: pd.DataFrame


def _build_scenario_key(df: pd.DataFrame) -> pd.Series:
    """Build a unique key for each conflict scenario."""
    if "id" in df.columns:
        return df["id"].astype(str)
    cols = [c for c in ["repo", "file_name"] if c in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df)), index=df.index).astype(str)
    return df[cols].astype(str).agg("__".join, axis=1)


def _coerce_bool_metric(series: pd.Series) -> pd.Series:
    """Coerce exact_match to numeric 0/1.

    Handles bool, numeric, and string representations.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    # If already numeric, just ensure it's 0/1 float
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)

    # String coercion - handle various representations
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "1.0", "yes", "y", "t"])
        .astype(float)
    )


def aggregate_to_instance_level(
    df: pd.DataFrame,
    config: RQ2Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Aggregate file-level metrics to instance-level metrics.

    For instances with multiple files:
    - Exact match: True only if ALL files have exact match
    - Soft metrics (similarity, bleu3, rouge_l): Averaged across files

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe with file-level metrics
    config : RQ2Config
        Configuration specifying metrics

    Returns
    -------
    pd.DataFrame
        Instance-level aggregated dataframe with one row per (id, eval_method, model_name)
    """
    if "id" not in df.columns:
        raise ValueError("DataFrame must have 'id' column for instance-level aggregation")

    work = df.copy()

    # Coerce exact_match to numeric
    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    # Define grouping columns - handle NaN model_name for baselines
    group_cols = ["id", "eval_method"]
    if "model_name" in work.columns:
        # Fill NaN model_name with a placeholder for grouping
        work["model_name"] = work["model_name"].fillna("__baseline__")
        group_cols.append("model_name")

    # Define aggregation rules
    agg_dict = {}

    # Exact match: all files must match (use min since 1=True, 0=False)
    if "exact_match" in work.columns:
        agg_dict["exact_match"] = "min"  # 1 only if all are 1

    # Soft metrics: average across files
    soft_metrics = ["similarity", "bleu3", "rouge_l"]
    for metric in soft_metrics:
        if metric in work.columns:
            agg_dict[metric] = "mean"

    # Keep other useful columns (take first value)
    other_cols = ["repo", "difficulty", "project_size", "bypass_method"]
    for col in other_cols:
        if col in work.columns:
            agg_dict[col] = "first"

    # Token columns: sum across files in instance
    token_cols = ["tokens_diff_a", "tokens_diff_b", "tokens_in", "tokens_original"]
    for col in token_cols:
        if col in work.columns:
            agg_dict[col] = "sum"

    # Perform aggregation
    result = work.groupby(group_cols, as_index=False).agg(agg_dict)

    # Add file count
    file_counts = work.groupby(group_cols).size().reset_index(name="n_files")
    result = result.merge(file_counts, on=group_cols, how="left")

    # Restore NaN for baseline model_name
    if "model_name" in result.columns:
        result.loc[result["model_name"] == "__baseline__", "model_name"] = np.nan

    return result


def _extract_file_type(file_name: str) -> str:
    """Extract file type category from file name."""
    if pd.isna(file_name):
        return "Other"
    
    ext = Path(str(file_name)).suffix.lower()
    for category, extensions in FILE_TYPE_CATEGORIES.items():
        if ext in extensions:
            return category
    return "Other"


def _assign_bucket(value: float, buckets: list[tuple[str, float, float]]) -> str:
    """Assign a value to a bucket based on ranges."""
    if pd.isna(value):
        return "Unknown"
    for name, low, high in buckets:
        if low <= value <= high:
            return name
    return "Unknown"


def prepare_improvement_data(
    df: pd.DataFrame,
    config: RQ2Config = DEFAULT_CONFIG,
    granularity: GranularityType = "file",
) -> ImprovementData:
    """Prepare paired improvement data with delta calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe with 'eval_method' column
    config : RQ2Config
        Configuration specifying methods to compare
    granularity : {"file", "instance"}
        Level of aggregation:
        - "file": Each file is treated independently (default)
        - "instance": Files grouped by 'id', EM requires all files to match,
          soft metrics averaged

    Returns
    -------
    ImprovementData
        Container with improvement deltas and characteristics
    """
    if "eval_method" not in df.columns:
        raise ValueError("DataFrame must have 'eval_method' column")

    # If instance-level, first aggregate to instance level
    if granularity == "instance":
        if "id" not in df.columns:
            raise ValueError("DataFrame must have 'id' column for instance-level granularity")
        work = aggregate_to_instance_level(df, config)
    else:
        work = df.copy()

    # Filter to relevant methods
    methods = {config.single_agent_method, config.multi_agent_method}
    work = work[work["eval_method"].isin(methods)]

    if work.empty:
        return ImprovementData(
            dataframe=pd.DataFrame(),
            n_pairs=0,
            single_method=config.single_agent_method,
            multi_method=config.multi_agent_method,
        )

    # Build scenario key
    work["_scenario_key"] = _build_scenario_key(work)

    # Coerce exact_match if present
    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    # Derive additional characteristics
    if "file_name" in work.columns:
        work["file_type"] = work["file_name"].apply(_extract_file_type)
    
    # Compute conflict size (sum of diff tokens)
    if "tokens_diff_a" in work.columns and "tokens_diff_b" in work.columns:
        work["conflict_size"] = (
            pd.to_numeric(work["tokens_diff_a"], errors="coerce").fillna(0) +
            pd.to_numeric(work["tokens_diff_b"], errors="coerce").fillna(0)
        )
    
    # Context size
    if "tokens_in" in work.columns:
        work["tokens_context"] = pd.to_numeric(work["tokens_in"], errors="coerce")
    elif "tokens_total" in work.columns:
        work["tokens_context"] = pd.to_numeric(work["tokens_total"], errors="coerce")

    # Pivot to get paired values for each metric
    characteristics = [
        "difficulty", "project_size", "file_type", "conflict_size", 
        "tokens_context", "model_name", "tokens_original", "tokens_diff_a", "tokens_diff_b"
    ]
    characteristics = [c for c in characteristics if c in work.columns]

    # Get first characteristic values per scenario (should be same across methods)
    char_df = (
        work.groupby("_scenario_key")[characteristics]
        .first()
        .reset_index()
    )

    # Pivot metrics
    result_dfs = [char_df]
    
    for metric in config.metrics:
        if metric not in work.columns:
            continue
        
        pivot = work.pivot_table(
            index="_scenario_key",
            columns="eval_method",
            values=metric,
            aggfunc="first",
        ).reset_index()
        
        # Rename columns
        if config.single_agent_method in pivot.columns:
            pivot = pivot.rename(columns={config.single_agent_method: f"single_{metric}"})
        if config.multi_agent_method in pivot.columns:
            pivot = pivot.rename(columns={config.multi_agent_method: f"multi_{metric}"})
        
        result_dfs.append(pivot.drop(columns=["_scenario_key"], errors="ignore"))
    
    # Merge all
    result = char_df.copy()
    for pivot_df in result_dfs[1:]:
        for col in pivot_df.columns:
            if col not in result.columns:
                result[col] = pivot_df[col].values

    # Compute deltas for each metric
    for metric in config.metrics:
        single_col = f"single_{metric}"
        multi_col = f"multi_{metric}"
        if single_col in result.columns and multi_col in result.columns:
            result[f"delta_{metric}"] = (
                pd.to_numeric(result[multi_col], errors="coerce") -
                pd.to_numeric(result[single_col], errors="coerce")
            )
            # Win indicator (multi better)
            result[f"win_{metric}"] = result[f"delta_{metric}"] > 0
            result[f"loss_{metric}"] = result[f"delta_{metric}"] < 0

    # Keep only rows with at least one valid delta
    delta_cols = [c for c in result.columns if c.startswith("delta_")]
    if delta_cols:
        result = result.dropna(subset=delta_cols, how="all")

    return ImprovementData(
        dataframe=result,
        n_pairs=len(result),
        single_method=config.single_agent_method,
        multi_method=config.multi_agent_method,
    )


def create_buckets(
    df: pd.DataFrame,
    config: RQ2Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add bucket columns for numeric characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numeric characteristic columns
    config : RQ2Config
        Configuration with bucket definitions

    Returns
    -------
    pd.DataFrame
        Dataframe with added bucket columns
    """
    result = df.copy()

    # Conflict size buckets
    if "conflict_size" in result.columns:
        result["conflict_size_bucket"] = result["conflict_size"].apply(
            lambda x: _assign_bucket(x, config.conflict_size_buckets)
        )

    # Context size buckets
    if "tokens_context" in result.columns:
        result["context_size_bucket"] = result["tokens_context"].apply(
            lambda x: _assign_bucket(x, config.context_size_buckets)
        )

    # Difficulty is already categorical, but normalize
    if "difficulty" in result.columns:
        result["difficulty"] = result["difficulty"].astype(str).str.lower().str.strip()
        # Map to standard order
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        result["difficulty_order"] = result["difficulty"].map(difficulty_order).fillna(3)

    return result


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(random_state)
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return (np.nan, np.nan)

    boot_means = np.empty(n_boot)
    n = len(clean)
    for i in range(n_boot):
        sample = clean[rng.integers(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci_level) / 2
    return (float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha)))


def compute_stratified_metrics(
    df: pd.DataFrame,
    characteristic: str,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    granularity: GranularityType = "file",
) -> StratifiedMetrics:
    """Compute improvement metrics stratified by a characteristic.

    Parameters
    ----------
    df : pd.DataFrame
        Improvement data (from prepare_improvement_data)
    characteristic : str
        Column to stratify by (e.g., "difficulty", "conflict_size_bucket")
    metric : str
        The metric to analyze
    config : RQ2Config
        Configuration
    granularity : {"file", "instance"}
        Level of aggregation (for labeling purposes; data should already be
        aggregated if instance-level)

    Returns
    -------
    StratifiedMetrics
        Stratified results with CIs
    """
    delta_col = f"delta_{metric}"
    win_col = f"win_{metric}"

    if characteristic not in df.columns or delta_col not in df.columns:
        return StratifiedMetrics(
            characteristic=characteristic,
            metric=metric,
            data=pd.DataFrame(),
        )

    results = []
    for bucket, group in df.groupby(characteristic, dropna=False):
        if len(group) < config.min_bucket_size:
            continue

        deltas = group[delta_col].dropna().to_numpy()
        if len(deltas) == 0:
            continue

        mean_delta = float(np.mean(deltas))
        ci_low, ci_high = _bootstrap_ci(
            deltas,
            n_boot=config.n_bootstrap,
            ci_level=config.ci_level,
            random_state=config.random_state,
        )

        # Win rate
        wins = group[win_col].sum() if win_col in group.columns else np.nan
        n = len(group)
        win_rate = wins / n if n > 0 else np.nan

        results.append({
            "bucket": str(bucket),
            "n": n,
            "mean_delta": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "std_delta": float(np.std(deltas)),
            "median_delta": float(np.median(deltas)),
            "win_rate": win_rate,
            "wins": int(wins) if not np.isnan(wins) else 0,
            "losses": int((group.get(f"loss_{metric}", False)).sum()) if f"loss_{metric}" in group.columns else 0,
        })

    return StratifiedMetrics(
        characteristic=characteristic,
        metric=metric,
        data=pd.DataFrame(results),
    )


def prepare_regression_data(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Prepare data for logistic regression analysis.

    Creates a dataset with binary outcome (multi-agent wins) and features.

    Parameters
    ----------
    df : pd.DataFrame
        Improvement data (from prepare_improvement_data)
    metric : str
        The metric for win/loss determination
    config : RQ2Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Regression-ready dataframe with dummy variables
    """
    win_col = f"win_{metric}"
    if win_col not in df.columns:
        return pd.DataFrame()

    # Select features
    feature_cols = []
    
    # Categorical features
    categorical = ["difficulty", "project_size", "file_type"]
    for col in categorical:
        if col in df.columns:
            feature_cols.append(col)
    
    # Numeric features
    numeric = ["conflict_size", "tokens_context", "tokens_original"]
    for col in numeric:
        if col in df.columns:
            feature_cols.append(col)

    if not feature_cols:
        return pd.DataFrame()

    # Build regression dataset
    result = df[[win_col] + feature_cols].copy()
    result = result.rename(columns={win_col: "win"})
    result["win"] = result["win"].astype(int)

    # Drop rows with missing values
    result = result.dropna()

    return result


def compute_interaction_matrix(
    df: pd.DataFrame,
    row_characteristic: str,
    col_characteristic: str,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute a 2D matrix of improvement by two characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Improvement data
    row_characteristic : str
        Characteristic for rows
    col_characteristic : str
        Characteristic for columns
    metric : str
        The metric to analyze
    config : RQ2Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Pivot table with mean delta values
    """
    delta_col = f"delta_{metric}"

    if (row_characteristic not in df.columns or 
        col_characteristic not in df.columns or 
        delta_col not in df.columns):
        return pd.DataFrame()

    # Create pivot table
    pivot = df.pivot_table(
        index=row_characteristic,
        columns=col_characteristic,
        values=delta_col,
        aggfunc="mean",
    )

    return pivot


def compute_interaction_counts(
    df: pd.DataFrame,
    row_characteristic: str,
    col_characteristic: str,
    config: RQ2Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute count matrix for two characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Improvement data
    row_characteristic : str
        Characteristic for rows
    col_characteristic : str
        Characteristic for columns
    config : RQ2Config
        Configuration

    Returns
    -------
    pd.DataFrame
        Pivot table with counts
    """
    if row_characteristic not in df.columns or col_characteristic not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=row_characteristic,
        columns=col_characteristic,
        values="_scenario_key" if "_scenario_key" in df.columns else df.columns[0],
        aggfunc="count",
    )

    return pivot
