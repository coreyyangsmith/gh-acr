"""Data preparation utilities for RQ1 comparisons.

Handles pairing single-agent and multi-agent results for the same scenarios,
computing per-model metrics, and preparing data for visualizations.

Supports two granularity levels:
- Per-file (overall): Each row in the CSV is treated independently
- Per-instance: Files are grouped by 'id', with EM requiring all files to match
  and soft metrics averaged across files
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from ..stats import (
    paired_bootstrap_mean_ci,
    p_value_binomial_sign_test_two_sided,
    p_value_wilcoxon_signed_rank,
)
from .config import RQ1Config, DEFAULT_CONFIG

# Type alias for granularity
GranularityType = Literal["file", "instance"]


def common_agent_bypass_ids(
    df: pd.DataFrame,
    *,
    single_method: str = "agent",
    multi_method: str = "bypass7",
) -> set[str]:
    """Scenario IDs present for both single and multi methods for every ``model_name``.

    Matches the common-set definition used in final paper figures.
    ``multi_method`` defaults to ``bypass7`` but should be ``better_judge``
    (or another multi-agent id) when that is what the results CSV contains.
    """
    if "id" not in df.columns or "eval_method" not in df.columns or "model_name" not in df.columns:
        return set()
    ab = df[df["eval_method"].isin([single_method, multi_method])]
    # Only count IDs that have both methods within each model
    models = ab["model_name"].dropna().unique()
    if len(models) == 0:
        return set()
    per_model: dict[object, set[str]] = {}
    for m in models:
        model_df = ab[ab["model_name"] == m]
        single_ids = set(
            model_df[model_df["eval_method"] == single_method]["id"].astype(str).unique()
        )
        multi_ids = set(
            model_df[model_df["eval_method"] == multi_method]["id"].astype(str).unique()
        )
        per_model[m] = single_ids & multi_ids
    if not per_model:
        return set()
    return set.intersection(*per_model.values())


@dataclass
class PairedData:
    """Container for paired single-agent vs multi-agent results.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Wide-format dataframe with columns for each method's metrics
    n_pairs : int
        Number of paired scenarios
    single_method : str
        Single-agent method identifier
    multi_method : str
        Multi-agent method identifier
    model_name : str | None
        Model name if filtered by model
    """

    dataframe: pd.DataFrame
    n_pairs: int
    single_method: str
    multi_method: str
    model_name: Optional[str] = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model.

    Attributes
    ----------
    model_name : str
        Name of the coding model
    n_scenarios : int
        Number of scenarios evaluated
    single_agent : dict[str, float]
        Metric values for single-agent
    multi_agent : dict[str, float]
        Metric values for multi-agent
    single_agent_ci : dict[str, tuple[float, float]]
        Confidence intervals for single-agent metrics
    multi_agent_ci : dict[str, tuple[float, float]]
        Confidence intervals for multi-agent metrics
    """

    model_name: str
    n_scenarios: int
    single_agent: dict[str, float]
    multi_agent: dict[str, float]
    single_agent_ci: dict[str, tuple[float, float]]
    multi_agent_ci: dict[str, tuple[float, float]]


@dataclass
class PairedDeltaStats:
    """Paired single-agent vs multi-agent delta statistics for one model and metric."""

    model_name: str
    metric: str
    granularity: GranularityType
    n_pairs: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    wins: int
    ties: int
    losses: int
    n_discordant: int


def _build_scenario_key(df: pd.DataFrame) -> pd.Series:
    """Build a unique key for each conflict scenario.

    Uses 'id' if present, otherwise combines 'repo' and 'file_name'.
    """
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
    config: RQ1Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Aggregate file-level metrics to instance-level metrics.

    For instances with multiple files:
    - Exact match: True only if ALL files have exact match
    - Soft metrics (similarity, bleu3, rouge_l): Averaged across files

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe with file-level metrics
    config : RQ1Config
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
    other_cols = ["repo", "bypass_method", "difficulty", "project_size"]
    for col in other_cols:
        if col in work.columns:
            agg_dict[col] = "first"

    # Count files per instance
    agg_dict["_n_files"] = ("exact_match" if "exact_match" in work.columns else list(agg_dict.keys())[0], "count")

    # Perform aggregation
    # Need to handle the count specially
    count_col = "exact_match" if "exact_match" in work.columns else list(agg_dict.keys())[0]

    # Build aggregation dict properly
    final_agg = {}
    for col, func in agg_dict.items():
        if col == "_n_files":
            continue
        final_agg[col] = func

    result = work.groupby(group_cols, as_index=False).agg(final_agg)

    # Add file count
    file_counts = work.groupby(group_cols).size().reset_index(name="n_files")
    result = result.merge(file_counts, on=group_cols, how="left")

    # Restore NaN for baseline model_name
    if "model_name" in result.columns:
        result.loc[result["model_name"] == "__baseline__", "model_name"] = np.nan

    return result


def prepare_paired_data(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    model_name: Optional[str] = None,
) -> PairedData:
    """Prepare paired data for single-agent vs multi-agent comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe with 'eval_method' column
    config : RQ1Config
        Configuration specifying methods to compare
    model_name : str, optional
        Filter to a specific model

    Returns
    -------
    PairedData
        Container with paired results in wide format
    """
    if "eval_method" not in df.columns:
        raise ValueError("DataFrame must have 'eval_method' column")

    work = df.copy()

    # Filter by model if specified
    if model_name is not None and "model_name" in work.columns:
        work = work[work["model_name"] == model_name]

    # Filter to relevant methods
    methods = {config.single_agent_method, config.multi_agent_method}
    work = work[work["eval_method"].isin(methods)]

    if work.empty:
        return PairedData(
            dataframe=pd.DataFrame(),
            n_pairs=0,
            single_method=config.single_agent_method,
            multi_method=config.multi_agent_method,
            model_name=model_name,
        )

    # Build scenario key
    work["_scenario_key"] = _build_scenario_key(work)

    # Coerce exact_match if present
    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    # Pivot to wide format
    value_cols = [m for m in config.metrics if m in work.columns]
    if not value_cols:
        return PairedData(
            dataframe=pd.DataFrame(),
            n_pairs=0,
            single_method=config.single_agent_method,
            multi_method=config.multi_agent_method,
            model_name=model_name,
        )

    # Create wide format with method-prefixed columns
    pivot_frames = []
    for metric in value_cols:
        pivot = work.pivot_table(
            index="_scenario_key",
            columns="eval_method",
            values=metric,
            aggfunc="first",
        )
        # Rename columns to include metric
        pivot.columns = [f"{col}_{metric}" for col in pivot.columns]
        pivot_frames.append(pivot)

    if not pivot_frames:
        return PairedData(
            dataframe=pd.DataFrame(),
            n_pairs=0,
            single_method=config.single_agent_method,
            multi_method=config.multi_agent_method,
            model_name=model_name,
        )

    wide = pd.concat(pivot_frames, axis=1).reset_index()

    # Keep only rows with both methods present for at least one metric
    single_cols = [f"{config.single_agent_method}_{m}" for m in value_cols]
    multi_cols = [f"{config.multi_agent_method}_{m}" for m in value_cols]
    single_cols = [c for c in single_cols if c in wide.columns]
    multi_cols = [c for c in multi_cols if c in wide.columns]

    # One (or both) methods missing entirely from this slice → no pairs
    if not single_cols or not multi_cols:
        return PairedData(
            dataframe=pd.DataFrame(),
            n_pairs=0,
            single_method=config.single_agent_method,
            multi_method=config.multi_agent_method,
            model_name=model_name,
        )

    # Check if both methods have at least one non-null value
    has_single = wide[single_cols].notna().any(axis=1)
    has_multi = wide[multi_cols].notna().any(axis=1)
    paired = wide[has_single & has_multi].copy()

    return PairedData(
        dataframe=paired,
        n_pairs=len(paired),
        single_method=config.single_agent_method,
        multi_method=config.multi_agent_method,
        model_name=model_name,
    )


def compute_paired_delta_statistics(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    granularity: GranularityType = "instance",
    *,
    tolerance: float = 1e-6,
) -> list[PairedDeltaStats]:
    """Paired (multi - single) delta per instance/file with bootstrap CI and significance.

    - Mean delta and 95% bootstrap CI resample paired rows (same granularity as RQ1).
    - ``exact_match``: two-sided exact binomial test on discordant pairs (McNemar-style).
    - Continuous metrics: two-sided Wilcoxon signed-rank on paired deltas.

    Parameters
    ----------
    df : pd.DataFrame
        Results with ``eval_method``, metrics, and ``model_name`` (unless unknown).
    config : RQ1Config
        Single vs multi method ids and bootstrap settings.
    granularity : {"file", "instance"}
        Match RQ1 aggregation: instance uses ``aggregate_to_instance_level``.
    tolerance : float
        Win/tie/loss tie band for continuous metrics.
    """
    if "eval_method" not in df.columns:
        raise ValueError("DataFrame must have 'eval_method' column")

    if granularity == "instance":
        if "id" not in df.columns:
            raise ValueError("DataFrame must have 'id' column for instance-level granularity")
        work = aggregate_to_instance_level(df, config)
    else:
        work = df.copy()

    methods = {config.single_agent_method, config.multi_agent_method}
    work = work[work["eval_method"].isin(methods)]

    if work.empty:
        return []

    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    if "model_name" in work.columns:
        models = work["model_name"].dropna().unique().tolist()
    else:
        models = ["unknown"]
        work["model_name"] = "unknown"

    out: list[PairedDeltaStats] = []

    for model in sorted(models):
        model_df = work[work["model_name"] == model]
        paired = prepare_paired_data(model_df, config, model_name=model)
        pwide = paired.dataframe
        if pwide.empty:
            continue

        for metric in config.metrics:
            sc = f"{config.single_agent_method}_{metric}"
            mc = f"{config.multi_agent_method}_{metric}"
            if sc not in pwide.columns or mc not in pwide.columns:
                continue

            sub = pwide[[sc, mc]].dropna()
            if sub.empty:
                continue

            single_vals = pd.to_numeric(sub[sc], errors="coerce").to_numpy(dtype=float)
            multi_vals = pd.to_numeric(sub[mc], errors="coerce").to_numpy(dtype=float)
            valid = ~(np.isnan(single_vals) | np.isnan(multi_vals))
            single_vals = single_vals[valid]
            multi_vals = multi_vals[valid]
            if single_vals.size == 0:
                continue

            delta = multi_vals - single_vals
            mean_delta = float(np.mean(delta))

            ci_low, ci_high = paired_bootstrap_mean_ci(
                delta,
                n_boot=config.n_bootstrap,
                ci=config.ci_level,
                random_state=config.random_state,
            )

            tol = 1e-9 if metric == "exact_match" else tolerance
            wins = int(np.sum(delta > tol))
            losses = int(np.sum(delta < -tol))
            ties = int(np.sum(np.abs(delta) <= tol))
            n_pairs = int(wins + ties + losses)
            n_discordant = int(wins + losses)

            if metric == "exact_match":
                p_val = p_value_binomial_sign_test_two_sided(wins, losses)
                test_name = "binomial_discordant"
            else:
                p_val = p_value_wilcoxon_signed_rank(delta)
                test_name = "wilcoxon_signed_rank"

            out.append(
                PairedDeltaStats(
                    model_name=str(model),
                    metric=metric,
                    granularity=granularity,
                    n_pairs=n_pairs,
                    mean_delta=mean_delta,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_val,
                    test=test_name,
                    wins=wins,
                    ties=ties,
                    losses=losses,
                    n_discordant=n_discordant,
                )
            )

    return out


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


def compute_model_metrics(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    granularity: GranularityType = "file",
) -> list[ModelMetrics]:
    """Compute aggregated metrics for each model.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration specifying methods and metrics
    granularity : {"file", "instance"}
        Level of aggregation:
        - "file": Each file is treated independently (default, original behavior)
        - "instance": Files grouped by 'id', EM requires all files to match,
          soft metrics averaged

    Returns
    -------
    list[ModelMetrics]
        List of per-model metric summaries
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
        return []

    # Coerce exact_match
    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    # Get unique models
    if "model_name" in work.columns:
        models = work["model_name"].dropna().unique().tolist()
    else:
        models = ["unknown"]
        work["model_name"] = "unknown"

    results: list[ModelMetrics] = []

    for model in sorted(models):
        model_df = work[work["model_name"] == model]

        # Get paired data for counting scenarios
        paired = prepare_paired_data(model_df, config, model_name=model)
        n_scenarios = paired.n_pairs

        single_metrics: dict[str, float] = {}
        multi_metrics: dict[str, float] = {}
        single_ci: dict[str, tuple[float, float]] = {}
        multi_ci: dict[str, tuple[float, float]] = {}

        for metric in config.metrics:
            if metric not in model_df.columns:
                continue

            single_data = model_df[model_df["eval_method"] == config.single_agent_method][metric]
            multi_data = model_df[model_df["eval_method"] == config.multi_agent_method][metric]

            # Convert to numeric
            single_vals = pd.to_numeric(single_data, errors="coerce").dropna().to_numpy()
            multi_vals = pd.to_numeric(multi_data, errors="coerce").dropna().to_numpy()

            # Compute means
            single_metrics[metric] = float(np.mean(single_vals)) if len(single_vals) > 0 else np.nan
            multi_metrics[metric] = float(np.mean(multi_vals)) if len(multi_vals) > 0 else np.nan

            # Compute confidence intervals
            single_ci[metric] = _bootstrap_ci(
                single_vals,
                n_boot=config.n_bootstrap,
                ci_level=config.ci_level,
                random_state=config.random_state,
            )
            multi_ci[metric] = _bootstrap_ci(
                multi_vals,
                n_boot=config.n_bootstrap,
                ci_level=config.ci_level,
                random_state=config.random_state,
            )

        results.append(
            ModelMetrics(
                model_name=str(model),
                n_scenarios=n_scenarios,
                single_agent=single_metrics,
                multi_agent=multi_metrics,
                single_agent_ci=single_ci,
                multi_agent_ci=multi_ci,
            )
        )

    return results


def compute_win_tie_loss(
    df: pd.DataFrame,
    metric: str,
    config: RQ1Config = DEFAULT_CONFIG,
    tolerance: float = 1e-6,
    granularity: GranularityType = "file",
) -> pd.DataFrame:
    """Compute win/tie/loss counts per model for a metric.

    A "win" means multi-agent outperformed single-agent.
    A "tie" means they performed equally (within tolerance).
    A "loss" means multi-agent underperformed.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric to compare (e.g., "exact_match", "similarity")
    config : RQ1Config
        Configuration
    tolerance : float
        Tolerance for tie detection
    granularity : {"file", "instance"}
        Level of aggregation:
        - "file": Each file is treated independently (default, original behavior)
        - "instance": Files grouped by 'id', EM requires all files to match,
          soft metrics averaged

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: model_name, wins, ties, losses, total,
        win_pct, tie_pct, loss_pct
    """
    if "eval_method" not in df.columns or metric not in df.columns:
        return pd.DataFrame()

    # If instance-level, first aggregate to instance level
    if granularity == "instance":
        if "id" not in df.columns:
            return pd.DataFrame()
        work = aggregate_to_instance_level(df, config)
    else:
        work = df.copy()

    # Coerce exact_match
    if metric == "exact_match":
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    # Filter to relevant methods
    methods = {config.single_agent_method, config.multi_agent_method}
    work = work[work["eval_method"].isin(methods)]

    if work.empty:
        return pd.DataFrame()

    work["_scenario_key"] = _build_scenario_key(work)

    # Get unique models
    if "model_name" in work.columns:
        models = work["model_name"].dropna().unique().tolist()
    else:
        models = ["unknown"]
        work["model_name"] = "unknown"

    results: list[dict] = []

    for model in sorted(models):
        model_df = work[work["model_name"] == model]

        # Pivot to get paired values
        pivot = model_df.pivot_table(
            index="_scenario_key",
            columns="eval_method",
            values=metric,
            aggfunc="first",
        )

        if config.single_agent_method not in pivot.columns or config.multi_agent_method not in pivot.columns:
            continue

        # Drop rows with missing values
        paired = pivot.dropna(subset=[config.single_agent_method, config.multi_agent_method])

        if paired.empty:
            continue

        single_vals = paired[config.single_agent_method].to_numpy()
        multi_vals = paired[config.multi_agent_method].to_numpy()

        # Compute wins/ties/losses
        diff = multi_vals - single_vals
        wins = int(np.sum(diff > tolerance))
        losses = int(np.sum(diff < -tolerance))
        ties = int(np.sum(np.abs(diff) <= tolerance))
        total = wins + ties + losses

        results.append({
            "model_name": model,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "total": total,
            "win_pct": 100.0 * wins / total if total > 0 else 0.0,
            "tie_pct": 100.0 * ties / total if total > 0 else 0.0,
            "loss_pct": 100.0 * losses / total if total > 0 else 0.0,
        })

    return pd.DataFrame(results)


@dataclass
class AllMethodsMetrics:
    """Aggregated metrics for all methods for a single model.

    Attributes
    ----------
    model_name : str
        Name of the coding model
    n_scenarios : int
        Number of scenarios per method
    methods : dict[str, dict[str, float]]
        {method: {metric: value}}
    methods_ci : dict[str, dict[str, tuple[float, float]]]
        {method: {metric: (ci_low, ci_high)}}
    """

    model_name: str
    n_scenarios: dict[str, int]
    methods: dict[str, dict[str, float]]
    methods_ci: dict[str, dict[str, tuple[float, float]]]


def compute_all_methods_metrics(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    granularity: GranularityType = "file",
) -> list[AllMethodsMetrics]:
    """Compute aggregated metrics for all methods (baselines + single + multi) for each model.

    Baselines (base_a, base_b) are model-agnostic and don't have a model_name in the data.
    They are treated as their own separate "Baselines" model category.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration specifying methods and metrics
    granularity : {"file", "instance"}
        Level of aggregation:
        - "file": Each file is treated independently (default, original behavior)
        - "instance": Files grouped by 'id', EM requires all files to match,
          soft metrics averaged

    Returns
    -------
    list[AllMethodsMetrics]
        List of per-model metric summaries for all methods.
        First entry is "Baselines" with base_a/base_b metrics,
        followed by each LLM model with agent/bypass7 metrics.
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

    # Get all methods to include
    all_methods = config.get_all_methods()

    # Filter to relevant methods
    work = work[work["eval_method"].isin(all_methods)]

    if work.empty:
        return []

    # Coerce exact_match
    if "exact_match" in work.columns:
        work["exact_match"] = _coerce_bool_metric(work["exact_match"])

    results: list[AllMethodsMetrics] = []

    # First, create a "Baselines" model entry for base_a and base_b
    if config.include_baselines and config.baseline_methods:
        baseline_method_metrics: dict[str, dict[str, float]] = {}
        baseline_method_ci: dict[str, dict[str, tuple[float, float]]] = {}
        baseline_n_scenarios: dict[str, int] = {}

        for baseline_method in config.baseline_methods:
            baseline_data = work[work["eval_method"] == baseline_method]
            baseline_n_scenarios[baseline_method] = len(baseline_data)

            if baseline_data.empty:
                baseline_method_metrics[baseline_method] = {}
                baseline_method_ci[baseline_method] = {}
                continue

            metrics_vals: dict[str, float] = {}
            ci_vals: dict[str, tuple[float, float]] = {}

            for metric in config.metrics:
                if metric not in baseline_data.columns:
                    continue

                vals = pd.to_numeric(baseline_data[metric], errors="coerce").dropna().to_numpy()

                if len(vals) > 0:
                    metrics_vals[metric] = float(np.mean(vals))
                    ci_vals[metric] = _bootstrap_ci(
                        vals,
                        n_boot=config.n_bootstrap,
                        ci_level=config.ci_level,
                        random_state=config.random_state,
                    )
                else:
                    metrics_vals[metric] = np.nan
                    ci_vals[metric] = (np.nan, np.nan)

            baseline_method_metrics[baseline_method] = metrics_vals
            baseline_method_ci[baseline_method] = ci_vals

        # Add baselines as their own "model" category
        results.append(
            AllMethodsMetrics(
                model_name="Baselines",
                n_scenarios=baseline_n_scenarios,
                methods=baseline_method_metrics,
                methods_ci=baseline_method_ci,
            )
        )

    # Get unique models from non-baseline methods (agent, bypass7 have model_name)
    non_baseline_methods = [m for m in all_methods if m not in config.baseline_methods]
    model_data = work[work["eval_method"].isin(non_baseline_methods)]

    if "model_name" in model_data.columns:
        models = model_data["model_name"].dropna().unique().tolist()
    else:
        models = ["unknown"]

    if not models:
        models = ["unknown"]

    # Then create entries for each LLM model with their agent/bypass7 methods
    for model in sorted(models):
        model_df = work[work["model_name"] == model]

        method_metrics: dict[str, dict[str, float]] = {}
        method_ci: dict[str, dict[str, tuple[float, float]]] = {}
        n_scenarios: dict[str, int] = {}

        for method in non_baseline_methods:
            method_data = model_df[model_df["eval_method"] == method]
            n_scenarios[method] = len(method_data)

            if method_data.empty:
                method_metrics[method] = {}
                method_ci[method] = {}
                continue

            metrics_vals: dict[str, float] = {}
            ci_vals: dict[str, tuple[float, float]] = {}

            for metric in config.metrics:
                if metric not in method_data.columns:
                    continue

                vals = pd.to_numeric(method_data[metric], errors="coerce").dropna().to_numpy()

                if len(vals) > 0:
                    metrics_vals[metric] = float(np.mean(vals))
                    ci_vals[metric] = _bootstrap_ci(
                        vals,
                        n_boot=config.n_bootstrap,
                        ci_level=config.ci_level,
                        random_state=config.random_state,
                    )
                else:
                    metrics_vals[metric] = np.nan
                    ci_vals[metric] = (np.nan, np.nan)

            method_metrics[method] = metrics_vals
            method_ci[method] = ci_vals

        results.append(
            AllMethodsMetrics(
                model_name=str(model),
                n_scenarios=n_scenarios,
                methods=method_metrics,
                methods_ci=method_ci,
            )
        )

    return results
