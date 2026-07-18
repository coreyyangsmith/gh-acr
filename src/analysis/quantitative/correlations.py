"""Correlation analysis between quantitative metrics and RQ2/RQ3 data.

Computes:
- Quantitative metrics vs performance (Spearman/Pearson)
- Quantitative metrics vs RQ3 classification labels (Mann-Whitney U)
- Quantitative metrics vs RQ3 code complexity (cross-correlation)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    QuantConfig,
    DEFAULT_CONFIG,
    PERFORMANCE_METRICS,
    SINGLE_AGENT_METHOD,
    MULTI_AGENT_METHOD,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────


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
    return (
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1 - alpha)),
    )


def _coerce_exact_match(series: pd.Series) -> pd.Series:
    """Coerce exact_match to numeric 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "1.0", "yes"])
        .astype(float)
    )


def _prepare_performance_pairs(
    results_df: pd.DataFrame,
    config: QuantConfig,
) -> pd.DataFrame:
    """Create paired (agent vs bypass) performance data from the results CSV.

    Returns a DataFrame with one row per sample, containing:
    - ``id`` (sample ID)
    - ``delta_{metric}`` for each performance metric
    - ``difficulty``, ``project_size``
    """
    if "eval_method" not in results_df.columns:
        return pd.DataFrame()

    df = results_df.copy()

    # Coerce exact_match
    if "exact_match" in df.columns:
        df["exact_match"] = _coerce_exact_match(df["exact_match"])

    # Filter to agent / bypass
    agent_df = df[df["eval_method"] == config.single_agent_method].copy()
    bypass_df = df[df["eval_method"] == config.multi_agent_method].copy()

    if agent_df.empty or bypass_df.empty:
        return pd.DataFrame()

    # Aggregate to instance level (in case of multi-file scenarios)
    id_col = "id"
    if id_col not in df.columns:
        return pd.DataFrame()

    agg_cols = {}
    for metric in config.metrics:
        if metric in df.columns:
            if metric == "exact_match":
                agg_cols[metric] = "min"  # all files must match
            else:
                agg_cols[metric] = "mean"

    for col in ["difficulty", "project_size"]:
        if col in df.columns:
            agg_cols[col] = "first"

    agent_agg = agent_df.groupby(id_col, as_index=False).agg(agg_cols)
    bypass_agg = bypass_df.groupby(id_col, as_index=False).agg(agg_cols)

    # Merge on id
    merged = agent_agg.merge(
        bypass_agg, on=id_col, suffixes=("_agent", "_bypass")
    )

    # Compute deltas
    for metric in config.metrics:
        a_col = f"{metric}_agent"
        b_col = f"{metric}_bypass"
        if a_col in merged.columns and b_col in merged.columns:
            merged[f"delta_{metric}"] = (
                pd.to_numeric(merged[b_col], errors="coerce")
                - pd.to_numeric(merged[a_col], errors="coerce")
            )

    # Keep useful columns
    keep_cols = [id_col]
    keep_cols += [f"delta_{m}" for m in config.metrics if f"delta_{m}" in merged.columns]
    for col in ["difficulty_agent", "project_size_agent"]:
        if col in merged.columns:
            merged[col.replace("_agent", "")] = merged[col]
            keep_cols.append(col.replace("_agent", ""))

    return merged[keep_cols].copy()


# ── 1. Quantitative vs Performance ───────────────────────────────────────


def compute_performance_correlations(
    deltas_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Correlate quantitative change metrics with performance deltas.

    Merges quantitative deltas (from ``quantitative_deltas.csv``) with
    performance deltas (agent vs bypass) from the results CSV, then
    computes Spearman and Pearson correlations.

    Parameters
    ----------
    deltas_df : pd.DataFrame
        Quantitative deltas DataFrame (one row per sample)
    results_df : pd.DataFrame
        Results CSV with ``eval_method``, ``id``, and metric columns
    config : QuantConfig
        Configuration

    Returns
    -------
    pd.DataFrame
        Correlation table with columns:
        quantitative_metric, performance_metric, n_samples,
        spearman_r, spearman_p, pearson_r, pearson_p
    """
    logger.info("Computing quantitative vs performance correlations...")

    if deltas_df.empty or results_df.empty:
        return pd.DataFrame()

    # Prepare performance pairs
    perf_pairs = _prepare_performance_pairs(results_df, config)
    if perf_pairs.empty:
        logger.warning("  No performance pairs computed")
        return pd.DataFrame()

    # Merge on sample_id ↔ id
    # Sample IDs may have a suffix (e.g. "12345-1") that needs stripping
    deltas_df = deltas_df.copy()
    deltas_df["sample_id"] = deltas_df["sample_id"].astype(str)
    deltas_df["_base_id"] = deltas_df["sample_id"].str.replace(r"-\d+$", "", regex=True)
    perf_pairs["id"] = perf_pairs["id"].astype(str)

    merged = deltas_df.merge(perf_pairs, left_on="_base_id", right_on="id", how="inner")
    merged = merged.drop(columns=["_base_id"], errors="ignore")
    logger.info(f"  Merged {len(merged)} samples for correlation")

    if len(merged) < config.min_samples:
        logger.warning(f"  Too few samples ({len(merged)}) for correlation")
        return pd.DataFrame()

    # Quantitative metrics to correlate
    quant_cols = [c for c in deltas_df.columns if c not in ("sample_id", "source")]
    # Filter to numeric
    quant_cols = [
        c for c in quant_cols
        if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c])
    ]

    perf_cols = [f"delta_{m}" for m in config.metrics if f"delta_{m}" in merged.columns]

    rows = []
    for qcol in quant_cols:
        for pcol in perf_cols:
            valid = merged[[qcol, pcol]].dropna()
            if len(valid) < config.min_samples:
                continue

            try:
                sp_r, sp_p = stats.spearmanr(valid[qcol], valid[pcol])
                pe_r, pe_p = stats.pearsonr(valid[qcol], valid[pcol])

                rows.append({
                    "quantitative_metric": qcol,
                    "performance_metric": pcol,
                    "n_samples": len(valid),
                    "spearman_r": sp_r,
                    "spearman_p": sp_p,
                    "pearson_r": pe_r,
                    "pearson_p": pe_p,
                })
            except Exception:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_spearman"] = result["spearman_r"].abs()
        result = result.sort_values("abs_spearman", ascending=False)
        result = result.drop(columns=["abs_spearman"])

    logger.info(f"  Computed {len(result)} correlation pairs")
    return result


# ── 2. Quantitative vs Labels ────────────────────────────────────────────


def compute_label_correlations(
    deltas_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Correlate quantitative metrics with RQ3 classification labels.

    For each label, compares the distribution of each quantitative metric
    between samples *with* vs *without* the label using Mann-Whitney U.

    Parameters
    ----------
    deltas_df : pd.DataFrame
        Quantitative deltas DataFrame (one row per sample)
    paired_df : pd.DataFrame
        RQ3 paired data with binary label columns and ``id`` column
    config : QuantConfig
        Configuration

    Returns
    -------
    pd.DataFrame
        Columns: label, quantitative_metric, mean_with, mean_without,
        diff_means, mann_whitney_u, mann_whitney_p, n_with, n_without
    """
    logger.info("Computing quantitative vs label correlations...")

    if deltas_df.empty or paired_df.empty:
        return pd.DataFrame()

    # Merge on sample_id / id (strip suffix like -1, -2)
    deltas_df = deltas_df.copy()
    paired_df = paired_df.copy()
    deltas_df["sample_id"] = deltas_df["sample_id"].astype(str)
    deltas_df["_base_id"] = deltas_df["sample_id"].str.replace(r"-\d+$", "", regex=True)

    if "id" in paired_df.columns:
        paired_df["id"] = paired_df["id"].astype(str)
        merged = deltas_df.merge(
            paired_df, left_on="_base_id", right_on="id", how="inner"
        )
    elif "sample_id" in paired_df.columns:
        paired_df["sample_id"] = paired_df["sample_id"].astype(str)
        paired_df["_base_id"] = paired_df["sample_id"].str.replace(r"-\d+$", "", regex=True)
        merged = deltas_df.merge(paired_df, on="_base_id", how="inner", suffixes=("", "_rq3"))
    else:
        logger.warning("  No join key found in paired_df")
        return pd.DataFrame()

    merged = merged.drop(columns=["_base_id"], errors="ignore")
    logger.info(f"  Merged {len(merged)} samples for label analysis")

    if len(merged) < config.min_samples:
        return pd.DataFrame()

    # Identify label columns (binary 0/1)
    candidate_labels = [
        c for c in paired_df.columns
        if c not in ("id", "sample_id", "difficulty", "project_size", "source_file")
        and not c.startswith("agent_")
        and not c.startswith("bypass_")
        and not c.startswith("delta_")
        and not c.endswith("_wins_exact_match")
        and not c.endswith("_wins_similarity")
        and not c.endswith("_wins_bleu3")
        and not c.endswith("_wins_rouge_l")
    ]

    # Keep only binary-like columns
    label_cols = []
    for col in candidate_labels:
        if col in merged.columns:
            unique_vals = merged[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                label_cols.append(col)

    # Quantitative metrics columns
    quant_cols = [
        c for c in deltas_df.columns
        if c not in ("sample_id", "source")
        and c in merged.columns
        and pd.api.types.is_numeric_dtype(merged[c])
    ]

    rows = []
    for label in label_cols:
        with_label = merged[merged[label] == 1]
        without_label = merged[merged[label] == 0]

        n_with = len(with_label)
        n_without = len(without_label)

        if n_with < 3 or n_without < 3:
            continue

        for qcol in quant_cols:
            vals_with = with_label[qcol].dropna()
            vals_without = without_label[qcol].dropna()

            if len(vals_with) < 3 or len(vals_without) < 3:
                continue

            mean_with = vals_with.mean()
            mean_without = vals_without.mean()
            diff_means = mean_with - mean_without

            try:
                u_stat, u_p = stats.mannwhitneyu(
                    vals_with, vals_without, alternative="two-sided"
                )
            except Exception:
                u_stat, u_p = np.nan, np.nan

            rows.append({
                "label": label,
                "quantitative_metric": qcol,
                "mean_with_label": mean_with,
                "mean_without_label": mean_without,
                "diff_means": diff_means,
                "mann_whitney_u": u_stat,
                "mann_whitney_p": u_p,
                "n_with": n_with,
                "n_without": n_without,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mann_whitney_p")

    logger.info(f"  Computed {len(result)} label-metric pairs")
    return result


# ── 3. Quantitative vs Complexity (cross-analysis) ───────────────────────


def compute_complexity_cross_correlations(
    deltas_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Cross-correlate quantitative change metrics with code complexity metrics.

    Examines whether higher diff_total_change correlates with higher
    cyclomatic complexity, maintainability index, etc.

    Parameters
    ----------
    deltas_df : pd.DataFrame
        Quantitative deltas DataFrame (one row per sample)
    complexity_df : pd.DataFrame
        RQ3 complexity_metrics.csv with ``sample_id``, ``method``,
        and complexity columns (sloc, cc_avg, mi_score, etc.)
    config : QuantConfig
        Configuration

    Returns
    -------
    pd.DataFrame
        Correlation table with quantitative_metric, complexity_metric,
        method, spearman_r, spearman_p, pearson_r, pearson_p
    """
    logger.info("Computing quantitative vs complexity cross-correlations...")

    if deltas_df.empty or complexity_df.empty:
        return pd.DataFrame()

    # Ensure string IDs (strip suffix for matching)
    deltas_df = deltas_df.copy()
    complexity_df = complexity_df.copy()
    deltas_df["sample_id"] = deltas_df["sample_id"].astype(str)
    deltas_df["_base_id"] = deltas_df["sample_id"].str.replace(r"-\d+$", "", regex=True)
    complexity_df["sample_id"] = complexity_df["sample_id"].astype(str)
    complexity_df["_base_id"] = complexity_df["sample_id"].str.replace(r"-\d+$", "", regex=True)

    # Complexity metrics of interest
    complexity_cols = [
        "sloc", "lloc", "cc_total", "cc_avg", "cc_max",
        "mi_score", "h_difficulty", "h_bugs",
    ]
    complexity_cols = [c for c in complexity_cols if c in complexity_df.columns]

    # Quantitative metrics
    quant_cols = [
        c for c in deltas_df.columns
        if c not in ("sample_id", "source")
        and pd.api.types.is_numeric_dtype(deltas_df[c])
    ]

    rows = []
    for method in ["ground_truth", "agent", "bypass"]:
        method_complexity = complexity_df[complexity_df["method"] == method]
        if method_complexity.empty:
            continue

        # Merge on base_id
        merged = deltas_df.merge(
            method_complexity[["_base_id"] + complexity_cols],
            on="_base_id",
            how="inner",
        )

        if len(merged) < config.min_samples:
            continue

        for qcol in quant_cols:
            for ccol in complexity_cols:
                valid = merged[[qcol, ccol]].dropna()
                if len(valid) < config.min_samples:
                    continue

                try:
                    sp_r, sp_p = stats.spearmanr(valid[qcol], valid[ccol])
                    pe_r, pe_p = stats.pearsonr(valid[qcol], valid[ccol])

                    rows.append({
                        "quantitative_metric": qcol,
                        "complexity_metric": ccol,
                        "method": method,
                        "n_samples": len(valid),
                        "spearman_r": sp_r,
                        "spearman_p": sp_p,
                        "pearson_r": pe_r,
                        "pearson_p": pe_p,
                    })
                except Exception:
                    pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_spearman"] = result["spearman_r"].abs()
        result = result.sort_values("abs_spearman", ascending=False)
        result = result.drop(columns=["abs_spearman"])

    logger.info(f"  Computed {len(result)} cross-correlation pairs")
    return result
