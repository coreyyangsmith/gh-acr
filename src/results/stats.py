"""Statistical helpers for result analyses.

This module groups small, dependency-light utilities used across result
summaries and comparisons: bootstrap confidence intervals, exact match rate,
cost/time per success, nonparametric tests, and simple correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(series: pd.Series, *, n_boot: int = 2000, ci: float = 0.95, statistic: Callable[[np.ndarray], float] | None = None, random_state: int = 42) -> tuple[float, float]:
    """Return a nonparametric bootstrap confidence interval for a statistic.

    - Defaults to the mean if `statistic` is not provided.
    - Returns a tuple `(low, high)` for the given `ci` level.
    """
    rng = np.random.default_rng(random_state)
    clean = series.dropna().to_numpy()
    if clean.size == 0:
        return (np.nan, np.nan)
    if statistic is None:
        statistic = np.mean
    boot_stats = np.empty(n_boot, dtype=float)
    n = clean.size
    for i in range(n_boot):
        sample = clean[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    alpha = (1 - ci) / 2
    return (float(np.quantile(boot_stats, alpha)), float(np.quantile(boot_stats, 1 - alpha)))


def exact_match_rate(series_bool: pd.Series) -> float:
    """Compute the mean of a boolean series as a float (exact match rate)."""
    return float(series_bool.astype(bool).mean())


def cost_per_success(total_cost: pd.Series, exact_match: pd.Series) -> float:
    """Total cost divided by the number of successes; NaN if zero successes."""
    successes = exact_match.astype(bool).sum()
    return float(total_cost.sum() / successes) if successes > 0 else float("nan")


def time_per_success(total_time: pd.Series, exact_match: pd.Series) -> float:
    """Total time divided by the number of successes; NaN if zero successes."""
    successes = exact_match.astype(bool).sum()
    return float(total_time.sum() / successes) if successes > 0 else float("nan")


def paired_wilcoxon(delta: pd.Series) -> tuple[float, float]:
    """Return (statistic, p_value) for Wilcoxon signed-rank on paired deltas.

    Requires SciPy; if unavailable, returns `(nan, nan)`.
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return (float("nan"), float("nan"))
    arr = delta.dropna().to_numpy()
    if arr.size < 1:
        return (float("nan"), float("nan"))
    stat, p = wilcoxon(arr, zero_method="pratt", alternative="two-sided", correction=False, mode="approx")
    return (float(stat), float(p))


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    """Compute Cliff's delta effect size for paired samples `x` and `y`.

    Positive values mean `x` tends to be larger than `y`. Range approximately
    in [-1, 1]. Uses all pairwise comparisons.
    """
    x_clean = x.dropna().to_numpy()
    y_clean = y.dropna().to_numpy()
    if x_clean.size == 0 or y_clean.size == 0:
        return float("nan")
    # Pairwise comparisons; for moderate sizes this is acceptable.
    greater = 0
    lesser = 0
    ties = 0
    for xi in x_clean:
        cmp = (xi > y_clean) - (xi < y_clean)
        greater += int((cmp == 1).sum())
        lesser += int((cmp == -1).sum())
        ties += int((cmp == 0).sum())
    n_pairs = greater + lesser + ties
    return (greater - lesser) / n_pairs if n_pairs > 0 else float("nan")


def correlation_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a rounded correlation table for the selected columns."""
    return df[cols].corr().round(3)

