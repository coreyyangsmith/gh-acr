from __future__ import annotations

"""Diagnostics: head-to-head plots, Bland–Altman, paired tests, and helpers."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .stats import paired_wilcoxon, cliffs_delta, correlation_table


def head_to_head_paired_plot(
    df: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    metric: str = "similarity",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> pd.DataFrame:
    """Paired dots per file comparing two methods on a metric.

    Returns the underlying paired data used for plotting.
    """
    if "eval_method" not in df.columns:
        raise ValueError("missing eval_method")
    if metric not in df.columns:
        raise ValueError(f"missing metric column {metric}")

    key = _build_key(df)
    tmp = df.assign(_key=key)
    subset = tmp[tmp["eval_method"].isin([method_a, method_b])]
    pivot = subset.pivot_table(index="_key", columns="eval_method", values=metric, aggfunc="first")
    paired = pivot.dropna(subset=[method_a, method_b]).reset_index()

    # Plot paired lines
    plt.figure(figsize=(10, 7))
    x_positions = [0, 1]
    for _, row in paired.iterrows():
        plt.plot(x_positions, [row[method_a], row[method_b]], color="gray", alpha=0.4)
        plt.scatter(x_positions, [row[method_a], row[method_b]], color=["tab:blue", "tab:orange"], s=20)
    plt.xticks(x_positions, [method_a, method_b])
    plt.ylabel(metric)
    plt.title(f"Head-to-Head: {method_a} vs {method_b} on {metric}")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    return paired


def bland_altman(
    df: pd.DataFrame,
    *,
    metric_x: str = "similarity",
    metric_y: str = "bleu3",
    facet_by: str = "eval_method",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Bland–Altman plot to assess agreement between two metrics; faceted by method."""
    if not {metric_x, metric_y}.issubset(df.columns):
        return
    work = df[[metric_x, metric_y, facet_by]].dropna().copy()
    work["mean"] = (work[metric_x] + work[metric_y]) / 2.0
    work["diff"] = work[metric_x] - work[metric_y]

    g = sns.FacetGrid(work, col=facet_by, col_wrap=3, sharex=False, sharey=True, height=3.2)
    g.map_dataframe(sns.scatterplot, x="mean", y="diff", alpha=0.6)
    g.set_axis_labels("mean", f"{metric_x} - {metric_y}")
    g.fig.suptitle("Bland–Altman by Method", y=1.02)
    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def compute_paired_tests(
    df: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    metrics: Sequence[str] = ("similarity", "total_cost"),
) -> dict[str, dict[str, float]]:
    """Paired Wilcoxon and Cliff's delta for method deltas per metric."""
    key = _build_key(df)
    tmp = df.assign(_key=key)
    subset = tmp[tmp["eval_method"].isin([method_a, method_b])]
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        pivot = subset.pivot_table(index="_key", columns="eval_method", values=metric, aggfunc="first")
        paired = pivot.dropna(subset=[method_a, method_b])
        delta = paired[method_a] - paired[method_b]
        stat, p = paired_wilcoxon(delta)
        d = cliffs_delta(paired[method_a], paired[method_b])
        out[metric] = {"wilcoxon_stat": float(stat), "p_value": float(p), "cliffs_delta": float(d), "n_pairs": int(len(paired))}
    return out


def outlier_table(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
    similarity_threshold: float = 0.9,
) -> pd.DataFrame:
    """Top-N most expensive misses: exact_match=0 or similarity below threshold."""
    cond = (~df.get("exact_match", False).astype(bool)) | (df.get("similarity", 1.0) < similarity_threshold)
    cols = [c for c in ["repo", "file_name", "difficulty", "eval_method", "total_cost", "similarity", "exact_match"] if c in df.columns]
    res = df.loc[cond, cols].sort_values("total_cost", ascending=False).head(top_n).reset_index(drop=True)
    return res


def correlation_metrics_table(df: pd.DataFrame, cols: Sequence[str] = ("similarity", "bleu3", "rouge_l")) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        return pd.DataFrame()
    return correlation_table(df, present)


def _build_key(df: pd.DataFrame) -> pd.Series:
    if "id" in df.columns:
        return df["id"].astype(str)
    cols = [c for c in ["repo", "file_name"] if c in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df)), index=df.index).astype(str)
    return df[cols].astype(str).agg("__".join, axis=1)

