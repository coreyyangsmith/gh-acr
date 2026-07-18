from __future__ import annotations

"""Tradeoff plots: Pareto frontiers, quality-time, per-success costs, and more."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pareto_scatter(df: pd.DataFrame, *, x: str = "total_cost", y: str = "similarity", hue: str = "eval_method", size: str = "processing_time_s", save_path: Optional[Path] = None, show: bool = True) -> None:
    """Scatterplot of quality vs cost with a simple efficiency frontier per method."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, sizes=(20, 200), alpha=0.7)
    plt.title("Pareto Scatter: Quality vs Cost")
    plt.xlabel(x)
    plt.ylabel(y)

    # Efficiency frontier per method (simple convex hull approximation by sorting)
    for method, g in df.groupby(hue):
        g_sorted = g.sort_values(x)
        frontier_x = []
        frontier_y = []
        best_y = -np.inf
        for _, row in g_sorted.iterrows():
            if row[y] >= best_y:
                frontier_x.append(row[x])
                frontier_y.append(row[y])
                best_y = row[y]
        if len(frontier_x) >= 2:
            plt.plot(frontier_x, frontier_y, label=f"{method} frontier", linewidth=1.5)

    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def quality_vs_time(df: pd.DataFrame, *, x: str = "processing_time_s", y: str = "similarity", hue: str = "eval_method", size: str = "total_cost", save_path: Optional[Path] = None, show: bool = True) -> None:
    """Scatterplot of quality vs time with bubble size proportional to cost."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, sizes=(20, 200), alpha=0.7)
    plt.title("Quality vs Time")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def cost_time_per_success_bars(df: pd.DataFrame, *, save_prefix: Optional[Path] = None, show: bool = True, n_boot: int = 2000) -> None:
    """Bar charts of cost/time per success by method with bootstrap 95% CIs."""
    def _cost_time_ps(g: pd.DataFrame) -> Tuple[float, float]:
        successes = int(g["exact_match"].astype(bool).sum()) if "exact_match" in g.columns else 0
        if successes <= 0:
            return (np.nan, np.nan)
        return (float(g["total_cost"].sum() / successes), float(g["processing_time_s"].sum() / successes))

    def _bootstrap_ci_ratio(g: pd.DataFrame, col: str) -> Tuple[float, float]:
        # Bootstrap cost/time per success by resampling files with replacement within the method
        rng = np.random.default_rng(42)
        n = len(g)
        if n == 0:
            return (np.nan, np.nan)
        stats = []
        for _ in range(n_boot):
            sample = g.iloc[rng.integers(0, n, size=n)]
            successes = int(sample["exact_match"].astype(bool).sum()) if "exact_match" in sample.columns else 0
            if successes <= 0:
                continue
            stats.append(float(sample[col].sum() / successes))
        if not stats:
            return (np.nan, np.nan)
        low, high = np.quantile(stats, [0.025, 0.975])
        return (float(low), float(high))

    methods = []
    cost_vals = []
    time_vals = []
    cost_err_low = []
    cost_err_high = []
    time_err_low = []
    time_err_high = []
    for method, g in df.groupby("eval_method"):
        c_ps, t_ps = _cost_time_ps(g)
        methods.append(method)
        cost_vals.append(c_ps)
        time_vals.append(t_ps)
        lo, hi = _bootstrap_ci_ratio(g, "total_cost")
        cost_err_low.append(c_ps - lo if np.isfinite(c_ps) and np.isfinite(lo) else np.nan)
        cost_err_high.append(hi - c_ps if np.isfinite(c_ps) and np.isfinite(hi) else np.nan)
        lo, hi = _bootstrap_ci_ratio(g, "processing_time_s")
        time_err_low.append(t_ps - lo if np.isfinite(t_ps) and np.isfinite(lo) else np.nan)
        time_err_high.append(hi - t_ps if np.isfinite(t_ps) and np.isfinite(hi) else np.nan)

    # Plot cost per success with error bars
    plt.figure(figsize=(9, 5))
    x = np.arange(len(methods))
    plt.bar(x, cost_vals, yerr=[cost_err_low, cost_err_high], color=sns.color_palette("viridis", len(methods)), capsize=4)
    plt.xticks(x, methods, rotation=20)
    plt.title("Cost per Success by Method (95% CI)")
    plt.ylabel("$ per exact match")
    plt.xlabel("Method")
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(Path(f"{save_prefix}_cost_per_success.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # Plot time per success with error bars
    plt.figure(figsize=(9, 5))
    x = np.arange(len(methods))
    plt.bar(x, time_vals, yerr=[time_err_low, time_err_high], color=sns.color_palette("magma", len(methods)), capsize=4)
    plt.xticks(x, methods, rotation=20)
    plt.title("Time per Success by Method (95% CI)")
    plt.ylabel("seconds per exact match")
    plt.xlabel("Method")
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(Path(f"{save_prefix}_time_per_success.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def slope_chart_by_difficulty(df: pd.DataFrame, *, metric: str = "exact_match", save_path: Optional[Path] = None, show: bool = True) -> None:
    """Line chart of performance across difficulties for each method."""
    if "difficulty" not in df.columns:
        return
    plt.figure(figsize=(10, 7))
    difficulties = ["easy", "medium", "hard"]
    for method, g in df.groupby("eval_method"):
        vals = []
        for d in difficulties:
            gd = g[g["difficulty"] == d]
            if metric == "exact_match":
                vals.append(gd["exact_match"].astype(bool).mean() if not gd.empty else np.nan)
            else:
                vals.append(gd[metric].median() if metric in gd else np.nan)
        plt.plot(difficulties, vals, marker="o", label=str(method))
    plt.title(f"Difficulty Slope Chart ({metric})")
    plt.xlabel("difficulty")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def tokens_to_quality_curve(df: pd.DataFrame, *, bins: int = 10, save_path: Optional[Path] = None, show: bool = True) -> None:
    """Avg similarity per quantile bin of input tokens, per method."""
    if "tokens_total_input" not in df.columns:
        return
    df = df.copy()
    df["tokens_bin"] = pd.qcut(df["tokens_total_input"], q=min(bins, df["tokens_total_input"].nunique()), duplicates="drop")
    plt.figure(figsize=(10, 7))
    for method, g in df.groupby("eval_method"):
        xy = g.groupby("tokens_bin").agg(similarity=("similarity", "mean"))
        plt.plot(range(len(xy)), xy["similarity"], marker="o", label=str(method))
    plt.title("Tokens → Quality Curve (mean similarity by tokens_in bin)")
    plt.xlabel("tokens_in quantile bin")
    plt.ylabel("mean similarity")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def pareto_heatmap_by_repo(df: pd.DataFrame, *, save_path: Optional[Path] = None, show: bool = True) -> None:
    """Cell shows quality rank – cost rank per (repo, method). Higher is better."""
    if "repo" not in df.columns:
        return
    pivot_quality = df.pivot_table(index="repo", columns="eval_method", values="similarity", aggfunc="mean")
    pivot_cost = df.pivot_table(index="repo", columns="eval_method", values="total_cost", aggfunc="mean")
    quality_rank = pivot_quality.rank(axis=1, method="average", ascending=False)
    cost_rank = pivot_cost.rank(axis=1, method="average", ascending=True)
    net = (quality_rank - cost_rank).fillna(0)

    plt.figure(figsize=(12, max(6, len(net) * 0.2)))
    sns.heatmap(net, annot=False, cmap="coolwarm", center=0)
    plt.title("Pareto Heatmap by Repo (quality rank − cost rank)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

