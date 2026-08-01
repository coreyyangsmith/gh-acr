"""Plots for Better-Judge ablation analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    AblationConfig,
    DEFAULT_CONFIG,
    get_component_label,
    get_method_color,
    get_method_label,
    get_short_model_name,
    METRIC_DISPLAY_NAMES,
)


sns.set_theme(
    style="whitegrid",
    rc={
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    },
)


def _save(fig: plt.Figure, output_path: Optional[Path], dpi: int, show: bool) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_component_forest(
    contributions: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "exact_match",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Forest plot of mean Δ (better_judge − ablation) per component × model."""
    sub = contributions[contributions["metric"] == metric].copy() if not contributions.empty else contributions
    fig, ax = plt.subplots(figsize=(10, 6))
    if sub.empty:
        ax.text(0.5, 0.5, f"No contribution data for {metric}", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    models = sorted(sub["model_name"].unique())
    ablations = [a for a in config.ablations if a in sub["ablation"].unique()]
    y_positions = []
    y_labels = []
    y = 0
    for ablation in ablations:
        for model in models:
            row = sub[(sub["ablation"] == ablation) & (sub["model_name"] == model)]
            if row.empty:
                continue
            r = row.iloc[0]
            color = get_method_color(ablation)
            ax.errorbar(
                r["mean_delta"],
                y,
                xerr=[[r["mean_delta"] - r["ci_low"]], [r["ci_high"] - r["mean_delta"]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=7,
            )
            y_positions.append(y)
            y_labels.append(f"{get_component_label(ablation)} | {get_short_model_name(model)}")
            y += 1
        y += 0.4  # gap between components

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    label = METRIC_DISPLAY_NAMES.get(metric, metric)
    ax.set_xlabel(f"Δ {label} (Better-Judge − ablation)")
    ax.set_title(f"Component contribution — {label}")
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_ablation_ladder(
    ladder: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "exact_match",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Grouped bars: mean quality per method on the common ID set."""
    sub = ladder[ladder["metric"] == metric].copy() if not ladder.empty else ladder
    fig, ax = plt.subplots(figsize=(12, 6))
    if sub.empty:
        ax.text(0.5, 0.5, f"No ladder data for {metric}", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    models = sorted(sub["model_name"].unique())
    methods = [m for m in config.ladder_methods() if m in sub["method"].unique()]
    x = np.arange(len(methods))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        vals, yerr_lo, yerr_hi = [], [], []
        for method in methods:
            row = sub[(sub["method"] == method) & (sub["model_name"] == model)]
            if row.empty:
                vals.append(np.nan)
                yerr_lo.append(0)
                yerr_hi.append(0)
            else:
                r = row.iloc[0]
                scale = 100.0 if metric == "exact_match" else 1.0
                mean = r["mean"] * scale
                vals.append(mean)
                yerr_lo.append(mean - r["ci_low"] * scale)
                yerr_hi.append(r["ci_high"] * scale - mean)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            label=get_short_model_name(model),
            yerr=np.array([yerr_lo, yerr_hi]),
            capsize=2,
            color=plt.cm.Set2(i),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([get_method_label(m) for m in methods], rotation=30, ha="right")
    ylabel = f"{METRIC_DISPLAY_NAMES.get(metric, metric)}" + (" %" if metric == "exact_match" else "")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Ablation ladder — {METRIC_DISPLAY_NAMES.get(metric, metric)}")
    ax.legend(title="Model")
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_wtl_bars(
    wtl: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    comparison: str = "anchor_vs_ablation",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Stacked win/tie/loss bars for a comparison family."""
    sub = wtl[wtl["comparison"] == comparison].copy() if not wtl.empty else wtl
    fig, ax = plt.subplots(figsize=(11, 6))
    if sub.empty:
        ax.text(0.5, 0.5, f"No WTL data for {comparison}", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    sub["label"] = sub.apply(
        lambda r: f"{get_method_label(r['method_a'])} vs {get_method_label(r['method_b'])}\n"
        f"{get_short_model_name(r['model_name'])}",
        axis=1,
    )
    labels = sub["label"].tolist()
    y = np.arange(len(sub))
    wins = sub["win_pct"].fillna(0).to_numpy()
    ties = sub["tie_pct"].fillna(0).to_numpy()
    losses = sub["loss_pct"].fillna(0).to_numpy()

    ax.barh(y, wins, color="#2ca02c", label="Win (A better)")
    ax.barh(y, ties, left=wins, color="#7f7f7f", label="Tie")
    ax.barh(y, losses, left=wins + ties, color="#d62728", label="Loss (A worse)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Proportion")
    ax.set_xlim(0, 1)
    title = (
        "Better-Judge vs ablations (WTL)"
        if comparison == "anchor_vs_ablation"
        else "Method vs single-agent (WTL)"
    )
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_cost_quality_pareto(
    cost_df: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    quality_col: str = "mean_similarity",
    cost_col: str = "mean_total_cost",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Scatter of mean quality vs mean cost; annotate dominated points lightly."""
    fig, ax = plt.subplots(figsize=(10, 7))
    if cost_df.empty or quality_col not in cost_df.columns or cost_col not in cost_df.columns:
        ax.text(0.5, 0.5, "No cost/quality data", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    for model, grp in cost_df.groupby("model_name"):
        for _, r in grp.iterrows():
            ax.scatter(
                r[cost_col],
                r[quality_col],
                s=90,
                color=get_method_color(r["method"]),
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                f"{get_method_label(r['method'])}\n{get_short_model_name(model)}",
                (r[cost_col], r[quality_col]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

    # Mark simple Pareto front per model (max quality, min cost)
    for model, grp in cost_df.groupby("model_name"):
        pts = grp[[cost_col, quality_col, "method"]].dropna()
        if pts.empty:
            continue
        # Sort by cost ascending; keep decreasing quality envelope
        pts = pts.sort_values(cost_col)
        best_q = -np.inf
        front_x, front_y = [], []
        for _, r in pts.iterrows():
            if r[quality_col] > best_q:
                best_q = r[quality_col]
                front_x.append(r[cost_col])
                front_y.append(r[quality_col])
        if front_x:
            ax.plot(front_x, front_y, "--", alpha=0.4, label=f"Pareto ({get_short_model_name(model)})")

    ax.set_xlabel(cost_col.replace("mean_", "").replace("_", " ").title())
    ax.set_ylabel(quality_col.replace("mean_", "").replace("_", " ").title())
    ax.set_title("Cost–quality trade-off across ablations")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_stratified_forest(
    stratified: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    stratum: str = "difficulty",
    model_name: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Forest of component Δ by stratum bucket."""
    sub = stratified[stratified["stratum"] == stratum].copy() if not stratified.empty else stratified
    if model_name is not None and not sub.empty:
        sub = sub[sub["model_name"] == model_name]
    fig, ax = plt.subplots(figsize=(11, 7))
    if sub.empty:
        ax.text(0.5, 0.5, f"No stratified data for {stratum}", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    y = 0
    yticks, ylabels = [], []
    for ablation in config.ablations:
        abl_rows = sub[sub["ablation"] == ablation]
        if abl_rows.empty:
            continue
        for _, r in abl_rows.sort_values("bucket").iterrows():
            ax.errorbar(
                r["mean_delta"],
                y,
                xerr=[[r["mean_delta"] - r["ci_low"]], [r["ci_high"] - r["mean_delta"]]],
                fmt="o",
                color=get_method_color(ablation),
                capsize=2,
                markersize=6,
            )
            yticks.append(y)
            model_s = get_short_model_name(r["model_name"])
            ylabels.append(f"{get_component_label(ablation)} | {r['bucket']} | {model_s}")
            y += 1
        y += 0.5

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ quality (Better-Judge − ablation)")
    ax.set_title(f"Component effect by {stratum}")
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_difficulty_component_heatmap(
    stratified: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    model_name: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Heatmap of mean component Δ by difficulty × ablation."""
    sub = stratified[stratified["stratum"] == "difficulty"].copy() if not stratified.empty else stratified
    if model_name is not None and not sub.empty:
        sub = sub[sub["model_name"] == model_name]
    fig, ax = plt.subplots(figsize=(9, 5))
    if sub.empty:
        ax.text(0.5, 0.5, "No difficulty×component data", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    # Average across models if multiple
    pivot = sub.pivot_table(
        index="ablation",
        columns="bucket",
        values="mean_delta",
        aggfunc="mean",
    )
    # Order rows/cols
    pivot = pivot.reindex([a for a in config.ablations if a in pivot.index])
    pivot.index = [get_component_label(i) for i in pivot.index]
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax)
    title = "Component Δ by difficulty"
    if model_name:
        title += f" — {get_short_model_name(model_name)}"
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_routing_conditional(
    conditional: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Bar chart: BJ − no_judge delta conditioned on BJ's A/B/MIX decision."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = conditional.copy() if conditional is not None else pd.DataFrame()
    if sub.empty:
        ax.text(0.5, 0.5, "No routing counterfactual data", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    sub = sub[sub["bj_decision"] != "ALL"] if "bj_decision" in sub.columns else sub
    models = sorted(sub["model_name"].unique())
    decisions = sorted(sub["bj_decision"].unique())
    x = np.arange(len(decisions))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        means, errs_lo, errs_hi = [], [], []
        for d in decisions:
            row = sub[(sub["model_name"] == model) & (sub["bj_decision"] == d)]
            if row.empty:
                means.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
            else:
                r = row.iloc[0]
                means.append(r["mean_delta"])
                errs_lo.append(r["mean_delta"] - r["ci_low"])
                errs_hi.append(r["ci_high"] - r["mean_delta"])
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.9,
            yerr=np.array([errs_lo, errs_hi]),
            capsize=3,
            label=get_short_model_name(model),
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(decisions)
    ax.set_xlabel("Better-Judge routing decision")
    ax.set_ylabel("Δ similarity (BJ − no judge)")
    ax.set_title("Analyzer value by routing decision")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig


def render_cross_model_stability(
    stability: pd.DataFrame,
    config: AblationConfig = DEFAULT_CONFIG,
    *,
    metric: str = "exact_match",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Paired bars of component Δ for two models."""
    sub = stability[stability["metric"] == metric].copy() if not stability.empty else stability
    fig, ax = plt.subplots(figsize=(10, 5))
    if sub.empty:
        ax.text(0.5, 0.5, "Need ≥2 models for cross-model plot", ha="center", va="center")
        ax.axis("off")
        _save(fig, output_path, config.dpi, show)
        return fig

    ablations = [a for a in config.ablations if a in sub["ablation"].unique()]
    x = np.arange(len(ablations))
    width = 0.35
    # Use first model pair in the frame
    m_a = sub["model_a"].iloc[0]
    m_b = sub["model_b"].iloc[0]
    deltas_a, deltas_b = [], []
    for abl in ablations:
        row = sub[sub["ablation"] == abl]
        if row.empty:
            deltas_a.append(0)
            deltas_b.append(0)
        else:
            r = row.iloc[0]
            deltas_a.append(r["delta_a"])
            deltas_b.append(r["delta_b"])

    ax.bar(x - width / 2, deltas_a, width, label=get_short_model_name(m_a), color="#1f77b4")
    ax.bar(x + width / 2, deltas_b, width, label=get_short_model_name(m_b), color="#ff7f0e")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([get_component_label(a) for a in ablations], rotation=20, ha="right")
    ax.set_ylabel(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)} (BJ − ablation)")
    ax.set_title("Cross-model component stability")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path, config.dpi, show)
    return fig
