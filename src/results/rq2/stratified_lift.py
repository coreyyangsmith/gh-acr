"""Stratified lift plots (forest plots / point-range plots) for RQ2.

Shows improvement (Δ = Multi - Single) per stratum with confidence intervals.
This is the primary RQ2 visualization for showing where multi-agent helps/hurts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RQ2Config, DEFAULT_CONFIG, CHARACTERISTIC_DISPLAY_NAMES, METRIC_DISPLAY_NAMES
from .data import prepare_improvement_data, create_buckets, compute_stratified_metrics


# Set consistent theme
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)


def render_forest_plot(
    stratified_data: pd.DataFrame,
    characteristic: str,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a forest plot (point-range plot) for stratified improvement.

    Parameters
    ----------
    stratified_data : pd.DataFrame
        Output from compute_stratified_metrics containing bucket, mean_delta, ci_low, ci_high
    characteristic : str
        The characteristic being visualized
    metric : str
        The metric being shown
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display
    title : str, optional
        Custom title

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if stratified_data.empty:
        fig, ax = plt.subplots(figsize=config.figsize_forest)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    df = stratified_data.copy()

    # Sort by mean_delta for visual clarity
    df = df.sort_values("mean_delta", ascending=True)

    fig, ax = plt.subplots(figsize=config.figsize_forest)

    y_positions = np.arange(len(df))
    buckets = df["bucket"].tolist()
    means = df["mean_delta"].to_numpy()
    ci_lows = df["ci_low"].to_numpy()
    ci_highs = df["ci_high"].to_numpy()
    counts = df["n"].tolist()

    # Color points by sign of improvement
    colors = [
        config.positive_color if m > 0 else config.negative_color if m < 0 else config.neutral_color
        for m in means
    ]

    # Draw horizontal error bars
    for i, (mean, ci_low, ci_high, color) in enumerate(zip(means, ci_lows, ci_highs, colors)):
        ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.7)
        ax.scatter(mean, i, color=color, s=100, zorder=3, edgecolors="white", linewidths=1)

    # Add vertical line at zero
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="No improvement")

    # Add annotations for each bucket
    for i, (bucket, mean, n) in enumerate(zip(buckets, means, counts)):
        sign = "+" if mean > 0 else ""
        ax.annotate(
            f"{sign}{mean:.3f} (n={n})",
            xy=(mean, i),
            xytext=(10 if mean >= 0 else -10, 0),
            textcoords="offset points",
            ha="left" if mean >= 0 else "right",
            va="center",
            fontsize=9,
            color=colors[i],
            fontweight="bold",
        )

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(buckets)
    ax.set_xlabel(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)} (Multi - Single)")
    ax.set_ylabel(CHARACTERISTIC_DISPLAY_NAMES.get(characteristic, characteristic))

    # Set x-axis limits with padding
    all_vals = np.concatenate([ci_lows[~np.isnan(ci_lows)], ci_highs[~np.isnan(ci_highs)]])
    if len(all_vals) > 0:
        margin = 0.15 * (all_vals.max() - all_vals.min() + 0.01)
        ax.set_xlim(all_vals.min() - margin, all_vals.max() + margin)

    # Add shaded regions
    xlim = ax.get_xlim()
    ax.axvspan(0, xlim[1], alpha=0.05, color=config.positive_color, label="_nolegend_")
    ax.axvspan(xlim[0], 0, alpha=0.05, color=config.negative_color, label="_nolegend_")

    # Add region labels
    ax.text(
        xlim[1] * 0.95, len(df) - 0.5, "Multi-agent\nbetter",
        ha="right", va="top", fontsize=10, color=config.positive_color, fontweight="bold",
    )
    ax.text(
        xlim[0] * 0.95 if xlim[0] < 0 else 0.02, len(df) - 0.5, "Single-agent\nbetter",
        ha="left", va="top", fontsize=10, color=config.negative_color, fontweight="bold",
    )

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(
            f"RQ2: Improvement by {CHARACTERISTIC_DISPLAY_NAMES.get(characteristic, characteristic)}\n"
            f"({METRIC_DISPLAY_NAMES.get(metric, metric)})",
            fontweight="bold",
        )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def render_stratified_lift_by_characteristic(
    df: pd.DataFrame,
    characteristics: Optional[list[str]] = None,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a multi-panel forest plot for multiple characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe (raw, not yet processed)
    characteristics : list[str], optional
        Characteristics to include. Defaults to all available.
    metric : str
        The metric to analyze
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Prepare improvement data
    improvement_data = prepare_improvement_data(df, config)
    if improvement_data.n_pairs == 0:
        fig, ax = plt.subplots(figsize=config.figsize_forest)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    # Determine which characteristics to use
    available_chars = []
    char_mapping = {
        "difficulty": "difficulty",
        "project_size": "project_size",
        "file_type": "file_type",
        "conflict_size_bucket": "conflict_size_bucket",
        "context_size_bucket": "context_size_bucket",
    }

    if characteristics is None:
        characteristics = list(char_mapping.keys())

    for char in characteristics:
        if char in work.columns:
            available_chars.append(char)
        elif char in char_mapping and char_mapping[char] in work.columns:
            available_chars.append(char_mapping[char])

    if not available_chars:
        fig, ax = plt.subplots(figsize=config.figsize_forest)
        ax.text(0.5, 0.5, "No characteristics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Compute stratified metrics for each
    stratified_results = {}
    for char in available_chars:
        result = compute_stratified_metrics(work, char, metric, config)
        if not result.data.empty:
            stratified_results[char] = result.data

    if not stratified_results:
        fig, ax = plt.subplots(figsize=config.figsize_forest)
        ax.text(0.5, 0.5, "No stratified data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create subplot grid
    n_chars = len(stratified_results)
    n_cols = min(2, n_chars)
    n_rows = (n_chars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(config.figsize_forest[0] * n_cols / 1.5, config.figsize_forest[1] * n_rows / 2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (char, data) in enumerate(stratified_results.items()):
        ax = axes_flat[idx]

        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(CHARACTERISTIC_DISPLAY_NAMES.get(char, char))
            continue

        # Sort by mean_delta
        df_sorted = data.sort_values("mean_delta", ascending=True)

        y_positions = np.arange(len(df_sorted))
        buckets = df_sorted["bucket"].tolist()
        means = df_sorted["mean_delta"].to_numpy()
        ci_lows = df_sorted["ci_low"].to_numpy()
        ci_highs = df_sorted["ci_high"].to_numpy()
        counts = df_sorted["n"].tolist()

        colors = [
            config.positive_color if m > 0 else config.negative_color if m < 0 else config.neutral_color
            for m in means
        ]

        # Draw error bars and points
        for i, (mean, ci_low, ci_high, color) in enumerate(zip(means, ci_lows, ci_highs, colors)):
            ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.7)
            ax.scatter(mean, i, color=color, s=60, zorder=3, edgecolors="white", linewidths=0.5)

        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

        # Add count annotations
        for i, (mean, n) in enumerate(zip(means, counts)):
            ax.annotate(
                f"n={n}",
                xy=(ax.get_xlim()[1] if mean >= 0 else ax.get_xlim()[0], i),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color="#666",
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(buckets, fontsize=9)
        ax.set_xlabel(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)}")
        ax.set_title(CHARACTERISTIC_DISPLAY_NAMES.get(char, char))

        # Shaded regions
        xlim = ax.get_xlim()
        ax.axvspan(0, xlim[1], alpha=0.03, color=config.positive_color)
        ax.axvspan(xlim[0], 0, alpha=0.03, color=config.negative_color)

    # Hide unused subplots
    for idx in range(n_chars, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"RQ2: Where Does Multi-Agent Help/Hurt? (by {METRIC_DISPLAY_NAMES.get(metric, metric)})",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def render_single_characteristic_forest(
    df: pd.DataFrame,
    characteristic: str,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a single forest plot for one characteristic.

    Convenience wrapper that prepares data and calls render_forest_plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    characteristic : str
        The characteristic to stratify by
    metric : str
        The metric to analyze
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Prepare improvement data
    improvement_data = prepare_improvement_data(df, config)
    if improvement_data.n_pairs == 0:
        fig, ax = plt.subplots(figsize=config.figsize_forest)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    # Use bucket version if available
    char_col = characteristic
    if characteristic == "conflict_size" and "conflict_size_bucket" in work.columns:
        char_col = "conflict_size_bucket"
    elif characteristic == "tokens_context" and "context_size_bucket" in work.columns:
        char_col = "context_size_bucket"

    result = compute_stratified_metrics(work, char_col, metric, config)

    return render_forest_plot(
        result.data,
        characteristic,
        metric,
        config,
        output_path=output_path,
        show=show,
    )
