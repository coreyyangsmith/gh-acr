"""Scatter plot for RQ1: Single vs Multi-agent per scenario.

Proves improvement is broad, not just average.
Points above the diagonal = multi-agent improved.
Points below = multi-agent regressed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RQ1Config, DEFAULT_CONFIG, METRIC_DISPLAY_NAMES, get_short_model_name
from .data import prepare_paired_data


# Publication-quality theme
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
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.weight": "medium",
    },
)


def render_scatter_plot(
    df: pd.DataFrame,
    metric: str = "similarity",
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    use_jitter: bool = True,
    jitter_amount: float = 0.02,
    output_path: Optional[Path] = None,
    show: bool = True,
    facet_by_model: bool = False,
) -> plt.Figure:
    """Render a scatter plot of single-agent vs multi-agent scores per scenario.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric to plot (e.g., "similarity", "exact_match")
    config : RQ1Config
        Configuration for the comparison
    use_jitter : bool
        Whether to add jitter for discrete metrics like exact_match
    jitter_amount : float
        Amount of jitter to add (as fraction of range)
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure
    facet_by_model : bool
        Whether to create separate subplots per model

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Get paired data
    paired_data = prepare_paired_data(df, config)

    if paired_data.n_pairs == 0:
        fig, ax = plt.subplots(figsize=config.figsize_scatter)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    wide_df = paired_data.dataframe
    single_col = f"{config.single_agent_method}_{metric}"
    multi_col = f"{config.multi_agent_method}_{metric}"

    if single_col not in wide_df.columns or multi_col not in wide_df.columns:
        fig, ax = plt.subplots(figsize=config.figsize_scatter)
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Get valid pairs
    valid_pairs = wide_df.dropna(subset=[single_col, multi_col])

    if valid_pairs.empty:
        fig, ax = plt.subplots(figsize=config.figsize_scatter)
        ax.text(0.5, 0.5, "No valid pairs found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    single_vals = valid_pairs[single_col].to_numpy()
    multi_vals = valid_pairs[multi_col].to_numpy()

    # For exact_match (binary), add jitter
    if metric == "exact_match" and use_jitter:
        rng = np.random.default_rng(config.random_state)
        single_vals = single_vals + rng.uniform(-jitter_amount, jitter_amount, len(single_vals))
        multi_vals = multi_vals + rng.uniform(-jitter_amount, jitter_amount, len(multi_vals))

    # Determine plot range
    all_vals = np.concatenate([single_vals, multi_vals])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = 0.05 * (val_max - val_min + 1e-6)
    plot_min = max(0, val_min - margin) if metric in ["exact_match", "similarity", "bleu3", "rouge_l"] else val_min - margin
    plot_max = min(1, val_max + margin) if metric in ["exact_match", "similarity", "bleu3", "rouge_l"] else val_max + margin

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize_scatter)

    # Color points by improvement/regression
    improvement = multi_vals > single_vals
    regression = multi_vals < single_vals
    tie = ~improvement & ~regression

    # Draw diagonal first (so points appear on top)
    ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", linewidth=1.5, alpha=0.5, label="y = x (no change)")

    # Plot points by category
    if tie.any():
        ax.scatter(
            single_vals[tie], multi_vals[tie],
            c=config.tie_color,
            s=50,
            alpha=0.6,
            label=f"Tie (n={tie.sum()})",
            edgecolors="white",
            linewidths=0.5,
        )
    if improvement.any():
        ax.scatter(
            single_vals[improvement], multi_vals[improvement],
            c=config.multi_agent_color,
            s=50,
            alpha=0.6,
            label=f"Multi-agent better (n={improvement.sum()})",
            edgecolors="white",
            linewidths=0.5,
        )
    if regression.any():
        ax.scatter(
            single_vals[regression], multi_vals[regression],
            c=config.regression_color,
            s=50,
            alpha=0.6,
            label=f"Single-agent better (n={regression.sum()})",
            edgecolors="white",
            linewidths=0.5,
        )

    # Add shaded regions
    ax.fill_between(
        [plot_min, plot_max], [plot_min, plot_max], [plot_max, plot_max],
        color=config.multi_agent_color, alpha=0.05, label="_nolegend_"
    )
    ax.fill_between(
        [plot_min, plot_max], [plot_min, plot_max], [plot_min, plot_min],
        color=config.regression_color, alpha=0.05, label="_nolegend_"
    )

    # Add annotation for regions
    ax.text(
        0.05, 0.95, "Multi-agent\nbetter",
        transform=ax.transAxes,
        fontsize=10, color=config.multi_agent_color,
        ha="left", va="top", fontweight="bold",
    )
    ax.text(
        0.95, 0.05, "Single-agent\nbetter",
        transform=ax.transAxes,
        fontsize=10, color=config.regression_color,
        ha="right", va="bottom", fontweight="bold",
    )

    # Formatting
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal")

    metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    ax.set_xlabel(f"{config.get_method_label(config.single_agent_method)} {metric_label}", 
                  fontweight="bold", fontsize=12)
    ax.set_ylabel(f"{config.get_method_label(config.multi_agent_method)} {metric_label}",
                  fontweight="bold", fontsize=12)

    ax.set_title(
        f"Per-Scenario Comparison: {metric_label}\n(n={len(valid_pairs):,} paired scenarios)",
        fontweight="bold",
        fontsize=14,
    )

    ax.legend(loc="lower right", frameon=True, fontsize=10)

    # Add summary statistics - larger font
    n_improved = improvement.sum()
    n_regressed = regression.sum()
    n_tie = tie.sum()
    total = len(valid_pairs)

    stats_text = (
        f"Improved: {n_improved:,} ({100*n_improved/total:.1f}%)\n"
        f"Regressed: {n_regressed:,} ({100*n_regressed/total:.1f}%)\n"
        f"Unchanged: {n_tie:,} ({100*n_tie/total:.1f}%)"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="medium",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#999", linewidth=1.5),
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def render_scatter_plot_by_model(
    df: pd.DataFrame,
    metric: str = "similarity",
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    use_jitter: bool = True,
    jitter_amount: float = 0.02,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render scatter plots faceted by model.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric to plot
    config : RQ1Config
        Configuration
    use_jitter : bool
        Add jitter for discrete metrics
    jitter_amount : float
        Amount of jitter
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if "model_name" not in df.columns:
        # Fall back to single plot
        return render_scatter_plot(
            df, metric, config,
            use_jitter=use_jitter,
            jitter_amount=jitter_amount,
            output_path=output_path,
            show=show,
        )

    models = sorted(df["model_name"].dropna().unique().tolist())
    if not models:
        return render_scatter_plot(
            df, metric, config,
            use_jitter=use_jitter,
            jitter_amount=jitter_amount,
            output_path=output_path,
            show=show,
        )

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        model_df = df[df["model_name"] == model]
        
        # Use short model name for display
        short_name = get_short_model_name(model)

        paired_data = prepare_paired_data(model_df, config, model_name=model)

        if paired_data.n_pairs == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(short_name, fontweight="bold", fontsize=13)
            continue

        wide_df = paired_data.dataframe
        single_col = f"{config.single_agent_method}_{metric}"
        multi_col = f"{config.multi_agent_method}_{metric}"

        if single_col not in wide_df.columns or multi_col not in wide_df.columns:
            ax.text(0.5, 0.5, f"No {metric} data", ha="center", va="center", fontsize=12)
            ax.set_title(short_name, fontweight="bold", fontsize=13)
            continue

        valid_pairs = wide_df.dropna(subset=[single_col, multi_col])

        if valid_pairs.empty:
            ax.text(0.5, 0.5, "No pairs", ha="center", va="center", fontsize=12)
            ax.set_title(short_name, fontweight="bold", fontsize=13)
            continue

        single_vals = valid_pairs[single_col].to_numpy()
        multi_vals = valid_pairs[multi_col].to_numpy()

        if metric == "exact_match" and use_jitter:
            rng = np.random.default_rng(config.random_state + idx)
            single_vals = single_vals + rng.uniform(-jitter_amount, jitter_amount, len(single_vals))
            multi_vals = multi_vals + rng.uniform(-jitter_amount, jitter_amount, len(multi_vals))

        # Plot range
        plot_min, plot_max = 0, 1

        # Diagonal line - thicker
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", linewidth=1.5, alpha=0.6)

        # Color by improvement - larger points
        improvement = multi_vals > single_vals
        regression = multi_vals < single_vals
        tie = ~improvement & ~regression

        if tie.any():
            ax.scatter(single_vals[tie], multi_vals[tie], c=config.tie_color, s=40, alpha=0.6)
        if improvement.any():
            ax.scatter(single_vals[improvement], multi_vals[improvement], c=config.multi_agent_color, s=40, alpha=0.6)
        if regression.any():
            ax.scatter(single_vals[regression], multi_vals[regression], c=config.regression_color, s=40, alpha=0.6)

        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal")
        ax.set_title(f"{short_name}\n(n={len(valid_pairs):,}, +{improvement.sum():,}, −{regression.sum():,})", 
                     fontweight="bold", fontsize=12)

        if idx % n_cols == 0:
            ax.set_ylabel(f"Multi-Agent {metric}", fontweight="bold", fontsize=11)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(f"Single-Agent {metric}", fontweight="bold", fontsize=11)

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    fig.suptitle(
        f"RQ1: Single vs Multi-Agent on {metric_label} by Model",
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
