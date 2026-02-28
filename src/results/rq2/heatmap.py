"""Heatmap visualizations for RQ2 interaction effects.

Shows interaction between two characteristics (e.g., difficulty × conflict size).
Cell value = Δ improvement, revealing where multi-agent helps/hurts most.
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
from .data import prepare_improvement_data, create_buckets, compute_interaction_matrix, compute_interaction_counts


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


def render_difficulty_size_heatmap(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a heatmap of difficulty × conflict size interaction.

    This is the "killer" RQ2 plot showing interaction effects.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
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
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    # Check if we have both characteristics
    row_char = "difficulty"
    col_char = "conflict_size_bucket"

    if row_char not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, f"Column '{row_char}' not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    if col_char not in work.columns:
        # Fall back to tokens_context bucket if available
        if "context_size_bucket" in work.columns:
            col_char = "context_size_bucket"
        else:
            fig, ax = plt.subplots(figsize=config.figsize_heatmap)
            ax.text(0.5, 0.5, f"Column '{col_char}' not available", ha="center", va="center", fontsize=14)
            ax.axis("off")
            return fig

    delta_col = f"delta_{metric}"
    if delta_col not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Compute interaction matrix
    pivot_mean = work.pivot_table(
        index=row_char,
        columns=col_char,
        values=delta_col,
        aggfunc="mean",
    )

    # Compute counts for annotations
    pivot_count = work.pivot_table(
        index=row_char,
        columns=col_char,
        values=delta_col,
        aggfunc="count",
    )

    if pivot_mean.empty:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No data for heatmap", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Order rows (difficulty) - reversed so easy is at bottom, hard at top
    difficulty_order = ["hard", "medium", "easy"]  # Top to bottom in imshow
    ordered_rows = [r for r in difficulty_order if r in pivot_mean.index]
    other_rows = [r for r in pivot_mean.index if r not in difficulty_order]
    pivot_mean = pivot_mean.reindex(ordered_rows + other_rows)
    pivot_count = pivot_count.reindex(ordered_rows + other_rows)

    fig, ax = plt.subplots(figsize=config.figsize_heatmap)

    # Determine color scale (centered at 0)
    vmax = max(abs(pivot_mean.min().min()), abs(pivot_mean.max().max()))
    vmin = -vmax

    # Draw heatmap
    im = ax.imshow(
        pivot_mean.values,
        cmap=config.heatmap_cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)}", fontsize=11)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_mean.columns)))
    ax.set_yticks(np.arange(len(pivot_mean.index)))
    ax.set_xticklabels(pivot_mean.columns, rotation=30, ha="right")
    ax.set_yticklabels(pivot_mean.index)

    ax.set_xlabel(CHARACTERISTIC_DISPLAY_NAMES.get(col_char, col_char))
    ax.set_ylabel(CHARACTERISTIC_DISPLAY_NAMES.get(row_char, row_char))

    # Add cell annotations (value and count)
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            val = pivot_mean.iloc[i, j]
            count = pivot_count.iloc[i, j]
            if not np.isnan(val):
                # Choose text color based on background
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                sign = "+" if val > 0 else ""
                ax.text(
                    j, i, f"{sign}{val:.3f}\n(n={int(count)})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color,
                )

    ax.set_title(
        f"RQ2: Improvement by {CHARACTERISTIC_DISPLAY_NAMES.get(row_char, row_char)} × "
        f"{CHARACTERISTIC_DISPLAY_NAMES.get(col_char, col_char)}\n"
        f"({METRIC_DISPLAY_NAMES.get(metric, metric)}: Green = Multi-agent better, Red = Worse)",
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


def render_interaction_heatmap(
    df: pd.DataFrame,
    row_characteristic: str,
    col_characteristic: str,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a heatmap for any two characteristics.

    Generic version of render_difficulty_size_heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    row_characteristic : str
        Characteristic for rows
    col_characteristic : str
        Characteristic for columns
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
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    # Map to bucket columns if needed
    char_mapping = {
        "conflict_size": "conflict_size_bucket",
        "tokens_context": "context_size_bucket",
    }
    row_char = char_mapping.get(row_characteristic, row_characteristic)
    col_char = char_mapping.get(col_characteristic, col_characteristic)

    if row_char not in work.columns or col_char not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, f"Characteristics not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    delta_col = f"delta_{metric}"
    if delta_col not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Compute interaction matrix
    pivot_mean = work.pivot_table(
        index=row_char,
        columns=col_char,
        values=delta_col,
        aggfunc="mean",
    )

    pivot_count = work.pivot_table(
        index=row_char,
        columns=col_char,
        values=delta_col,
        aggfunc="count",
    )

    if pivot_mean.empty:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No data for heatmap", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Order rows if difficulty - reversed so easy is at bottom, hard at top
    if row_char == "difficulty":
        difficulty_order = ["hard", "medium", "easy"]  # Top to bottom in imshow
        ordered_rows = [r for r in difficulty_order if r in pivot_mean.index]
        other_rows = [r for r in pivot_mean.index if r not in difficulty_order]
        pivot_mean = pivot_mean.reindex(ordered_rows + other_rows)
        pivot_count = pivot_count.reindex(ordered_rows + other_rows)

    # Order columns if project_size - small to large (left to right)
    if col_char == "project_size":
        size_order = ["small", "medium", "large", "huge"]
        ordered_cols = [c for c in size_order if c in pivot_mean.columns]
        other_cols = [c for c in pivot_mean.columns if c not in size_order]
        pivot_mean = pivot_mean[ordered_cols + other_cols]
        pivot_count = pivot_count[ordered_cols + other_cols]

    # Order rows if project_size (when used as rows)
    if row_char == "project_size":
        size_order = ["huge", "large", "medium", "small"]  # Top to bottom (reversed for imshow)
        ordered_rows = [r for r in size_order if r in pivot_mean.index]
        other_rows = [r for r in pivot_mean.index if r not in size_order]
        pivot_mean = pivot_mean.reindex(ordered_rows + other_rows)
        pivot_count = pivot_count.reindex(ordered_rows + other_rows)

    # Order columns if difficulty (when used as columns)
    if col_char == "difficulty":
        difficulty_order = ["easy", "medium", "hard"]  # Left to right
        ordered_cols = [c for c in difficulty_order if c in pivot_mean.columns]
        other_cols = [c for c in pivot_mean.columns if c not in difficulty_order]
        pivot_mean = pivot_mean[ordered_cols + other_cols]
        pivot_count = pivot_count[ordered_cols + other_cols]

    fig, ax = plt.subplots(figsize=config.figsize_heatmap)

    # Determine color scale
    vmax = max(abs(pivot_mean.min().min()), abs(pivot_mean.max().max()))
    vmin = -vmax

    im = ax.imshow(
        pivot_mean.values,
        cmap=config.heatmap_cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)}", fontsize=11)

    ax.set_xticks(np.arange(len(pivot_mean.columns)))
    ax.set_yticks(np.arange(len(pivot_mean.index)))
    ax.set_xticklabels(pivot_mean.columns, rotation=30, ha="right")
    ax.set_yticklabels(pivot_mean.index)

    ax.set_xlabel(CHARACTERISTIC_DISPLAY_NAMES.get(col_characteristic, col_characteristic))
    ax.set_ylabel(CHARACTERISTIC_DISPLAY_NAMES.get(row_characteristic, row_characteristic))

    # Annotations
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            val = pivot_mean.iloc[i, j]
            count = pivot_count.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                sign = "+" if val > 0 else ""
                ax.text(
                    j, i, f"{sign}{val:.3f}\n(n={int(count)})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color,
                )

    ax.set_title(
        f"RQ2: {CHARACTERISTIC_DISPLAY_NAMES.get(row_characteristic, row_characteristic)} × "
        f"{CHARACTERISTIC_DISPLAY_NAMES.get(col_characteristic, col_characteristic)}\n"
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


def render_win_rate_heatmap(
    df: pd.DataFrame,
    row_characteristic: str = "difficulty",
    col_characteristic: str = "conflict_size",
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a heatmap showing win rate (% where multi-agent is better).

    Alternative to mean improvement - shows proportion of wins.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    row_characteristic : str
        Characteristic for rows
    col_characteristic : str
        Characteristic for columns
    metric : str
        The metric for win determination
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
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    char_mapping = {
        "conflict_size": "conflict_size_bucket",
        "tokens_context": "context_size_bucket",
    }
    row_char = char_mapping.get(row_characteristic, row_characteristic)
    col_char = char_mapping.get(col_characteristic, col_characteristic)

    win_col = f"win_{metric}"
    if win_col not in work.columns or row_char not in work.columns or col_char not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "Required columns not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Compute win rate
    pivot_win = work.pivot_table(
        index=row_char,
        columns=col_char,
        values=win_col,
        aggfunc="mean",
    ) * 100  # Convert to percentage

    pivot_count = work.pivot_table(
        index=row_char,
        columns=col_char,
        values=win_col,
        aggfunc="count",
    )

    if pivot_win.empty:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No data for heatmap", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Order rows if difficulty - reversed so easy is at bottom, hard at top
    if row_char == "difficulty":
        difficulty_order = ["hard", "medium", "easy"]  # Top to bottom in imshow
        ordered_rows = [r for r in difficulty_order if r in pivot_win.index]
        other_rows = [r for r in pivot_win.index if r not in difficulty_order]
        pivot_win = pivot_win.reindex(ordered_rows + other_rows)
        pivot_count = pivot_count.reindex(ordered_rows + other_rows)

    # Order columns if project_size - small to large (left to right)
    if col_char == "project_size":
        size_order = ["small", "medium", "large", "huge"]
        ordered_cols = [c for c in size_order if c in pivot_win.columns]
        other_cols = [c for c in pivot_win.columns if c not in size_order]
        pivot_win = pivot_win[ordered_cols + other_cols]
        pivot_count = pivot_count[ordered_cols + other_cols]

    # Order rows if project_size (when used as rows)
    if row_char == "project_size":
        size_order = ["huge", "large", "medium", "small"]  # Top to bottom (reversed for imshow)
        ordered_rows = [r for r in size_order if r in pivot_win.index]
        other_rows = [r for r in pivot_win.index if r not in size_order]
        pivot_win = pivot_win.reindex(ordered_rows + other_rows)
        pivot_count = pivot_count.reindex(ordered_rows + other_rows)

    # Order columns if difficulty (when used as columns)
    if col_char == "difficulty":
        difficulty_order = ["easy", "medium", "hard"]  # Left to right
        ordered_cols = [c for c in difficulty_order if c in pivot_win.columns]
        other_cols = [c for c in pivot_win.columns if c not in difficulty_order]
        pivot_win = pivot_win[ordered_cols + other_cols]
        pivot_count = pivot_count[ordered_cols + other_cols]

    fig, ax = plt.subplots(figsize=config.figsize_heatmap)

    # Win rate scale: 0-100%, centered at 50%
    im = ax.imshow(
        pivot_win.values,
        cmap=config.heatmap_cmap,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Win Rate (%)", fontsize=11)

    ax.set_xticks(np.arange(len(pivot_win.columns)))
    ax.set_yticks(np.arange(len(pivot_win.index)))
    ax.set_xticklabels(pivot_win.columns, rotation=30, ha="right")
    ax.set_yticklabels(pivot_win.index)

    ax.set_xlabel(CHARACTERISTIC_DISPLAY_NAMES.get(col_characteristic, col_characteristic))
    ax.set_ylabel(CHARACTERISTIC_DISPLAY_NAMES.get(row_characteristic, row_characteristic))

    # Annotations
    for i in range(len(pivot_win.index)):
        for j in range(len(pivot_win.columns)):
            val = pivot_win.iloc[i, j]
            count = pivot_count.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 70 or val < 30 else "black"
                ax.text(
                    j, i, f"{val:.0f}%\n(n={int(count)})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color,
                )

    ax.set_title(
        f"RQ2: Multi-Agent Win Rate by {CHARACTERISTIC_DISPLAY_NAMES.get(row_characteristic, row_characteristic)} × "
        f"{CHARACTERISTIC_DISPLAY_NAMES.get(col_characteristic, col_characteristic)}\n"
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
