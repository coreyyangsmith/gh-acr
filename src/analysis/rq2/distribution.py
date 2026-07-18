"""Distribution plots (violin/box) for RQ2.

Shows not just mean improvement, but the full spread of Δ scores.
Reveals whether multi-agent is consistently helpful or bimodal.
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
from .data import prepare_improvement_data, create_buckets


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


def render_violin_by_bucket(
    df: pd.DataFrame,
    characteristic: str,
    metric: str = "similarity",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
    kind: str = "violin",  # "violin", "box", or "boxen"
) -> plt.Figure:
    """Render violin or box plots of improvement distribution by bucket.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    characteristic : str
        The characteristic to bucket by
    metric : str
        The metric to show improvement for
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save
    show : bool
        Whether to display
    kind : str
        Plot type: "violin", "box", or "boxen"

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Prepare improvement data
    improvement_data = prepare_improvement_data(df, config)
    if improvement_data.n_pairs == 0:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    # Map to bucket column if needed
    char_mapping = {
        "conflict_size": "conflict_size_bucket",
        "tokens_context": "context_size_bucket",
    }
    char_col = char_mapping.get(characteristic, characteristic)

    delta_col = f"delta_{metric}"

    if char_col not in work.columns or delta_col not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(
            0.5, 0.5,
            f"Characteristic '{characteristic}' or metric '{metric}' not available",
            ha="center", va="center", fontsize=12,
        )
        ax.axis("off")
        return fig

    # Prepare plot data
    plot_data = work[[char_col, delta_col]].dropna()

    if plot_data.empty:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Order buckets if possible
    order = None
    if char_col == "difficulty":
        order = ["easy", "medium", "hard"]
        order = [o for o in order if o in plot_data[char_col].unique()]
    elif "_bucket" in char_col:
        # Try to order buckets by first token size
        unique_buckets = plot_data[char_col].unique().tolist()
        # Simple sorting - numeric prefix
        def extract_first_num(s):
            import re
            match = re.search(r'\d+', str(s))
            return int(match.group()) if match else 999
        order = sorted(unique_buckets, key=extract_first_num)

    fig, ax = plt.subplots(figsize=config.figsize_violin)

    # Create the plot
    palette = {
        bucket: config.positive_color if plot_data[plot_data[char_col] == bucket][delta_col].median() > 0
        else config.negative_color if plot_data[plot_data[char_col] == bucket][delta_col].median() < 0
        else config.neutral_color
        for bucket in plot_data[char_col].unique()
    }

    if kind == "violin":
        sns.violinplot(
            data=plot_data,
            x=char_col,
            y=delta_col,
            order=order,
            palette=palette,
            inner="box",
            cut=0,
            ax=ax,
        )
    elif kind == "box":
        sns.boxplot(
            data=plot_data,
            x=char_col,
            y=delta_col,
            order=order,
            palette=palette,
            showfliers=True,
            ax=ax,
        )
    else:  # boxen
        sns.boxenplot(
            data=plot_data,
            x=char_col,
            y=delta_col,
            order=order,
            palette=palette,
            ax=ax,
        )

    # Add horizontal line at zero
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add sample size annotations
    bucket_order = order if order else plot_data[char_col].unique().tolist()
    for i, bucket in enumerate(bucket_order):
        bucket_data = plot_data[plot_data[char_col] == bucket][delta_col]
        n = len(bucket_data)
        median = bucket_data.median()
        ax.annotate(
            f"n={n}\nmed={median:.3f}",
            xy=(i, ax.get_ylim()[1]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color="#666",
        )

    # Formatting
    ax.set_xlabel(CHARACTERISTIC_DISPLAY_NAMES.get(characteristic, characteristic))
    ax.set_ylabel(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)} (Multi - Single)")

    # Rotate x labels if needed
    if len(bucket_order) > 4:
        plt.xticks(rotation=30, ha="right")

    # Add shaded regions
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], alpha=0.03, color=config.positive_color)
    ax.axhspan(ylim[0], 0, alpha=0.03, color=config.negative_color)

    ax.set_title(
        f"RQ2: Improvement Distribution by {CHARACTERISTIC_DISPLAY_NAMES.get(characteristic, characteristic)}\n"
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


def render_improvement_distributions(
    df: pd.DataFrame,
    characteristics: Optional[list[str]] = None,
    metric: str = "similarity",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
    kind: str = "violin",
) -> plt.Figure:
    """Render a multi-panel distribution plot for multiple characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    characteristics : list[str], optional
        Characteristics to include
    metric : str
        The metric to show improvement for
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save
    show : bool
        Whether to display
    kind : str
        Plot type: "violin", "box", or "boxen"

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Prepare improvement data
    improvement_data = prepare_improvement_data(df, config)
    if improvement_data.n_pairs == 0:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    delta_col = f"delta_{metric}"
    if delta_col not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Determine which characteristics to use
    char_mapping = {
        "difficulty": "difficulty",
        "project_size": "project_size",
        "file_type": "file_type",
        "conflict_size": "conflict_size_bucket",
        "tokens_context": "context_size_bucket",
    }

    if characteristics is None:
        characteristics = ["difficulty", "project_size", "conflict_size", "file_type"]

    available_chars = []
    for char in characteristics:
        char_col = char_mapping.get(char, char)
        if char_col in work.columns:
            available_chars.append((char, char_col))

    if not available_chars:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "No characteristics available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Create subplot grid
    n_chars = len(available_chars)
    n_cols = min(2, n_chars)
    n_rows = (n_chars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(config.figsize_violin[0] * n_cols / 1.5, config.figsize_violin[1] * n_rows / 1.5),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (char_name, char_col) in enumerate(available_chars):
        ax = axes_flat[idx]

        plot_data = work[[char_col, delta_col]].dropna()

        if plot_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(CHARACTERISTIC_DISPLAY_NAMES.get(char_name, char_name))
            continue

        # Order buckets
        order = None
        if char_col == "difficulty":
            order = ["easy", "medium", "hard"]
            order = [o for o in order if o in plot_data[char_col].unique()]
        elif "_bucket" in char_col:
            unique_buckets = plot_data[char_col].unique().tolist()
            def extract_first_num(s):
                import re
                match = re.search(r'\d+', str(s))
                return int(match.group()) if match else 999
            order = sorted(unique_buckets, key=extract_first_num)

        # Color by median improvement
        palette = {
            bucket: config.positive_color if plot_data[plot_data[char_col] == bucket][delta_col].median() > 0
            else config.negative_color if plot_data[plot_data[char_col] == bucket][delta_col].median() < 0
            else config.neutral_color
            for bucket in plot_data[char_col].unique()
        }

        if kind == "violin":
            sns.violinplot(
                data=plot_data, x=char_col, y=delta_col,
                order=order, palette=palette,
                inner="box", cut=0, ax=ax,
            )
        else:
            sns.boxplot(
                data=plot_data, x=char_col, y=delta_col,
                order=order, palette=palette,
                showfliers=True, ax=ax,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("")
        ax.set_ylabel(f"Δ {METRIC_DISPLAY_NAMES.get(metric, metric)}")
        ax.set_title(CHARACTERISTIC_DISPLAY_NAMES.get(char_name, char_name))

        # Rotate labels if needed
        if len(plot_data[char_col].unique()) > 3:
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_ha("right")

    # Hide unused subplots
    for idx in range(n_chars, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"RQ2: Improvement Distributions ({METRIC_DISPLAY_NAMES.get(metric, metric)})",
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


def render_bimodality_analysis(
    df: pd.DataFrame,
    characteristic: str,
    metric: str = "similarity",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render histograms with KDE to check for bimodality.

    Reveals if improvement is unimodal (consistent) or bimodal (sometimes wins, sometimes loses).

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    characteristic : str
        The characteristic to bucket by
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
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "No paired data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    work = create_buckets(improvement_data.dataframe, config)

    char_mapping = {
        "conflict_size": "conflict_size_bucket",
        "tokens_context": "context_size_bucket",
    }
    char_col = char_mapping.get(characteristic, characteristic)

    delta_col = f"delta_{metric}"

    if char_col not in work.columns or delta_col not in work.columns:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Get unique buckets
    buckets = work[char_col].dropna().unique().tolist()
    if not buckets:
        fig, ax = plt.subplots(figsize=config.figsize_violin)
        ax.text(0.5, 0.5, "No buckets found", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Order buckets
    if char_col == "difficulty":
        order = ["easy", "medium", "hard"]
        buckets = [b for b in order if b in buckets] + [b for b in buckets if b not in order]
    elif "_bucket" in char_col:
        def extract_first_num(s):
            import re
            match = re.search(r'\d+', str(s))
            return int(match.group()) if match else 999
        buckets = sorted(buckets, key=extract_first_num)

    # Create subplot grid
    n_buckets = len(buckets)
    n_cols = min(3, n_buckets)
    n_rows = (n_buckets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, bucket in enumerate(buckets):
        ax = axes_flat[idx]
        bucket_data = work[work[char_col] == bucket][delta_col].dropna()

        if len(bucket_data) < 5:
            ax.text(0.5, 0.5, f"n={len(bucket_data)}\n(too few)", ha="center", va="center")
            ax.set_title(str(bucket))
            continue

        # Draw histogram with KDE
        sns.histplot(
            bucket_data,
            kde=True,
            color=config.positive_color if bucket_data.median() > 0 else config.negative_color,
            alpha=0.6,
            ax=ax,
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(bucket_data.median(), color="blue", linestyle="-", linewidth=2, label=f"med={bucket_data.median():.3f}")

        ax.set_xlabel(f"Δ {metric}")
        ax.set_title(f"{bucket} (n={len(bucket_data)})")
        ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for idx in range(n_buckets, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"RQ2: Improvement Distribution by {CHARACTERISTIC_DISPLAY_NAMES.get(characteristic, characteristic)}\n"
        f"(Check for bimodality)",
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
