"""Win/Tie/Loss stacked bar charts for RQ1.

Shows trade-offs and regression risk per model.
Answers: "How often does multi-agent hurt?"
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
from .data import compute_win_tie_loss


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


def render_win_tie_loss_chart(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    metrics: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a stacked bar chart showing win/tie/loss per model.

    "Win" = multi-agent outperformed single-agent
    "Tie" = no change
    "Loss" = multi-agent underperformed

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration for the comparison
    metrics : list[str], optional
        Metrics to include (default: all configured metrics)
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if metrics is None:
        metrics = config.metrics

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_win_tie_loss)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Compute win/tie/loss for each metric
    wtl_data = {}
    for metric in available_metrics:
        wtl = compute_win_tie_loss(df, metric, config)
        if not wtl.empty:
            wtl_data[metric] = wtl

    if not wtl_data:
        fig, ax = plt.subplots(figsize=config.figsize_win_tie_loss)
        ax.text(0.5, 0.5, "No win/tie/loss data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create subplot grid
    n_metrics = len(wtl_data)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(config.figsize_win_tie_loss[0], config.figsize_win_tie_loss[1] * n_rows / 1.5),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (metric, wtl) in enumerate(wtl_data.items()):
        ax = axes_flat[idx]

        if wtl.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=14)
            continue

        # Use short model names
        models = [get_short_model_name(m) for m in wtl["model_name"].tolist()]
        x = np.arange(len(models))
        width = 0.65

        # Get percentages
        win_pct = wtl["win_pct"].to_numpy()
        tie_pct = wtl["tie_pct"].to_numpy()
        loss_pct = wtl["loss_pct"].to_numpy()

        # Create stacked bars
        bars_win = ax.bar(
            x, win_pct, width,
            label="Multi-agent wins",
            color=config.multi_agent_color,
            edgecolor="white",
            linewidth=1,
        )
        bars_tie = ax.bar(
            x, tie_pct, width,
            bottom=win_pct,
            label="Tie",
            color=config.tie_color,
            edgecolor="white",
            linewidth=1,
        )
        bars_loss = ax.bar(
            x, loss_pct, width,
            bottom=win_pct + tie_pct,
            label="Multi-agent loses",
            color=config.regression_color,
            edgecolor="white",
            linewidth=1,
        )

        # Add percentage labels inside bars - larger font
        for i, (w, t, l) in enumerate(zip(win_pct, tie_pct, loss_pct)):
            # Win label
            if w > 5:
                ax.text(
                    i, w / 2, f"{w:.0f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white",
                )
            # Tie label
            if t > 5:
                ax.text(
                    i, w + t / 2, f"{t:.0f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white",
                )
            # Loss label
            if l > 5:
                ax.text(
                    i, w + t + l / 2, f"{l:.0f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white",
                )

        # Add count annotations on top
        for i, total in enumerate(wtl["total"].tolist()):
            ax.text(
                i, 103, f"n={total:,}",
                ha="center", va="bottom",
                fontsize=10, fontweight="medium", color="#444",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha="center", fontweight="bold", fontsize=11)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Percentage of Scenarios", fontweight="bold", fontsize=12)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=14)

        # Horizontal line at 50%
        ax.axhline(50, color="black", linestyle=":", alpha=0.4, linewidth=1.5)

        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "RQ1: Win/Tie/Loss - How Often Does Multi-Agent Improve?",
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


def render_win_tie_loss_summary(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a horizontal stacked bar chart for a single metric.

    This is an alternative layout that works well for presentations.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric to visualize
    config : RQ1Config
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
    if metric not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Metric '{metric}' not available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    wtl = compute_win_tie_loss(df, metric, config)

    if wtl.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No win/tie/loss data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(12, max(3, len(wtl) * 1.0)))

    # Use short model names
    models = [get_short_model_name(m) for m in wtl["model_name"].tolist()]
    y = np.arange(len(models))
    height = 0.65

    win_pct = wtl["win_pct"].to_numpy()
    tie_pct = wtl["tie_pct"].to_numpy()
    loss_pct = wtl["loss_pct"].to_numpy()

    # Horizontal stacked bars
    ax.barh(
        y, win_pct, height,
        label="Multi-agent wins",
        color=config.multi_agent_color,
        edgecolor="white",
        linewidth=1,
    )
    ax.barh(
        y, tie_pct, height,
        left=win_pct,
        label="Tie",
        color=config.tie_color,
        edgecolor="white",
        linewidth=1,
    )
    ax.barh(
        y, loss_pct, height,
        left=win_pct + tie_pct,
        label="Multi-agent loses",
        color=config.regression_color,
        edgecolor="white",
        linewidth=1,
    )

    # Add percentage labels - larger font
    for i, (w, t, l) in enumerate(zip(win_pct, tie_pct, loss_pct)):
        if w > 8:
            ax.text(w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        if t > 8:
            ax.text(w + t / 2, i, f"{t:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        if l > 8:
            ax.text(w + t + l / 2, i, f"{l:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    # Add counts on the right
    for i, (total, wins, losses) in enumerate(zip(wtl["total"], wtl["wins"], wtl["losses"])):
        ax.text(102, i, f"+{wins}/-{losses} (n={total:,})", va="center", fontsize=10, fontweight="medium", color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontweight="bold", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Scenarios", fontweight="bold", fontsize=12)
    ax.axvline(50, color="black", linestyle=":", alpha=0.4, linewidth=1.5)

    ax.legend(loc="lower right", frameon=True, fontsize=11)

    metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    ax.set_title(
        f"Win/Tie/Loss: {metric_label}",
        fontweight="bold",
        fontsize=14,
    )

    ax.invert_yaxis()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
