"""Bypass method distribution visualization for RQ1.

Shows how often the multi-agent system chooses Base A, Base B, or MIX
for its final resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RQ1Config, DEFAULT_CONFIG, METHOD_DISPLAY_NAMES, get_short_model_name


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


# Display names for bypass methods
BYPASS_DISPLAY_NAMES: dict[str, str] = {
    "A": "Choose A",
    "B": "Choose B",
    "MIX": "Mix",
}

# Colors for bypass methods (colorblind-friendly palette)
BYPASS_COLORS: dict[str, str] = {
    "A": "#0072B2",      # Blue (colorblind-safe)
    "B": "#E69F00",      # Orange/amber (colorblind-safe)
    "MIX": "#009E73",    # Teal/green (colorblind-safe)
}


@dataclass
class BypassDistribution:
    """Distribution of bypass method choices for a model.
    
    Attributes
    ----------
    model_name : str
        Name of the coding model
    total : int
        Total number of multi-agent scenarios
    counts : dict[str, int]
        {bypass_method: count}
    percentages : dict[str, float]
        {bypass_method: percentage}
    """
    model_name: str
    total: int
    counts: dict[str, int]
    percentages: dict[str, float]


def compute_bypass_distribution(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
) -> list[BypassDistribution]:
    """Compute bypass method distribution for each model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration specifying the multi-agent method
        
    Returns
    -------
    list[BypassDistribution]
        List of per-model bypass method distributions
    """
    if "eval_method" not in df.columns or "bypass_method" not in df.columns:
        raise ValueError("DataFrame must have 'eval_method' and 'bypass_method' columns")
    
    # Filter to multi-agent method only
    work = df[df["eval_method"] == config.multi_agent_method].copy()
    
    if work.empty:
        return []
    
    # Get unique models
    if "model_name" in work.columns:
        models = work["model_name"].dropna().unique().tolist()
    else:
        models = ["unknown"]
        work["model_name"] = "unknown"
    
    results: list[BypassDistribution] = []
    
    for model in sorted(models):
        model_df = work[work["model_name"] == model]
        total = len(model_df)
        
        if total == 0:
            continue
        
        # Count bypass methods
        counts: dict[str, int] = {}
        for bypass_method in ["A", "B", "MIX"]:
            counts[bypass_method] = int((model_df["bypass_method"] == bypass_method).sum())
        
        # Handle any other/unknown bypass methods
        known_count = sum(counts.values())
        if known_count < total:
            counts["Other"] = total - known_count
        
        # Calculate percentages
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in counts.items()}
        
        results.append(BypassDistribution(
            model_name=model,
            total=total,
            counts=counts,
            percentages=percentages,
        ))
    
    return results


def compute_bypass_distribution_per_instance(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
) -> list[BypassDistribution]:
    """Compute bypass method distribution for each model at the instance level.
    
    For instances with multiple files, uses majority voting to determine
    the bypass method for the instance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration specifying the multi-agent method
        
    Returns
    -------
    list[BypassDistribution]
        List of per-model bypass method distributions at instance level
    """
    if "eval_method" not in df.columns or "bypass_method" not in df.columns:
        raise ValueError("DataFrame must have 'eval_method' and 'bypass_method' columns")
    
    if "id" not in df.columns:
        raise ValueError("DataFrame must have 'id' column for instance-level aggregation")
    
    # Filter to multi-agent method only
    work = df[df["eval_method"] == config.multi_agent_method].copy()
    
    if work.empty:
        return []
    
    # Get unique models
    if "model_name" in work.columns:
        models = work["model_name"].dropna().unique().tolist()
    else:
        models = ["unknown"]
        work["model_name"] = "unknown"
    
    results: list[BypassDistribution] = []
    
    for model in sorted(models):
        model_df = work[work["model_name"] == model]
        
        if model_df.empty:
            continue
        
        # Aggregate to instance level using majority voting
        def get_majority_bypass(group: pd.DataFrame) -> str:
            """Get most common bypass method, with tie-breaker order A > B > MIX."""
            counts = group["bypass_method"].value_counts()
            if counts.empty:
                return "A"  # Default
            max_count = counts.max()
            candidates = counts[counts == max_count].index.tolist()
            # Tie-breaker: prefer A, then B, then MIX
            for preferred in ["A", "B", "MIX"]:
                if preferred in candidates:
                    return preferred
            return candidates[0]
        
        instance_bypass = model_df.groupby("id").apply(get_majority_bypass, include_groups=False).reset_index()
        instance_bypass.columns = ["id", "bypass_method"]
        
        total = len(instance_bypass)
        
        if total == 0:
            continue
        
        # Count bypass methods
        counts: dict[str, int] = {}
        for bypass_method in ["A", "B", "MIX"]:
            counts[bypass_method] = int((instance_bypass["bypass_method"] == bypass_method).sum())
        
        # Handle any other/unknown bypass methods
        known_count = sum(counts.values())
        if known_count < total:
            counts["Other"] = total - known_count
        
        # Calculate percentages
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in counts.items()}
        
        results.append(BypassDistribution(
            model_name=model,
            total=total,
            counts=counts,
            percentages=percentages,
        ))
    
    return results


def render_bypass_distribution_bars(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a stacked bar chart showing bypass method distribution per model.
    
    Shows percentage and count for each segment (A, B, Mix).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration for the comparison
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    distributions = compute_bypass_distribution(df, config)
    
    if not distributions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    
    # Setup - use short model names
    model_names = [get_short_model_name(d.model_name) for d in distributions]
    bypass_methods = ["A", "B", "MIX"]
    x = np.arange(len(model_names))
    width = 0.65
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacked bar chart (percentages)
    bottom = np.zeros(len(model_names))
    for bypass_method in bypass_methods:
        values = np.array([d.percentages.get(bypass_method, 0) for d in distributions])
        counts = np.array([d.counts.get(bypass_method, 0) for d in distributions])
        color = BYPASS_COLORS.get(bypass_method, "#333333")
        label = BYPASS_DISPLAY_NAMES.get(bypass_method, bypass_method)
        
        bars = ax.bar(x, values, width, label=label, bottom=bottom, color=color, edgecolor="white", linewidth=1)
        
        # Add percentage and count labels in the middle of each segment
        for i, (val, cnt, bot) in enumerate(zip(values, counts, bottom)):
            if val > 8:  # Only label if segment is large enough
                ax.text(
                    x[i], bot + val / 2,
                    f"{val:.1f}%\n(n={cnt:,})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white"
                )
            elif val > 3:  # Smaller segment - just show percentage
                ax.text(
                    x[i], bot + val / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white"
                )
        
        bottom += values
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Bypass Method Distribution by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    # Include n= counts in x-axis labels to avoid overlap
    xlabels = [f"{name}\n(n={d.total:,})" for name, d in zip(model_names, distributions)]
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 105)
    
    # Legend positioned below the chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=11,
    )
    
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    # Add extra bottom margin for legend
    fig.subplots_adjust(bottom=0.22)
    
    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def render_bypass_distribution_bars_per_instance(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a stacked bar chart showing bypass method distribution per model at instance level.
    
    Shows percentage and count for each segment (A, B, Mix).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration for the comparison
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    distributions = compute_bypass_distribution_per_instance(df, config)
    
    if not distributions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    
    # Setup - use short model names
    model_names = [get_short_model_name(d.model_name) for d in distributions]
    bypass_methods = ["A", "B", "MIX"]
    x = np.arange(len(model_names))
    width = 0.65
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacked bar chart (percentages)
    bottom = np.zeros(len(model_names))
    for bypass_method in bypass_methods:
        values = np.array([d.percentages.get(bypass_method, 0) for d in distributions])
        counts = np.array([d.counts.get(bypass_method, 0) for d in distributions])
        color = BYPASS_COLORS.get(bypass_method, "#333333")
        label = BYPASS_DISPLAY_NAMES.get(bypass_method, bypass_method)
        
        bars = ax.bar(x, values, width, label=label, bottom=bottom, color=color, edgecolor="white", linewidth=1)
        
        # Add percentage and count labels in the middle of each segment
        for i, (val, cnt, bot) in enumerate(zip(values, counts, bottom)):
            if val > 8:  # Only label if segment is large enough
                ax.text(
                    x[i], bot + val / 2,
                    f"{val:.1f}%\n(n={cnt:,})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white"
                )
            elif val > 3:  # Smaller segment - just show percentage
                ax.text(
                    x[i], bot + val / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white"
                )
        
        bottom += values
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Bypass Method Distribution by Model (Per Instance)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    # Include n= counts in x-axis labels to avoid overlap
    xlabels = [f"{name}\n(n={d.total:,})" for name, d in zip(model_names, distributions)]
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 105)
    
    # Legend positioned below the chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=11,
    )
    
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    # Add extra bottom margin for legend
    fig.subplots_adjust(bottom=0.22)
    
    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def render_bypass_pie_charts(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render pie charts showing bypass method distribution per model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ1Config
        Configuration for the comparison
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    distributions = compute_bypass_distribution(df, config)
    
    if not distributions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    
    # Create grid of pie charts
    n_models = len(distributions)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    bypass_methods = ["A", "B", "MIX"]
    
    for idx, dist in enumerate(distributions):
        ax = axes_flat[idx]
        
        # Use short model name
        short_name = get_short_model_name(dist.model_name)
        
        # Get non-zero values
        sizes = []
        labels = []
        colors = []
        
        for method in bypass_methods:
            count = dist.counts.get(method, 0)
            if count > 0:
                sizes.append(count)
                labels.append(f"{BYPASS_DISPLAY_NAMES.get(method, method)}\n({count:,}, {dist.percentages[method]:.1f}%)")
                colors.append(BYPASS_COLORS.get(method, "#333333"))
        
        if sizes:
            wedges, texts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2.5},
            )
            
            # Style labels - larger, bold
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight("bold")
        
        ax.set_title(f"{short_name}\n(n={dist.total:,})", fontsize=14, fontweight="bold")
    
    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")
    
    fig.suptitle(
        "Bypass Method Distribution by Model",
        y=1.02,
        fontsize=16,
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
