"""Dumbbell and grouped bar charts for RQ1 single vs multi-agent comparison.

The dumbbell plot shows two dots (single vs multi) per model, connected by a line.
The direction and magnitude of improvement is instantly obvious.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    RQ1Config,
    DEFAULT_CONFIG,
    METRIC_DISPLAY_NAMES,
    METHOD_COLORS,
    METHOD_DISPLAY_NAMES,
    ALL_METHODS_ORDER,
    get_short_model_name,
)
from .data import compute_model_metrics, ModelMetrics, compute_all_methods_metrics


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
        "legend.title_fontsize": 12,
        "font.weight": "medium",
    },
)


def render_dumbbell_chart(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a dumbbell plot comparing single-agent vs multi-agent per model.

    Each model has two dots connected by a line for each metric.
    Points to the right indicate improvement from single to multi-agent.

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
    # Compute metrics per model
    model_metrics = compute_model_metrics(df, config)

    if not model_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_dumbbell)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Determine which metrics have data
    available_metrics = []
    for metric in config.metrics:
        has_data = any(
            not np.isnan(m.single_agent.get(metric, np.nan))
            and not np.isnan(m.multi_agent.get(metric, np.nan))
            for m in model_metrics
        )
        if has_data:
            available_metrics.append(metric)

    if not available_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_dumbbell)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create subplot grid
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(config.figsize_dumbbell[0], config.figsize_dumbbell[1] * n_rows / 2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Sort models by name for consistent ordering - use short names
    model_names = [get_short_model_name(m.model_name) for m in model_metrics]
    y_positions = np.arange(len(model_names))

    for idx, metric in enumerate(available_metrics):
        ax = axes_flat[idx]

        # Extract values and CIs
        single_vals = np.array([m.single_agent.get(metric, np.nan) for m in model_metrics])
        multi_vals = np.array([m.multi_agent.get(metric, np.nan) for m in model_metrics])
        single_ci = [m.single_agent_ci.get(metric, (np.nan, np.nan)) for m in model_metrics]
        multi_ci = [m.multi_agent_ci.get(metric, (np.nan, np.nan)) for m in model_metrics]

        # Scale exact_match to percentage
        scale = 100.0 if metric == "exact_match" else 1.0
        single_vals = single_vals * scale
        multi_vals = multi_vals * scale
        single_ci = [(lo * scale, hi * scale) for lo, hi in single_ci]
        multi_ci = [(lo * scale, hi * scale) for lo, hi in multi_ci]

        # Draw connecting lines
        for i, (s, m) in enumerate(zip(single_vals, multi_vals)):
            if not np.isnan(s) and not np.isnan(m):
                color = config.multi_agent_color if m > s else config.regression_color if m < s else config.tie_color
                ax.plot([s, m], [i, i], color=color, linewidth=2.5, alpha=0.7, zorder=1)

        # Draw error bars and points
        for i, (s_val, s_ci) in enumerate(zip(single_vals, single_ci)):
            if not np.isnan(s_val):
                # Error bar
                if not np.isnan(s_ci[0]) and not np.isnan(s_ci[1]):
                    ax.errorbar(
                        s_val, i,
                        xerr=[[s_val - s_ci[0]], [s_ci[1] - s_val]],
                        color=config.single_agent_color,
                        capsize=4,
                        capthick=2,
                        linewidth=2,
                        zorder=2,
                    )
                # Point
                ax.scatter(s_val, i, color=config.single_agent_color, s=120, zorder=3, edgecolors="white", linewidths=1.5)

        for i, (m_val, m_ci) in enumerate(zip(multi_vals, multi_ci)):
            if not np.isnan(m_val):
                # Error bar
                if not np.isnan(m_ci[0]) and not np.isnan(m_ci[1]):
                    ax.errorbar(
                        m_val, i,
                        xerr=[[m_val - m_ci[0]], [m_ci[1] - m_val]],
                        color=config.multi_agent_color,
                        capsize=4,
                        capthick=2,
                        linewidth=2,
                        zorder=2,
                    )
                # Point
                ax.scatter(m_val, i, color=config.multi_agent_color, s=120, zorder=3, edgecolors="white", linewidths=1.5, marker="s")

        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names, fontweight="bold", fontsize=12)
        ax.set_xlabel(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=12)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=14)

        # Set x-axis limits with padding
        valid_vals = np.concatenate([single_vals[~np.isnan(single_vals)], multi_vals[~np.isnan(multi_vals)]])
        if len(valid_vals) > 0:
            margin = 0.05 * (valid_vals.max() - valid_vals.min() + 1e-6)
            ax.set_xlim(valid_vals.min() - margin, valid_vals.max() + margin)

        ax.invert_yaxis()  # Top model at top
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Add legend
    legend_elements = [
        mpatches.Patch(color=config.single_agent_color, label=config.get_method_label(config.single_agent_method)),
        mpatches.Patch(color=config.multi_agent_color, label=config.get_method_label(config.multi_agent_method)),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    fig.suptitle(
        "RQ1: Single-Agent vs Multi-Agent Performance by Model",
        y=1.06,
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


def render_grouped_bar_chart(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a grouped bar chart comparing single-agent vs multi-agent per model.

    Alternative to dumbbell plot - shows bars side by side with error bars.

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
    # Compute metrics per model
    model_metrics = compute_model_metrics(df, config)

    if not model_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_dumbbell)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Determine which metrics have data
    available_metrics = []
    for metric in config.metrics:
        has_data = any(
            not np.isnan(m.single_agent.get(metric, np.nan))
            and not np.isnan(m.multi_agent.get(metric, np.nan))
            for m in model_metrics
        )
        if has_data:
            available_metrics.append(metric)

    if not available_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_dumbbell)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create subplot grid
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(config.figsize_dumbbell[0], config.figsize_dumbbell[1] * n_rows / 2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Use short model names
    model_names = [get_short_model_name(m.model_name) for m in model_metrics]
    x = np.arange(len(model_names))
    width = 0.35

    for idx, metric in enumerate(available_metrics):
        ax = axes_flat[idx]

        # Extract values and compute error bars
        single_vals = np.array([m.single_agent.get(metric, np.nan) for m in model_metrics])
        multi_vals = np.array([m.multi_agent.get(metric, np.nan) for m in model_metrics])

        # Get CI for error bars
        single_ci = [m.single_agent_ci.get(metric, (np.nan, np.nan)) for m in model_metrics]
        multi_ci = [m.multi_agent_ci.get(metric, (np.nan, np.nan)) for m in model_metrics]

        # Scale exact_match to percentage
        scale = 100.0 if metric == "exact_match" else 1.0
        single_vals = single_vals * scale
        multi_vals = multi_vals * scale

        # Compute error bar sizes (asymmetric)
        single_yerr_low = np.array([v - ci[0] * scale if not np.isnan(v) else 0 for v, ci in zip(single_vals, single_ci)])
        single_yerr_high = np.array([ci[1] * scale - v if not np.isnan(v) else 0 for v, ci in zip(single_vals, single_ci)])
        multi_yerr_low = np.array([v - ci[0] * scale if not np.isnan(v) else 0 for v, ci in zip(multi_vals, multi_ci)])
        multi_yerr_high = np.array([ci[1] * scale - v if not np.isnan(v) else 0 for v, ci in zip(multi_vals, multi_ci)])

        # Create bars
        bars1 = ax.bar(
            x - width / 2,
            np.nan_to_num(single_vals),
            width,
            label=config.get_method_label(config.single_agent_method),
            color=config.single_agent_color,
            yerr=[single_yerr_low, single_yerr_high],
            capsize=4,
            error_kw={"linewidth": 2},
        )
        bars2 = ax.bar(
            x + width / 2,
            np.nan_to_num(multi_vals),
            width,
            label=config.get_method_label(config.multi_agent_method),
            color=config.multi_agent_color,
            yerr=[multi_yerr_low, multi_yerr_high],
            capsize=4,
            error_kw={"linewidth": 2},
        )

        # Add value labels on bars - larger font, bold
        for bar, val in zip(bars1, single_vals):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.1f}" if metric == "exact_match" else f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        for bar, val in zip(bars2, multi_vals):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.1f}" if metric == "exact_match" else f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=12)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=14)

        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=11)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "RQ1: Single-Agent vs Multi-Agent Performance by Model",
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


def render_all_methods_comparison(
    df: pd.DataFrame,
    config: RQ1Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a grouped bar chart comparing ALL methods including baselines.

    Shows baselines as their own model category, then each LLM model 
    with Single-Agent and Multi-Agent results side by side.
    This provides context for how methods compare to trivial baselines.

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
    # Compute metrics for all methods
    all_metrics = compute_all_methods_metrics(df, config)

    if not all_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_comparison)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Determine which metrics have data across all models/methods
    all_methods = config.get_all_methods()
    available_metrics = []
    for metric in config.metrics:
        has_data = any(
            any(not np.isnan(m.methods.get(method, {}).get(metric, np.nan)) for method in all_methods)
            for m in all_metrics
        )
        if has_data:
            available_metrics.append(metric)

    if not available_metrics:
        fig, ax = plt.subplots(figsize=config.figsize_comparison)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create subplot grid
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(config.figsize_comparison[0], config.figsize_comparison[1] * n_rows / 2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Determine applicable methods per model category
    baseline_methods = list(config.baseline_methods)
    llm_methods = [config.single_agent_method, config.multi_agent_method]

    # Build bar positions - use short model names
    model_names = [get_short_model_name(m.model_name) for m in all_metrics]
    
    # Calculate x positions with appropriate spacing
    x_positions = []
    current_x = 0
    bar_width = 0.38
    group_spacing = 0.8
    
    for m in all_metrics:
        if m.model_name == "Baselines":
            # Baselines get bars for base_a, base_b
            n_bars = len(baseline_methods)
        else:
            # LLM models get bars for agent, bypass7
            n_bars = len(llm_methods)
        x_positions.append(current_x + (n_bars - 1) * bar_width / 2)
        current_x += n_bars * bar_width + group_spacing

    x = np.array(x_positions)
    
    # Track which methods we've added to legend
    legend_handles = {}

    for idx, metric in enumerate(available_metrics):
        ax = axes_flat[idx]

        for model_idx, model_data in enumerate(all_metrics):
            # Determine which methods apply to this model
            if model_data.model_name == "Baselines":
                applicable_methods = baseline_methods
            else:
                applicable_methods = llm_methods
            
            n_bars = len(applicable_methods)
            
            for method_idx, method in enumerate(applicable_methods):
                # Get value and CI
                val = model_data.methods.get(method, {}).get(metric, np.nan)
                ci = model_data.methods_ci.get(method, {}).get(metric, (np.nan, np.nan))

                # Scale exact_match to percentage
                scale = 100.0 if metric == "exact_match" else 1.0
                val_scaled = val * scale if not np.isnan(val) else np.nan

                # Compute error bars
                if not np.isnan(val_scaled):
                    yerr_low = val_scaled - ci[0] * scale
                    yerr_high = ci[1] * scale - val_scaled
                else:
                    yerr_low, yerr_high = 0, 0

                # Position this bar
                offset = (method_idx - (n_bars - 1) / 2) * bar_width
                color = config.get_method_color(method)
                label = config.get_method_label(method)

                # Only add to legend once per method
                bar_label = label if method not in legend_handles else None

                bar = ax.bar(
                    x[model_idx] + offset,
                    np.nan_to_num(val_scaled),
                    bar_width * 0.9,
                    label=bar_label,
                    color=color,
                    yerr=[[yerr_low], [yerr_high]],
                    capsize=3,
                    error_kw={"linewidth": 1.5},
                )

                if method not in legend_handles:
                    legend_handles[method] = bar

                # Add value labels on bars - larger, bold
                if not np.isnan(val_scaled) and val_scaled > 0:
                    ax.annotate(
                        f"{val_scaled:.1f}" if metric == "exact_match" else f"{val_scaled:.2f}",
                        xy=(x[model_idx] + offset, val_scaled),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        rotation=0,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=12)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontweight="bold", fontsize=14)

        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=10, title_fontsize=11)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "RQ1: All Methods Comparison (Baselines vs LLM Methods)",
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
