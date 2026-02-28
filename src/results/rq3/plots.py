"""Visualization functions for RQ3 analyses.

Creates publication-quality plots for label distribution, co-occurrence,
method comparison, and stratified analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RQ3Config, DEFAULT_CONFIG, label_to_column_name


logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_label_distribution(
    label_summary: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a bar chart of label frequencies.

    Parameters
    ----------
    label_summary : pd.DataFrame
        Label summary from compute_label_summary
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize_bar)
    
    # Sort by count descending
    data = label_summary.sort_values("count", ascending=True)
    
    # Create horizontal bar chart
    colors = [config.get_label_color(label) for label in data["label"]]
    bars = ax.barh(data["display_name"], data["count"], color=colors, edgecolor="white", linewidth=0.5)
    
    # Add count labels
    for bar, count, pct in zip(bars, data["count"], data["percentage"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{count} ({pct:.1f}%)", va="center", fontsize=9)
    
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_co_occurrence_heatmap(
    co_occurrence_matrix: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a heatmap of label co-occurrence.

    Parameters
    ----------
    co_occurrence_matrix : pd.DataFrame
        Co-occurrence matrix from compute_co_occurrence_matrix
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize_heatmap)
    
    # Filter to labels that appear at least once
    mask = (co_occurrence_matrix.sum(axis=0) > 0) & (co_occurrence_matrix.sum(axis=1) > 0)
    filtered = co_occurrence_matrix.loc[mask, mask]
    
    if filtered.empty:
        ax.text(0.5, 0.5, "No co-occurrences found", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Create display names
        display_names = [config.get_label_display(label) for label in filtered.index]
        
        # Plot heatmap
        sns.heatmap(
            filtered.values.astype(int),
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=display_names,
            yticklabels=display_names,
            ax=ax,
            cbar_kws={"label": "Co-occurrence Count"},
            linewidths=0.5,
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    ax.set_title("Label Co-occurrence Matrix", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_method_comparison_by_label(
    performance_by_label: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a grouped bar chart comparing agent vs bypass by label.

    Parameters
    ----------
    performance_by_label : pd.DataFrame
        Performance statistics from compute_performance_by_label
    metric : str
        Metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize_bar)
    
    agent_col = f"agent_{metric}_with"
    bypass_col = f"bypass_{metric}_with"
    
    # Filter to labels with both metrics
    data = performance_by_label.dropna(subset=[agent_col, bypass_col]).copy()
    
    if data.empty:
        ax.text(0.5, 0.5, f"No data available for {metric}", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Sort by delta
        delta_col = f"delta_{metric}_with"
        if delta_col in data.columns:
            data = data.sort_values(delta_col, ascending=True)
        
        x = np.arange(len(data))
        width = 0.35
        
        agent_bars = ax.barh(x - width/2, data[agent_col], width, label="Agent", color="#1f77b4", alpha=0.8)
        bypass_bars = ax.barh(x + width/2, data[bypass_col], width, label="Bypass", color="#2ca02c", alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(data["display_name"])
        ax.set_xlabel(f"{config.metrics[config.metrics.index(metric)] if metric in config.metrics else metric}", fontsize=12)
        ax.set_title(f"Agent vs Bypass Performance by Label ({metric})", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_performance_delta_by_label(
    performance_by_label: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a forest plot showing performance delta with CIs.

    Parameters
    ----------
    performance_by_label : pd.DataFrame
        Performance statistics from compute_performance_by_label
    metric : str
        Metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize_bar)
    
    delta_col = f"delta_{metric}_with"
    ci_low_col = f"delta_{metric}_ci_low"
    ci_high_col = f"delta_{metric}_ci_high"
    
    # Filter to labels with delta values
    data = performance_by_label.dropna(subset=[delta_col]).copy()
    
    if data.empty:
        ax.text(0.5, 0.5, f"No delta data available for {metric}", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Sort by delta
        data = data.sort_values(delta_col, ascending=True)
        
        y = np.arange(len(data))
        
        # Get error bars if available
        xerr = None
        if ci_low_col in data.columns and ci_high_col in data.columns:
            ci_low = data[delta_col] - data[ci_low_col]
            ci_high = data[ci_high_col] - data[delta_col]
            xerr = [ci_low.fillna(0).values, ci_high.fillna(0).values]
        
        # Color by sign
        colors = ["#2ca02c" if d > 0 else "#d62728" for d in data[delta_col]]
        
        ax.errorbar(
            data[delta_col], y,
            xerr=xerr,
            fmt="o",
            color="none",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            markersize=8,
        )
        
        # Plot points
        for i, (delta, color) in enumerate(zip(data[delta_col], colors)):
            ax.scatter(delta, i, color=color, s=100, zorder=5, edgecolor="white", linewidth=1)
        
        # Add vertical line at 0
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        
        ax.set_yticks(y)
        ax.set_yticklabels(data["display_name"])
        ax.set_xlabel(f"Delta {metric} (Bypass - Agent)", fontsize=12)
        ax.set_title(f"Performance Improvement by Label ({metric})", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Add annotation
        ax.text(0.02, 0.98, "← Agent better | Bypass better →",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                style="italic", color="gray")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_difficulty_interaction(
    stratified_df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a heatmap showing label x difficulty interaction.

    Parameters
    ----------
    stratified_df : pd.DataFrame
        Stratified analysis from compute_stratified_analysis
    metric : str
        Metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize_heatmap)
    
    delta_col = f"delta_{metric}"
    
    # Filter to difficulty stratification
    data = stratified_df[stratified_df["stratify_by"] == "difficulty"].copy()
    
    if data.empty or delta_col not in data.columns:
        ax.text(0.5, 0.5, "No stratified data available", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Pivot to matrix
        pivot = data.pivot_table(
            index="display_name",
            columns="stratum",
            values=delta_col,
            aggfunc="mean",
        )
        
        # Reorder columns if possible
        difficulty_order = ["easy", "medium", "hard"]
        cols = [c for c in difficulty_order if c in pivot.columns]
        cols += [c for c in pivot.columns if c not in cols]
        pivot = pivot[cols]
        
        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": f"Delta {metric}"},
            linewidths=0.5,
        )
        
        ax.set_xlabel("Difficulty", fontsize=12)
        ax.set_ylabel("Label", fontsize=12)
    
    ax.set_title(f"Label x Difficulty Interaction ({metric})", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_project_size_interaction(
    stratified_df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a heatmap showing label x project_size interaction.

    Parameters
    ----------
    stratified_df : pd.DataFrame
        Stratified analysis from compute_stratified_analysis
    metric : str
        Metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize_heatmap)
    
    delta_col = f"delta_{metric}"
    
    # Filter to project_size stratification
    data = stratified_df[stratified_df["stratify_by"] == "project_size"].copy()
    
    if data.empty or delta_col not in data.columns:
        ax.text(0.5, 0.5, "No stratified data available", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Pivot to matrix
        pivot = data.pivot_table(
            index="display_name",
            columns="stratum",
            values=delta_col,
            aggfunc="mean",
        )
        
        # Reorder columns if possible
        size_order = ["small", "medium", "large"]
        cols = [c for c in size_order if c in pivot.columns]
        cols += [c for c in pivot.columns if c not in cols]
        pivot = pivot[cols]
        
        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": f"Delta {metric}"},
            linewidths=0.5,
        )
        
        ax.set_xlabel("Project Size", fontsize=12)
        ax.set_ylabel("Label", fontsize=12)
    
    ax.set_title(f"Label x Project Size Interaction ({metric})", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_violin_by_label(
    paired_df: pd.DataFrame,
    label: str,
    metric: str = "similarity",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a violin plot comparing delta distributions with/without a label.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data with metrics and label columns
    label : str
        Label to analyze
    metric : str
        Metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    col_name = label_to_column_name(label)
    delta_col = f"delta_{metric}"
    
    if col_name not in paired_df.columns or delta_col not in paired_df.columns:
        ax.text(0.5, 0.5, f"No data for {label} / {metric}", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Prepare data
        plot_data = paired_df[[col_name, delta_col]].copy()
        plot_data = plot_data.dropna()
        plot_data["group"] = plot_data[col_name].map({1: f"With {label}", 0: f"Without {label}"})
        
        if len(plot_data) > 0:
            sns.violinplot(
                data=plot_data,
                x="group",
                y=delta_col,
                hue="group",
                ax=ax,
                palette=["#2ca02c", "#1f77b4"],
                inner="box",
                legend=False,
            )
            
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xlabel("")
            ax.set_ylabel(f"Delta {metric} (Bypass - Agent)", fontsize=12)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
    
    ax.set_title(f"Performance Delta by {config.get_label_display(label)}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_all_labels_violin(
    paired_df: pd.DataFrame,
    metric: str = "similarity",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a multi-panel violin plot for all labels.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired data
    metric : str
        Metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    delta_col = f"delta_{metric}"
    
    if delta_col not in paired_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No delta data for {metric}", ha="center", va="center", fontsize=14)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return fig
    
    # Find labels with sufficient data
    valid_labels = []
    for label in config.canonical_labels:
        col_name = label_to_column_name(label)
        if col_name in paired_df.columns:
            n_with = (paired_df[col_name] == 1).sum()
            if n_with >= config.min_samples:
                valid_labels.append(label)
    
    if not valid_labels:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No labels with sufficient data", ha="center", va="center", fontsize=14)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return fig
    
    # Create figure with subplots
    n_labels = len(valid_labels)
    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_labels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, label in enumerate(valid_labels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        col_name = label_to_column_name(label)
        
        # Prepare data
        plot_data = paired_df[[col_name, delta_col]].copy()
        plot_data = plot_data.dropna()
        plot_data["group"] = plot_data[col_name].map({1: "With", 0: "Without"})
        
        if len(plot_data) > 0:
            sns.violinplot(
                data=plot_data,
                x="group",
                y=delta_col,
                hue="group",
                ax=ax,
                palette=["#2ca02c", "#1f77b4"],
                inner="box",
                legend=False,
            )
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        
        ax.set_title(config.get_label_display(label), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Hide unused subplots
    for idx in range(len(valid_labels), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    
    fig.suptitle(f"Delta {metric} by Label (With vs Without)", fontsize=14, fontweight="bold")
    fig.supylabel(f"Delta {metric}", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Complexity Visualization Functions
# =============================================================================

def plot_complexity_by_method(
    complexity_df: pd.DataFrame,
    metric: str = "cc_avg",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create boxplot comparing complexity across methods.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Complexity metrics DataFrame
    metric : str
        Complexity metric to plot
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if complexity_df.empty or metric not in complexity_df.columns:
        fig, ax = plt.subplots(figsize=config.figsize_bar)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
    
    # Filter out parse errors
    valid_df = complexity_df[~complexity_df["parse_error"]].copy()
    
    fig, ax = plt.subplots(figsize=config.figsize_bar)
    
    # Define method order and colors
    method_order = ["a_only", "b_only", "ground_truth", "agent", "bypass"]
    method_order = [m for m in method_order if m in valid_df["method"].unique()]
    
    palette = {
        "a_only": "#7fcdbb",
        "b_only": "#41b6c4",
        "ground_truth": "#2c7fb8",
        "agent": "#d95f02",
        "bypass": "#1b9e77",
    }
    
    sns.boxplot(
        data=valid_df,
        x="method",
        y=metric,
        hue="method",
        order=method_order,
        hue_order=method_order,
        palette=palette,
        ax=ax,
        legend=False,
    )
    
    # Add labels
    metric_labels = {
        "cc_avg": "Average Cyclomatic Complexity",
        "cc_total": "Total Cyclomatic Complexity",
        "cc_max": "Maximum Cyclomatic Complexity",
        "mi_score": "Maintainability Index",
        "sloc": "Source Lines of Code",
        "lloc": "Logical Lines of Code",
        "h_difficulty": "Halstead Difficulty",
        "h_bugs": "Estimated Bugs (Halstead)",
    }
    
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(f"Code Complexity by Method: {metric_labels.get(metric, metric)}", fontsize=14, fontweight="bold")
    
    # Rotate x labels
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_complexity_vs_performance(
    merged_df: pd.DataFrame,
    complexity_metric: str = "cc_avg",
    performance_metric: str = "similarity",
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create scatter plot of complexity vs performance.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with both complexity and performance metrics
    complexity_metric : str
        Complexity metric for x-axis
    performance_metric : str
        Performance metric for y-axis
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if merged_df.empty:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
    
    fig, ax = plt.subplots(figsize=config.figsize_heatmap)
    
    valid = merged_df[[complexity_metric, performance_metric]].dropna()
    
    if len(valid) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=12)
    else:
        ax.scatter(
            valid[complexity_metric],
            valid[performance_metric],
            alpha=0.5,
            edgecolor="none",
            s=30,
        )
        
        # Add trend line
        z = np.polyfit(valid[complexity_metric], valid[performance_metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[complexity_metric].min(), valid[complexity_metric].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Trend")
        
        # Add correlation annotation
        from scipy import stats
        r, pval = stats.pearsonr(valid[complexity_metric], valid[performance_metric])
        ax.annotate(
            f"r = {r:.3f}\np = {pval:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    
    ax.set_xlabel(complexity_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(performance_metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Complexity vs Performance: {complexity_metric} vs {performance_metric}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_mi_distribution(
    complexity_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create distribution plot of Maintainability Index by method.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Complexity metrics DataFrame
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if complexity_df.empty or "mi_score" not in complexity_df.columns:
        fig, ax = plt.subplots(figsize=config.figsize_bar)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
    
    valid_df = complexity_df[~complexity_df["parse_error"]].copy()
    
    fig, ax = plt.subplots(figsize=config.figsize_bar)
    
    method_order = ["a_only", "b_only", "ground_truth", "agent", "bypass"]
    method_order = [m for m in method_order if m in valid_df["method"].unique()]
    
    palette = {
        "a_only": "#7fcdbb",
        "b_only": "#41b6c4",
        "ground_truth": "#2c7fb8",
        "agent": "#d95f02",
        "bypass": "#1b9e77",
    }
    
    for method in method_order:
        method_data = valid_df[valid_df["method"] == method]["mi_score"]
        if len(method_data) > 0:
            sns.kdeplot(
                data=method_data,
                ax=ax,
                label=method,
                color=palette.get(method, "#333333"),
                linewidth=2,
            )
    
    # Add MI grade thresholds
    ax.axvline(x=20, color="green", linestyle="--", alpha=0.5, label="A threshold (20)")
    ax.axvline(x=10, color="orange", linestyle="--", alpha=0.5, label="B threshold (10)")
    
    ax.set_xlabel("Maintainability Index", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Maintainability Index Distribution by Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_complexity_correlation_heatmap(
    correlation_df: pd.DataFrame,
    config: RQ3Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create heatmap of complexity-performance correlations.

    Parameters
    ----------
    correlation_df : pd.DataFrame
        Correlation DataFrame from compute_complexity_performance_correlation
    config : RQ3Config
        Configuration
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if correlation_df.empty:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "No correlation data available", ha="center", va="center", fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
    
    # Pivot for heatmap
    # Use one method (e.g., combined or agent)
    pivot_data = correlation_df.pivot_table(
        index="complexity_metric",
        columns="performance_metric",
        values="spearman_r",
        aggfunc="mean",
    )
    
    if pivot_data.empty:
        fig, ax = plt.subplots(figsize=config.figsize_heatmap)
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center", va="center", fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig
    
    fig, ax = plt.subplots(figsize=config.figsize_heatmap)
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman Correlation"},
    )
    
    ax.set_xlabel("Performance Metric", fontsize=12)
    ax.set_ylabel("Complexity Metric", fontsize=12)
    ax.set_title("Complexity vs Performance Correlation", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
