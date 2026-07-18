"""Visualization functions for quantitative change metrics analysis.

Creates publication-quality plots for:
- Size metrics by version (bar / boxplot)
- Change magnitude distributions
- Commit count distributions
- Correlation heatmaps
- Scatter plots (metric vs performance)
- Label interaction charts
- Summary tables
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import (
    QuantConfig,
    DEFAULT_CONFIG,
    VERSIONS,
    VERSION_DISPLAY_NAMES,
    VERSION_COLORS,
)

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")


# ── Helpers ───────────────────────────────────────────────────────────────


def _save_and_close(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def _version_order_and_palette(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
    """Get ordered version list and colour palette for versions present in *df*."""
    present = [v for v in VERSIONS if v in df["version"].unique()]
    palette = {v: VERSION_COLORS.get(v, "#333333") for v in present}
    return present, palette


# ── 1. Size by version ────────────────────────────────────────────────────


def plot_size_by_version(
    metrics_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Multi-panel boxplot: LOC, SLOC, blank lines, comment lines by version.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of ``process_all_samples`` (one row per sample-version)
    config : QuantConfig
        Configuration
    output_path : Path, optional
        Where to save the figure
    show : bool
        Display interactively

    Returns
    -------
    plt.Figure
    """
    order, palette = _version_order_and_palette(metrics_df)

    fig, axes = plt.subplots(2, 2, figsize=config.figsize_bar)
    axes = axes.flatten()

    panels = [
        ("loc", "Lines of Code"),
        ("sloc", "Source Lines of Code"),
        ("blank_lines", "Blank Lines"),
        ("comment_lines", "Comment Lines"),
    ]

    for idx, (col, title) in enumerate(panels):
        ax = axes[idx]
        if col in metrics_df.columns:
            sns.boxplot(
                data=metrics_df,
                x="version",
                y=col,
                hue="version",
                order=order,
                hue_order=order,
                palette=palette,
                ax=ax,
                legend=False,
            )
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        "Code Size Metrics by Version", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 2. Change magnitude by version ───────────────────────────────────────


def plot_change_magnitude_by_version(
    metrics_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Boxplot of diff_total_change (change magnitude) by version.

    Excludes "previous" since its diff from itself is trivially zero.
    """
    df = metrics_df[metrics_df["version"] != "previous"].copy()
    order = [v for v in VERSIONS if v != "previous" and v in df["version"].unique()]
    palette = {v: VERSION_COLORS.get(v, "#333333") for v in order}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("diff_total_change", "Change Magnitude (Added + Removed)"),
        ("diff_lines_added", "Lines Added"),
        ("diff_lines_removed", "Lines Removed"),
    ]

    for idx, (col, title) in enumerate(panels):
        ax = axes[idx]
        if col in df.columns:
            sns.boxplot(
                data=df,
                x="version",
                y=col,
                hue="version",
                order=order,
                hue_order=order,
                palette=palette,
                ax=ax,
                legend=False,
            )
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        "Diff-Based Change Metrics by Version", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 3. Commit count distribution ─────────────────────────────────────────


def plot_commit_count_distribution(
    metrics_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Histogram of commit counts for branch A vs branch B."""
    # Take one row per sample (commits are the same across versions)
    sample_df = metrics_df.drop_duplicates(subset=["sample_id"]).copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A commits
    ax = axes[0]
    if "n_commits_a" in sample_df.columns:
        vals = sample_df["n_commits_a"].dropna()
        ax.hist(vals, bins=range(0, int(vals.max()) + 2), color="#7fcdbb", edgecolor="white")
        ax.axvline(vals.median(), color="red", linestyle="--", label=f"Median = {vals.median():.0f}")
        ax.legend()
    ax.set_xlabel("Commits (Branch A)")
    ax.set_ylabel("Samples")
    ax.set_title("Branch A Commit Count")

    # B commits
    ax = axes[1]
    if "n_commits_b" in sample_df.columns:
        vals = sample_df["n_commits_b"].dropna()
        ax.hist(vals, bins=range(0, int(vals.max()) + 2), color="#41b6c4", edgecolor="white")
        ax.axvline(vals.median(), color="red", linestyle="--", label=f"Median = {vals.median():.0f}")
        ax.legend()
    ax.set_xlabel("Commits (Branch B)")
    ax.set_ylabel("Samples")
    ax.set_title("Branch B Commit Count")

    # Total commits
    ax = axes[2]
    if "n_commits_total" in sample_df.columns:
        vals = sample_df["n_commits_total"].dropna()
        max_val = int(vals.max()) + 2 if len(vals) > 0 else 10
        ax.hist(vals, bins=range(0, max_val), color="#2c7fb8", edgecolor="white")
        ax.axvline(vals.median(), color="red", linestyle="--", label=f"Median = {vals.median():.0f}")
        ax.legend()
    ax.set_xlabel("Total Commits (A + B)")
    ax.set_ylabel("Samples")
    ax.set_title("Total Commit Count")

    plt.suptitle(
        "Commit Count Distribution", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 4. Change metrics by difficulty ──────────────────────────────────────


def plot_change_by_difficulty(
    metrics_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Grouped bar chart: change magnitude by difficulty level.

    Merges metrics with results CSV to get difficulty labels.
    """
    # Get difficulty per sample from results
    if "difficulty" not in results_df.columns or "id" not in results_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No difficulty data available", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    difficulty_map = (
        results_df[["id", "difficulty"]]
        .drop_duplicates(subset=["id"])
        .copy()
    )
    difficulty_map["id"] = difficulty_map["id"].astype(str)

    # Take ground_truth version only for cleaner comparison
    gt_df = metrics_df[metrics_df["version"] == "ground_truth"].copy()
    gt_df["sample_id"] = gt_df["sample_id"].astype(str)
    gt_df["_base_id"] = gt_df["sample_id"].str.replace(r"-\d+$", "", regex=True)
    merged = gt_df.merge(difficulty_map, left_on="_base_id", right_on="id", how="inner")
    merged = merged.drop(columns=["_base_id"], errors="ignore")

    if merged.empty or "difficulty" not in merged.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No merged data available", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics_to_plot = [
        ("diff_total_change", "Change Magnitude"),
        ("loc_delta", "LOC Delta (vs Ancestor)"),
        ("diff_hunks", "Diff Hunks"),
    ]

    diff_order = ["easy", "medium", "hard"]
    diff_palette = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}

    for idx, (col, label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        if col in merged.columns:
            sns.boxplot(
                data=merged,
                x="difficulty",
                y=col,
                hue="difficulty",
                order=diff_order,
                hue_order=diff_order,
                palette=diff_palette,
                ax=ax,
                legend=False,
            )
        ax.set_xlabel("Difficulty")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Difficulty")

    plt.suptitle(
        "Ground Truth Change Metrics by Difficulty",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 5. Correlation heatmap ────────────────────────────────────────────────


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Heatmap of Spearman correlations (quantitative vs performance).

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of ``compute_performance_correlations``
    """
    if corr_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    # Pivot to matrix
    pivot = corr_df.pivot_table(
        index="quantitative_metric",
        columns="performance_metric",
        values="spearman_r",
    )

    if pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    # Also get p-values for annotation
    p_pivot = corr_df.pivot_table(
        index="quantitative_metric",
        columns="performance_metric",
        values="spearman_p",
    )

    # Build annotation: r value + significance stars
    annot = pivot.copy().astype(str)
    for row in pivot.index:
        for col in pivot.columns:
            r_val = pivot.loc[row, col]
            p_val = p_pivot.loc[row, col] if row in p_pivot.index and col in p_pivot.columns else 1.0
            if pd.isna(r_val):
                annot.loc[row, col] = ""
            else:
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                annot.loc[row, col] = f"{r_val:.2f}{stars}"

    fig, ax = plt.subplots(figsize=config.figsize_heatmap)
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "Quantitative Metrics vs Performance (Spearman r)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Performance Metric")
    ax.set_ylabel("Quantitative Metric")

    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 6. Scatter: metric vs performance ────────────────────────────────────


def plot_metric_vs_performance_scatter(
    deltas_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """2x2 scatter plot of key quantitative metrics vs performance deltas."""

    from .correlations import _prepare_performance_pairs, _coerce_exact_match

    perf_pairs = _prepare_performance_pairs(results_df, config)
    if perf_pairs.empty or deltas_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    deltas_df = deltas_df.copy()
    deltas_df["sample_id"] = deltas_df["sample_id"].astype(str)
    deltas_df["_base_id"] = deltas_df["sample_id"].str.replace(r"-\d+$", "", regex=True)
    perf_pairs["id"] = perf_pairs["id"].astype(str)
    merged = deltas_df.merge(perf_pairs, left_on="_base_id", right_on="id", how="inner")
    merged = merged.drop(columns=["_base_id"], errors="ignore")

    fig, axes = plt.subplots(2, 2, figsize=config.figsize_scatter)

    # Select best available columns
    candidate_plots = [
        ("n_commits_total", "delta_exact_match", "Total Commits", "Delta Exact Match"),
        ("gt_diff_total_change", "delta_exact_match", "GT Change Magnitude", "Delta Exact Match"),
        ("n_commits_total", "delta_similarity", "Total Commits", "Delta Similarity"),
        ("gt_diff_total_change", "delta_similarity", "GT Change Magnitude", "Delta Similarity"),
    ]

    # Fallback if gt_ columns not present
    if "gt_diff_total_change" not in merged.columns:
        candidate_plots = [
            ("n_commits_total", "delta_exact_match", "Total Commits", "Delta Exact Match"),
            ("n_commits_a", "delta_exact_match", "Commits Branch A", "Delta Exact Match"),
            ("n_commits_total", "delta_similarity", "Total Commits", "Delta Similarity"),
            ("n_commits_b", "delta_similarity", "Commits Branch B", "Delta Similarity"),
        ]

    for idx, (x_col, y_col, x_label, y_label) in enumerate(candidate_plots):
        ax = axes.flatten()[idx]

        if x_col not in merged.columns or y_col not in merged.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            continue

        valid = merged[[x_col, y_col]].dropna()
        if len(valid) < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            continue

        ax.scatter(valid[x_col], valid[y_col], alpha=0.4, s=30, edgecolor="none")

        # Trend line
        z = np.polyfit(valid[x_col], valid[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Correlation annotation
        r, pval = stats.spearmanr(valid[x_col], valid[y_col])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        ax.annotate(
            f"r = {r:.3f}{sig}\nn = {len(valid)}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle(
        "Quantitative Metrics vs Performance (Bypass Advantage)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 7. Metrics by label ──────────────────────────────────────────────────


def plot_metrics_by_label(
    deltas_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Grouped bar chart: mean quantitative metrics by label (with vs without).

    Shows the top labels with the largest difference in change magnitude
    between samples with and without the label.
    """
    if deltas_df.empty or paired_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    # Merge (strip suffix for matching)
    deltas_df = deltas_df.copy()
    paired_df = paired_df.copy()
    deltas_df["sample_id"] = deltas_df["sample_id"].astype(str)
    deltas_df["_base_id"] = deltas_df["sample_id"].str.replace(r"-\d+$", "", regex=True)

    if "id" in paired_df.columns:
        paired_df["id"] = paired_df["id"].astype(str)
        merged = deltas_df.merge(
            paired_df, left_on="_base_id", right_on="id", how="inner"
        )
    elif "sample_id" in paired_df.columns:
        paired_df["sample_id"] = paired_df["sample_id"].astype(str)
        paired_df["_base_id"] = paired_df["sample_id"].str.replace(r"-\d+$", "", regex=True)
        merged = deltas_df.merge(paired_df, on="_base_id", how="inner", suffixes=("", "_rq3"))
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Cannot merge data", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    # Identify label columns (binary)
    label_cols = []
    for col in paired_df.columns:
        if col in merged.columns:
            uniq = merged[col].dropna().unique()
            if set(uniq).issubset({0, 1, 0.0, 1.0, True, False}):
                if col not in (
                    "id", "sample_id", "difficulty", "project_size",
                    "source_file", "exact_match",
                ):
                    label_cols.append(col)

    # Pick a key metric to show
    key_metric = "n_commits_total"
    for candidate in ["gt_diff_total_change", "n_commits_total", "gt_sloc"]:
        if candidate in merged.columns:
            key_metric = candidate
            break

    # Compute mean per label (with vs without)
    label_diffs = []
    for label in label_cols:
        with_mean = merged.loc[merged[label] == 1, key_metric].mean()
        without_mean = merged.loc[merged[label] == 0, key_metric].mean()
        n_with = (merged[label] == 1).sum()
        if pd.notna(with_mean) and pd.notna(without_mean) and n_with >= 3:
            label_diffs.append({
                "label": label,
                "mean_with": with_mean,
                "mean_without": without_mean,
                "diff": with_mean - without_mean,
                "n_with": n_with,
            })

    if not label_diffs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No label data available", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    label_df = pd.DataFrame(label_diffs).sort_values("diff", key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=config.figsize_bar)

    x = np.arange(len(label_df))
    width = 0.35

    ax.barh(x - width / 2, label_df["mean_with"], width, label="With Label",
            color="#d95f02", alpha=0.8)
    ax.barh(x + width / 2, label_df["mean_without"], width, label="Without Label",
            color="#1b9e77", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(
        [l.replace("_", " ").title() for l in label_df["label"]],
        fontsize=9,
    )
    ax.set_xlabel(config.get_metric_label(key_metric))
    ax.set_title(
        f"Mean {config.get_metric_label(key_metric)} by Label",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig


# ── 8. Summary table ─────────────────────────────────────────────────────


def plot_summary_table(
    summary_df: pd.DataFrame,
    config: QuantConfig = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Render the version-level summary as a matplotlib table figure."""
    if summary_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No summary data", ha="center", va="center")
        if output_path:
            _save_and_close(fig, output_path, config.dpi)
        return fig

    # Select key columns for the table
    display_cols = ["version", "n_samples"]
    for metric in ["loc", "sloc", "diff_total_change", "diff_hunks"]:
        for agg in ["mean", "median", "std"]:
            col = f"{metric}_{agg}"
            if col in summary_df.columns:
                display_cols.append(col)

    available = [c for c in display_cols if c in summary_df.columns]
    table_df = summary_df[available].copy()

    # Round numeric columns
    for col in table_df.columns:
        if col not in ("version", "n_samples"):
            table_df[col] = table_df[col].round(1)

    # Rename for display
    rename_map = {"version": "Version", "n_samples": "N"}
    for col in table_df.columns:
        if col not in rename_map:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                metric_name = config.get_metric_label(parts[0])
                agg_name = parts[1].title()
                rename_map[col] = f"{metric_name}\n({agg_name})"
    table_df = table_df.rename(columns=rename_map)

    # Map version names to display names
    table_df["Version"] = table_df["Version"].map(
        lambda v: VERSION_DISPLAY_NAMES.get(v, v)
    )

    fig, ax = plt.subplots(
        figsize=(max(12, len(table_df.columns) * 1.5), 1 + len(table_df) * 0.5)
    )
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(table_df.columns))))

    # Style header
    for j in range(len(table_df.columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_df) + 1):
        bg = "#F2F2F2" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(table_df.columns)):
            table[i, j].set_facecolor(bg)

    ax.set_title(
        "Quantitative Metrics Summary by Version",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    if output_path:
        _save_and_close(fig, output_path, config.dpi)
    if show:
        plt.show()
    return fig
