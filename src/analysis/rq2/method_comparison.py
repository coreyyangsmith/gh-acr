"""Method comparison visualizations for RQ2.

Shows performance of baselines vs single-agent vs multi-agent
stratified by difficulty and project size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RQ2Config, DEFAULT_CONFIG, get_short_model_name


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
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.weight": "medium",
    },
)


# Method display names and colors
METHOD_DISPLAY_NAMES = {
    "base_a": "Base A",
    "base_b": "Base B",
    "agent": "Single",
    "bypass7": "Multi",
}

# Model-specific color palettes (single-agent lighter, multi-agent darker)
MODEL_COLOR_PALETTES = {
    "Qwen": {"agent": "#7fbfff", "bypass7": "#1f77b4"},      # Light blue → Dark blue
    "Llama": {"agent": "#90ee90", "bypass7": "#228b22"},    # Light green → Forest green
    "GPT": {"agent": "#ffb3b3", "bypass7": "#d62728"},      # Light red → Dark red
}

# Baseline colors (neutral, distinct)
BASELINE_COLORS = {
    "base_a": "#888888",      # Gray
    "base_b": "#555555",      # Darker gray
}

# Hatching patterns for additional differentiation
MODEL_HATCHES = {
    "Qwen": "",
    "Llama": "//",
    "GPT": "\\\\",
}

METHOD_ORDER = ["base_a", "base_b", "agent", "bypass7"]


def _get_bar_color(method: str, model: str) -> str:
    """Get bar color based on method and model."""
    if method in BASELINE_COLORS:
        return BASELINE_COLORS[method]
    
    # Find which model palette to use
    for model_key, palette in MODEL_COLOR_PALETTES.items():
        if model_key.lower() in model.lower():
            return palette.get(method, "#999999")
    
    # Fallback
    return "#999999"


def _get_bar_hatch(model: str) -> str:
    """Get bar hatch pattern based on model."""
    for model_key, hatch in MODEL_HATCHES.items():
        if model_key.lower() in model.lower():
            return hatch
    return ""


def _coerce_bool_metric(series: pd.Series) -> pd.Series:
    """Coerce exact_match to numeric 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "1.0", "yes", "y", "t"])
        .astype(float)
    )


def compute_method_performance_by_stratum(
    df: pd.DataFrame,
    stratum_col: str,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute performance for each method within each stratum.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    stratum_col : str
        Column to stratify by (e.g., "difficulty", "project_size")
    metric : str
        Metric to compute (exact_match or similarity)
    config : RQ2Config
        Configuration
        
    Returns
    -------
    pd.DataFrame
        Performance summary with columns: stratum, method, model, metric_value, n, ci_low, ci_high
    """
    work = df.copy()
    
    # Coerce exact_match
    if metric == "exact_match" and metric in work.columns:
        work[metric] = _coerce_bool_metric(work[metric])
    
    # Ensure stratum column exists
    if stratum_col not in work.columns:
        raise ValueError(f"Column '{stratum_col}' not found in dataframe")
    
    # Normalize stratum values
    work[stratum_col] = work[stratum_col].astype(str).str.lower().str.strip()
    
    results = []
    
    # For each stratum
    for stratum in work[stratum_col].dropna().unique():
        stratum_df = work[work[stratum_col] == stratum]
        
        # For each eval_method
        for method in METHOD_ORDER:
            method_df = stratum_df[stratum_df["eval_method"] == method]
            
            if method_df.empty:
                continue
            
            # For baselines, there's no model_name
            if method in ["base_a", "base_b"]:
                models = ["Baseline"]
            else:
                models = method_df["model_name"].dropna().unique().tolist()
                if not models:
                    models = ["Unknown"]
            
            for model in models:
                if method in ["base_a", "base_b"]:
                    model_method_df = method_df
                else:
                    model_method_df = method_df[method_df["model_name"] == model]
                
                if model_method_df.empty or metric not in model_method_df.columns:
                    continue
                
                values = pd.to_numeric(model_method_df[metric], errors="coerce").dropna()
                n = len(values)
                
                if n == 0:
                    continue
                
                mean_val = values.mean()
                
                # Bootstrap CI
                ci_low, ci_high = _bootstrap_ci(values.to_numpy())
                
                results.append({
                    "stratum": stratum,
                    "method": method,
                    "method_display": METHOD_DISPLAY_NAMES.get(method, method),
                    "model": get_short_model_name(model) if model != "Baseline" else model,
                    "metric": metric,
                    "value": mean_val,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })
    
    return pd.DataFrame(results)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return (np.nan, np.nan)
    
    boot_means = np.empty(n_boot)
    n = len(clean)
    for i in range(n_boot):
        sample = clean[rng.integers(0, n, size=n)]
        boot_means[i] = np.mean(sample)
    
    alpha = (1 - ci_level) / 2
    return (float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha)))


def render_method_comparison_by_difficulty(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render grouped bar chart comparing methods across difficulty levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        Metric to plot
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save figure
    show : bool
        Display figure
        
    Returns
    -------
    plt.Figure
    """
    perf_df = compute_method_performance_by_stratum(df, "difficulty", metric, config)
    
    if perf_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig
    
    # Create combined method-model labels for non-baselines
    perf_df["method_model"] = perf_df.apply(
        lambda r: r["method_display"] if r["model"] == "Baseline" 
        else f"{r['model']} ({r['method_display']})", axis=1
    )
    
    # Store original model for color lookup
    perf_df["model_orig"] = perf_df["model"]
    
    # Order difficulty
    difficulty_order = ["easy", "medium", "hard"]
    perf_df["stratum"] = pd.Categorical(perf_df["stratum"], categories=difficulty_order, ordered=True)
    perf_df = perf_df.sort_values("stratum")
    
    # Get unique methods (in order): baselines first, then models grouped
    unique_methods = []
    method_info = {}  # Store method and model for color lookup
    
    # First add baselines
    for method in ["base_a", "base_b"]:
        method_rows = perf_df[perf_df["method"] == method]
        for _, row in method_rows.drop_duplicates("method_model").iterrows():
            if row["method_model"] not in unique_methods:
                unique_methods.append(row["method_model"])
                method_info[row["method_model"]] = (method, row["model_orig"])
    
    # Then add models: group by model, then single/multi within each
    model_list = perf_df[~perf_df["method"].isin(["base_a", "base_b"])]["model"].unique()
    for model in sorted(model_list):
        for method in ["agent", "bypass7"]:
            method_rows = perf_df[(perf_df["method"] == method) & (perf_df["model"] == model)]
            for _, row in method_rows.drop_duplicates("method_model").iterrows():
                if row["method_model"] not in unique_methods:
                    unique_methods.append(row["method_model"])
                    method_info[row["method_model"]] = (method, row["model_orig"])
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strata = [s for s in difficulty_order if s in perf_df["stratum"].values]
    x = np.arange(len(strata))
    n_methods = len(unique_methods)
    width = 0.85 / n_methods
    
    for i, method_model in enumerate(unique_methods):
        method_data = perf_df[perf_df["method_model"] == method_model]
        
        # Get method and model for color
        base_method, model = method_info.get(method_model, ("agent", "Unknown"))
        color = _get_bar_color(base_method, model)
        hatch = _get_bar_hatch(model) if base_method not in ["base_a", "base_b"] else ""
        
        values = []
        errors_low = []
        errors_high = []
        
        for stratum in strata:
            row = method_data[method_data["stratum"] == stratum]
            if not row.empty:
                val = row["value"].iloc[0]
                values.append(val * 100 if metric == "exact_match" else val)
                errors_low.append((val - row["ci_low"].iloc[0]) * 100 if metric == "exact_match" else (val - row["ci_low"].iloc[0]))
                errors_high.append((row["ci_high"].iloc[0] - val) * 100 if metric == "exact_match" else (row["ci_high"].iloc[0] - val))
            else:
                values.append(0)
                errors_low.append(0)
                errors_high.append(0)
        
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width * 0.88,
            label=method_model,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )
        
        # Add error bars
        ax.errorbar(
            x + offset, values,
            yerr=[errors_low, errors_high],
            fmt="none", color="black", capsize=2, capthick=0.8, linewidth=0.8,
        )
        
        # Add value callouts on top of bars
        for j, (bar, val, err_high) in enumerate(zip(bars, values, errors_high)):
            if val > 0:
                # Position text above error bar
                y_pos = val + err_high + 0.5
                ax.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    rotation=90,
                )
    
    # Calculate y-axis limit for proper spacing (extra room for labels)
    max_val = max(perf_df["value"].max() * 100 if metric == "exact_match" else perf_df["value"].max(), 1)
    ax.set_ylim(0, max_val * 1.35)
    
    metric_label = "Exact Match (%)" if metric == "exact_match" else "Similarity"
    ax.set_xlabel("Difficulty", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=13, fontweight="bold")
    ax.set_title(f"Method Performance by Difficulty", fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strata], fontsize=12, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)
    
    # Create legend with better layout
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        title="Method",
        title_fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def render_method_comparison_by_project_size(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render grouped bar chart comparing methods across project sizes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        Metric to plot
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save figure
    show : bool
        Display figure
        
    Returns
    -------
    plt.Figure
    """
    perf_df = compute_method_performance_by_stratum(df, "project_size", metric, config)
    
    if perf_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig
    
    # Create combined method-model labels
    perf_df["method_model"] = perf_df.apply(
        lambda r: r["method_display"] if r["model"] == "Baseline" 
        else f"{r['model']} ({r['method_display']})", axis=1
    )
    
    # Store original model for color lookup
    perf_df["model_orig"] = perf_df["model"]
    
    # Order project sizes
    size_order = ["small", "medium", "large", "huge"]
    perf_df["stratum"] = pd.Categorical(perf_df["stratum"], categories=size_order, ordered=True)
    perf_df = perf_df.sort_values("stratum")
    
    # Get unique methods (in order): baselines first, then models grouped
    unique_methods = []
    method_info = {}  # Store method and model for color lookup
    
    # First add baselines
    for method in ["base_a", "base_b"]:
        method_rows = perf_df[perf_df["method"] == method]
        for _, row in method_rows.drop_duplicates("method_model").iterrows():
            if row["method_model"] not in unique_methods:
                unique_methods.append(row["method_model"])
                method_info[row["method_model"]] = (method, row["model_orig"])
    
    # Then add models: group by model, then single/multi within each
    model_list = perf_df[~perf_df["method"].isin(["base_a", "base_b"])]["model"].unique()
    for model in sorted(model_list):
        for method in ["agent", "bypass7"]:
            method_rows = perf_df[(perf_df["method"] == method) & (perf_df["model"] == model)]
            for _, row in method_rows.drop_duplicates("method_model").iterrows():
                if row["method_model"] not in unique_methods:
                    unique_methods.append(row["method_model"])
                    method_info[row["method_model"]] = (method, row["model_orig"])
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    strata = [s for s in size_order if s in perf_df["stratum"].values]
    x = np.arange(len(strata))
    n_methods = len(unique_methods)
    width = 0.85 / n_methods
    
    for i, method_model in enumerate(unique_methods):
        method_data = perf_df[perf_df["method_model"] == method_model]
        
        # Get method and model for color
        base_method, model = method_info.get(method_model, ("agent", "Unknown"))
        color = _get_bar_color(base_method, model)
        hatch = _get_bar_hatch(model) if base_method not in ["base_a", "base_b"] else ""
        
        values = []
        errors_low = []
        errors_high = []
        
        for stratum in strata:
            row = method_data[method_data["stratum"] == stratum]
            if not row.empty:
                val = row["value"].iloc[0]
                values.append(val * 100 if metric == "exact_match" else val)
                errors_low.append((val - row["ci_low"].iloc[0]) * 100 if metric == "exact_match" else (val - row["ci_low"].iloc[0]))
                errors_high.append((row["ci_high"].iloc[0] - val) * 100 if metric == "exact_match" else (row["ci_high"].iloc[0] - val))
            else:
                values.append(0)
                errors_low.append(0)
                errors_high.append(0)
        
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width * 0.88,
            label=method_model,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )
        
        # Add error bars
        ax.errorbar(
            x + offset, values,
            yerr=[errors_low, errors_high],
            fmt="none", color="black", capsize=2, capthick=0.8, linewidth=0.8,
        )
        
        # Add value callouts on top of bars
        for j, (bar, val, err_high) in enumerate(zip(bars, values, errors_high)):
            if val > 0:
                # Position text above error bar
                y_pos = val + err_high + 0.5
                ax.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    rotation=90,
                )
    
    # Calculate y-axis limit for proper spacing (extra room for labels)
    max_val = max(perf_df["value"].max() * 100 if metric == "exact_match" else perf_df["value"].max(), 1)
    ax.set_ylim(0, max_val * 1.35)
    
    metric_label = "Exact Match (%)" if metric == "exact_match" else "Similarity"
    ax.set_xlabel("Project Size", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=13, fontweight="bold")
    ax.set_title(f"Method Performance by Project Size", fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strata], fontsize=12, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)
    
    # Create legend with better layout
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        title="Method",
        title_fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def render_method_comparison_heatmap(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render heatmap showing method performance across difficulty × project size.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        Metric to plot
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save figure
    show : bool
        Display figure
        
    Returns
    -------
    plt.Figure
    """
    work = df.copy()
    
    # Coerce exact_match
    if metric == "exact_match" and metric in work.columns:
        work[metric] = _coerce_bool_metric(work[metric])
    
    # Get unique methods with models - order: baselines first, then models grouped
    methods = []
    
    # First add baselines
    for method in ["base_a", "base_b"]:
        method_df = work[work["eval_method"] == method]
        if not method_df.empty:
            methods.append((method, "Baseline", METHOD_DISPLAY_NAMES[method], "#888888"))
    
    # Then add models grouped by model name
    model_methods = []
    for method in ["agent", "bypass7"]:
        method_df = work[work["eval_method"] == method]
        for model in sorted(method_df["model_name"].dropna().unique()):
            short_model = get_short_model_name(model)
            color = _get_bar_color(method, model)
            model_methods.append((method, model, f"{short_model} ({METHOD_DISPLAY_NAMES[method]})", color))
    
    # Sort model_methods to group by model
    model_methods.sort(key=lambda x: (get_short_model_name(x[1]), x[0]))
    methods.extend(model_methods)
    
    if not methods:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    # Reversed so easy is at bottom, hard at top in heatmap
    difficulty_order_display = ["hard", "medium", "easy"]  # Top to bottom
    size_order = ["small", "medium", "large", "huge"]  # Left to right
    
    # Create multi-panel figure with better layout
    n_methods = len(methods)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    if n_methods == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    # Determine global vmax for consistent color scale
    all_values = []
    for method, model, label, color in methods:
        if method in ["base_a", "base_b"]:
            method_df = work[work["eval_method"] == method]
        else:
            method_df = work[(work["eval_method"] == method) & (work["model_name"] == model)]
        if metric in method_df.columns:
            vals = pd.to_numeric(method_df[metric], errors="coerce").dropna()
            if len(vals) > 0:
                all_values.extend((vals * 100 if metric == "exact_match" else vals).tolist())
    
    vmax = max(all_values) * 1.1 if all_values else (50 if metric == "exact_match" else 1.0)
    
    for idx, (method, model, label, title_color) in enumerate(methods):
        ax = axes_flat[idx]
        
        if method in ["base_a", "base_b"]:
            method_df = work[work["eval_method"] == method]
        else:
            method_df = work[(work["eval_method"] == method) & (work["model_name"] == model)]
        
        # Compute pivot table
        pivot_data = []
        for diff in difficulty_order_display:
            row_data = []
            for size in size_order:
                subset = method_df[
                    (method_df["difficulty"].astype(str).str.lower() == diff) &
                    (method_df["project_size"].astype(str).str.lower() == size)
                ]
                if not subset.empty and metric in subset.columns:
                    val = pd.to_numeric(subset[metric], errors="coerce").mean()
                    row_data.append(val * 100 if metric == "exact_match" else val)
                else:
                    row_data.append(np.nan)
            pivot_data.append(row_data)
        
        pivot_df = pd.DataFrame(
            pivot_data,
            index=[d.capitalize() for d in difficulty_order_display],
            columns=[s.capitalize() for s in size_order],
        )
        
        # Draw heatmap with consistent scale
        sns.heatmap(
            pivot_df, ax=ax, annot=True, fmt=".1f",
            cmap="YlGnBu", vmin=0, vmax=vmax,
            cbar_kws={"label": "EM %" if metric == "exact_match" else "Similarity", "shrink": 0.8},
            annot_kws={"fontsize": 9, "fontweight": "bold"},
        )
        
        # Color-coded title based on model
        ax.set_title(label, fontsize=11, fontweight="bold", color=title_color, pad=8)
        ax.set_xlabel("Project Size", fontsize=10, fontweight="bold")
        ax.set_ylabel("Difficulty", fontsize=10, fontweight="bold")
        ax.tick_params(axis='both', labelsize=9)
    
    # Hide unused
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].axis("off")
    
    metric_label = "Exact Match" if metric == "exact_match" else "Similarity"
    fig.suptitle(
        f"Performance by Difficulty × Project Size ({metric_label})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def export_method_comparison_table(
    df: pd.DataFrame,
    config: RQ2Config = DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Export a comprehensive table comparing all methods across strata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save CSV
        
    Returns
    -------
    pd.DataFrame
        Combined performance table
    """
    all_results = []
    
    for stratum_col in ["difficulty", "project_size"]:
        for metric in ["exact_match", "similarity"]:
            try:
                perf_df = compute_method_performance_by_stratum(df, stratum_col, metric, config)
                perf_df["stratum_type"] = stratum_col
                all_results.append(perf_df)
            except Exception as e:
                print(f"Warning: Could not compute {stratum_col} x {metric}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    cols = ["stratum_type", "stratum", "method", "method_display", "model", "metric", "value", "n", "ci_low", "ci_high"]
    combined = combined[[c for c in cols if c in combined.columns]]
    
    if output_path is not None:
        combined.to_csv(output_path, index=False)
    
    return combined
