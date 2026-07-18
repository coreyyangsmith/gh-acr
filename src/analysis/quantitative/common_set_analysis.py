"""Common-set analysis: non-label metrics across all IDs where all 3 models have outputs.

This script computes scenario-metadata-vs-performance correlations using the
full intersection of IDs that have results for ALL models (GPT-5-nano, Qwen3-32B,
LLaMA-3.1-8B).  It does NOT require case folders—it works entirely from:

1. ``results CSV`` – per-file performance rows (exact_match, similarity, …)
2. ``dataset CSV`` – scenario metadata (conflict counts, repo stats, difficulty, …)

Statistics produced
-------------------
- **Scenario metadata → Performance correlations** (Spearman / Pearson) for each
  model and aggregated, separately for agent and bypass methods
- **Performance by difficulty** and **performance by project_size** breakdowns
- **Per-model comparison** of performance on the common set
- **Scenario metadata distribution** across the common set
- **Token / cost analysis** per model

Usage::

    python -m src.analysis.quantitative.common_set_analysis \\
        --results-csv data/2026_01_results_final.csv \\
        --dataset-csv data/git_good_bench_merge_commits_all.csv \\
        --output-dir results/rq_quantitative_common
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import warnings
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

# ── Constants ────────────────────────────────────────────────────────────

PERFORMANCE_METRICS = ["exact_match", "similarity", "bleu3", "rouge_l"]
EVAL_METHODS = ["agent", "bypass7"]

SCENARIO_METRICS = [
    "n_conflict_files",
    "n_total_conflicts",
    "repo_commits",
    "repo_code_lines",
    "repo_contributors",
]

MODEL_COLORS = {
    "openai/gpt-5-nano": "#1f77b4",
    "groq:qwen/qwen3-32b": "#ff7f0e",
    "local:meta-llama/Llama-3.1-8B-Instruct": "#2ca02c",
}

MODEL_SHORT_NAMES = {
    "openai/gpt-5-nano": "GPT-5-nano",
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
}

METHOD_DISPLAY = {"agent": "Agent (Single)", "bypass7": "Bypass (Multi)"}
DIFF_ORDER = ["easy", "medium", "hard"]
SIZE_ORDER = ["small", "medium", "large", "huge"]


# ── CLI dataclass ────────────────────────────────────────────────────────


@dataclass
class CommonSetFlags:
    """CLI arguments for common-set analysis."""

    results_csv: Path
    dataset_csv: Path
    output_dir: Path = Path("results/rq_quantitative_common")
    dpi: int = 150


# ── Helpers ──────────────────────────────────────────────────────────────


def _coerce_exact_match(series: pd.Series) -> pd.Series:
    """Coerce exact_match to numeric 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "1.0", "yes"])
        .astype(float)
    )


def _significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _save_and_close(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ── Step 1: Load and find common IDs ─────────────────────────────────────


def find_common_ids(results_df: pd.DataFrame) -> set[str]:
    """Return IDs present for all models in agent+bypass results."""
    ab = results_df[results_df["eval_method"].isin(EVAL_METHODS)].copy()
    ab["id"] = ab["id"].astype(str)

    model_ids: dict[str, set[str]] = {}
    for model in ab["model_name"].dropna().unique():
        model_ids[model] = set(ab[ab["model_name"] == model]["id"].unique())
        logger.info(f"  {MODEL_SHORT_NAMES.get(model, model)}: {len(model_ids[model])} IDs")

    if not model_ids:
        return set()

    common = set.intersection(*model_ids.values())
    logger.info(f"  Common across all {len(model_ids)} models: {len(common)} IDs")
    return common


# ── Step 2: Load scenario metadata ──────────────────────────────────────


def load_scenario_metadata(dataset_csv: Path) -> pd.DataFrame:
    """Load scenario-level metadata from the benchmark dataset CSV.

    The dataset CSV's first column (``Unnamed: 0``) contains the numeric
    scenario IDs that match the results CSV ``id`` column.  The ``id``
    column in the dataset CSV is a repo-name string (not the numeric ID).
    """
    logger.info(f"Loading scenario metadata from {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    rows = []
    for _, row in df.iterrows():
        entry: dict = {}

        # The numeric ID is in the first column (original pandas index)
        if "Unnamed: 0" in df.columns:
            entry["id"] = str(row["Unnamed: 0"])
        elif "id" in df.columns:
            entry["id"] = str(row["id"])
        else:
            entry["id"] = str(row.name)

        # Parse scenario JSON
        if "scenario" in df.columns:
            try:
                scenario = ast.literal_eval(str(row["scenario"]))
                entry["n_conflict_files"] = scenario.get(
                    "number_of_files_with_merge_conflict", 0
                )
                entry["n_total_conflicts"] = scenario.get(
                    "total_number_of_merge_conflicts", 0
                )
            except (ValueError, SyntaxError):
                entry["n_conflict_files"] = 0
                entry["n_total_conflicts"] = 0

        # Repo-level metadata
        for src_col, dst_col in [
            ("commits", "repo_commits"),
            ("code_lines", "repo_code_lines"),
            ("contributors", "repo_contributors"),
        ]:
            if src_col in df.columns:
                try:
                    entry[dst_col] = int(row[src_col])
                except (ValueError, TypeError):
                    entry[dst_col] = 0

        # Categorical columns
        for col in ["difficulty", "project_size"]:
            if col in df.columns:
                entry[col] = str(row[col])

        rows.append(entry)

    return pd.DataFrame(rows)


# ── Step 3: Build the common-set DataFrame ───────────────────────────────


def build_common_performance_df(
    results_df: pd.DataFrame,
    common_ids: set[str],
) -> pd.DataFrame:
    """Build per-(id, model, method) aggregated performance DataFrame."""
    df = results_df.copy()
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(common_ids)]
    df = df[df["eval_method"].isin(EVAL_METHODS)]

    if "exact_match" in df.columns:
        df["exact_match"] = _coerce_exact_match(df["exact_match"])

    # Aggregate multi-file scenarios to instance level
    group_cols = ["id", "model_name", "eval_method"]
    agg_dict = {}
    for m in PERFORMANCE_METRICS:
        if m in df.columns:
            agg_dict[m] = "min" if m == "exact_match" else "mean"

    for col in ["difficulty", "project_size"]:
        if col in df.columns:
            agg_dict[col] = "first"

    # Token/cost columns
    for col in ["tokens_total", "tokens_in", "tokens_out", "total_cost", "processing_time_s"]:
        if col in df.columns:
            agg_dict[col] = "sum"

    agg_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    logger.info(f"  Built common performance DF: {len(agg_df)} rows, {len(common_ids)} IDs")
    return agg_df


# ── Step 4: Scenario-metadata → Performance correlations ────────────────


def compute_scenario_performance_correlations(
    perf_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    min_samples: int = 20,
) -> pd.DataFrame:
    """Correlate scenario metadata with performance, per model and method."""
    perf_df = perf_df.copy()
    scenario_df = scenario_df.copy()
    perf_df["id"] = perf_df["id"].astype(str)
    scenario_df["id"] = scenario_df["id"].astype(str)

    merged = perf_df.merge(scenario_df, on="id", how="left", suffixes=("", "_scen"))

    rows = []
    # Per model + method
    for model in merged["model_name"].dropna().unique():
        short = MODEL_SHORT_NAMES.get(model, model)
        for method in EVAL_METHODS:
            subset = merged[
                (merged["model_name"] == model) & (merged["eval_method"] == method)
            ]
            if len(subset) < min_samples:
                continue

            for scen_col in SCENARIO_METRICS:
                if scen_col not in subset.columns:
                    continue
                for perf_col in PERFORMANCE_METRICS:
                    if perf_col not in subset.columns:
                        continue
                    valid = subset[[scen_col, perf_col]].dropna()
                    if len(valid) < min_samples:
                        continue
                    try:
                        sp_r, sp_p = stats.spearmanr(valid[scen_col], valid[perf_col])
                        pe_r, pe_p = stats.pearsonr(valid[scen_col], valid[perf_col])
                        rows.append({
                            "model": short,
                            "method": METHOD_DISPLAY.get(method, method),
                            "scenario_metric": scen_col,
                            "performance_metric": perf_col,
                            "n_samples": len(valid),
                            "spearman_r": sp_r,
                            "spearman_p": sp_p,
                            "pearson_r": pe_r,
                            "pearson_p": pe_p,
                        })
                    except Exception:
                        pass

    # Aggregated across all models (average performance per ID+method)
    for method in EVAL_METHODS:
        method_df = merged[merged["eval_method"] == method]
        if method_df.empty:
            continue
        agg_cols = {m: "mean" for m in PERFORMANCE_METRICS if m in method_df.columns}
        for sc in SCENARIO_METRICS:
            if sc in method_df.columns:
                agg_cols[sc] = "first"
        agg = method_df.groupby("id", as_index=False).agg(agg_cols)

        for scen_col in SCENARIO_METRICS:
            if scen_col not in agg.columns:
                continue
            for perf_col in PERFORMANCE_METRICS:
                if perf_col not in agg.columns:
                    continue
                valid = agg[[scen_col, perf_col]].dropna()
                if len(valid) < min_samples:
                    continue
                try:
                    sp_r, sp_p = stats.spearmanr(valid[scen_col], valid[perf_col])
                    pe_r, pe_p = stats.pearsonr(valid[scen_col], valid[perf_col])
                    rows.append({
                        "model": "All Models (avg)",
                        "method": METHOD_DISPLAY.get(method, method),
                        "scenario_metric": scen_col,
                        "performance_metric": perf_col,
                        "n_samples": len(valid),
                        "spearman_r": sp_r,
                        "spearman_p": sp_p,
                        "pearson_r": pe_r,
                        "pearson_p": pe_p,
                    })
                except Exception:
                    pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_spearman"] = result["spearman_r"].abs()
        result = result.sort_values("abs_spearman", ascending=False)
        result = result.drop(columns=["abs_spearman"])

    logger.info(f"  Computed {len(result)} scenario-performance correlation pairs")
    return result


# ── Step 5: Performance by difficulty / project_size ─────────────────────


def compute_performance_by_category(
    perf_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean performance broken down by difficulty and project_size."""
    dfs = []
    for cat_col, order in [("difficulty", DIFF_ORDER), ("project_size", SIZE_ORDER)]:
        if cat_col not in perf_df.columns:
            dfs.append(pd.DataFrame())
            continue

        rows = []
        for model in perf_df["model_name"].dropna().unique():
            short = MODEL_SHORT_NAMES.get(model, model)
            for method in EVAL_METHODS:
                subset = perf_df[
                    (perf_df["model_name"] == model) & (perf_df["eval_method"] == method)
                ]
                for cat_val in order:
                    cat_sub = subset[subset[cat_col] == cat_val]
                    if cat_sub.empty:
                        continue
                    row = {
                        "model": short,
                        "method": METHOD_DISPLAY.get(method, method),
                        cat_col: cat_val,
                        "n_samples": len(cat_sub),
                    }
                    for m in PERFORMANCE_METRICS:
                        if m in cat_sub.columns:
                            row[f"{m}_mean"] = cat_sub[m].mean()
                            row[f"{m}_std"] = cat_sub[m].std()
                    rows.append(row)
        dfs.append(pd.DataFrame(rows))

    return dfs[0], dfs[1]


# ── Step 6: Per-model comparison ─────────────────────────────────────────


def compute_model_comparison(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table: mean performance per model+method on the common set."""
    rows = []
    for model in perf_df["model_name"].dropna().unique():
        short = MODEL_SHORT_NAMES.get(model, model)
        for method in EVAL_METHODS:
            subset = perf_df[
                (perf_df["model_name"] == model) & (perf_df["eval_method"] == method)
            ]
            if subset.empty:
                continue
            row = {
                "model": short,
                "method": METHOD_DISPLAY.get(method, method),
                "n_samples": len(subset),
            }
            for m in PERFORMANCE_METRICS:
                if m in subset.columns:
                    row[f"{m}_mean"] = subset[m].mean()
                    row[f"{m}_std"] = subset[m].std()
                    row[f"{m}_median"] = subset[m].median()
            for col in ["tokens_total", "total_cost", "processing_time_s"]:
                if col in subset.columns:
                    row[f"{col}_mean"] = subset[col].mean()
                    row[f"{col}_sum"] = subset[col].sum()
            rows.append(row)
    return pd.DataFrame(rows)


# ── Step 7: Scenario metadata distribution ───────────────────────────────


def compute_scenario_distribution(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for scenario metrics on the common set."""
    rows = []
    for col in SCENARIO_METRICS:
        if col not in scenario_df.columns:
            continue
        vals = scenario_df[col].dropna()
        rows.append({
            "metric": col,
            "n": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "median": vals.median(),
            "min": vals.min(),
            "max": vals.max(),
            "q25": vals.quantile(0.25),
            "q75": vals.quantile(0.75),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════


def plot_scenario_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: scenario metadata vs performance (Spearman r) for each model+method."""
    if corr_df.empty:
        return

    # One heatmap per method
    for method_label in corr_df["method"].unique():
        sub = corr_df[corr_df["method"] == method_label]

        # Build a multi-index pivot: rows = scenario_metric, columns = (model, perf_metric)
        sub = sub.copy()
        sub["col_key"] = sub["model"] + "\n" + sub["performance_metric"]

        pivot = sub.pivot_table(
            index="scenario_metric",
            columns="col_key",
            values="spearman_r",
        )
        p_pivot = sub.pivot_table(
            index="scenario_metric",
            columns="col_key",
            values="spearman_p",
        )

        if pivot.empty:
            continue

        # Annotation with significance stars
        annot = pivot.copy().astype(str)
        for row in pivot.index:
            for col in pivot.columns:
                r_val = pivot.loc[row, col]
                p_val = p_pivot.loc[row, col] if row in p_pivot.index and col in p_pivot.columns else 1.0
                if pd.isna(r_val):
                    annot.loc[row, col] = ""
                else:
                    annot.loc[row, col] = f"{r_val:.2f}{_significance(p_val)}"

        fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 1.2), 6))
        sns.heatmap(
            pivot,
            annot=annot,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            ax=ax,
            linewidths=0.5,
        )
        method_clean = method_label.replace(" ", "_").replace("(", "").replace(")", "")
        ax.set_title(
            f"Scenario Metadata vs Performance – {method_label}\n(Spearman r, n={corr_df['n_samples'].max():.0f} common IDs)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()

        path = output_path.parent / f"{output_path.stem}_{method_clean}{output_path.suffix}"
        _save_and_close(fig, path, dpi)


def plot_performance_by_difficulty(
    diff_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Grouped bar chart: performance by difficulty, faceted by method."""
    if diff_df.empty or "difficulty" not in diff_df.columns:
        return

    metrics_to_plot = [m for m in PERFORMANCE_METRICS if f"{m}_mean" in diff_df.columns]
    if not metrics_to_plot:
        return

    for method_label in diff_df["method"].unique():
        sub = diff_df[diff_df["method"] == method_label].copy()
        if sub.empty:
            continue

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            for i, model in enumerate(sub["model"].unique()):
                model_sub = sub[sub["model"] == model]
                x = np.arange(len(DIFF_ORDER))
                width = 0.25
                vals = []
                errs = []
                for d in DIFF_ORDER:
                    row = model_sub[model_sub["difficulty"] == d]
                    vals.append(row[mean_col].values[0] if len(row) > 0 else 0)
                    errs.append(row[std_col].values[0] if len(row) > 0 and std_col in row.columns else 0)

                color = list(MODEL_COLORS.values())[i % len(MODEL_COLORS)]
                ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

            ax.set_xticks(x + width)
            ax.set_xticklabels(DIFF_ORDER)
            ax.set_xlabel("Difficulty")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            if idx == 0:
                ax.legend(fontsize=8)

        method_clean = method_label.replace(" ", "_").replace("(", "").replace(")", "")
        plt.suptitle(
            f"Performance by Difficulty – {method_label}\n(Common Set, n={sub['n_samples'].sum() // len(sub['model'].unique()) // len(DIFF_ORDER):.0f}/difficulty)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        path = output_path.parent / f"{output_path.stem}_{method_clean}{output_path.suffix}"
        _save_and_close(fig, path, dpi)


def plot_model_comparison(
    model_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Bar chart: per-model performance on common set, agent vs bypass side-by-side."""
    if model_df.empty:
        return

    metrics_to_plot = [m for m in PERFORMANCE_METRICS if f"{m}_mean" in model_df.columns]
    if not metrics_to_plot:
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    models = model_df["model"].unique()
    methods = model_df["method"].unique()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        mean_col = f"{metric}_mean"

        x = np.arange(len(models))
        total_bars = len(methods)
        width = 0.8 / total_bars

        for j, method in enumerate(methods):
            method_sub = model_df[model_df["method"] == method]
            vals = []
            for m in models:
                row = method_sub[method_sub["model"] == m]
                vals.append(row[mean_col].values[0] if len(row) > 0 else 0)

            color = "#d95f02" if "Agent" in method else "#1b9e77"
            ax.bar(x + j * width - (total_bars - 1) * width / 2, vals, width,
                   label=method if idx == 0 else "", color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=15)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())

    axes[0].legend(fontsize=8)
    n_common = model_df["n_samples"].max()
    plt.suptitle(
        f"Model Comparison on Common Set (n={n_common:.0f} per model-method)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_and_close(fig, output_path, dpi)


def plot_scenario_distributions(
    scenario_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Histograms of scenario metadata on the common set."""
    cols = [c for c in SCENARIO_METRICS if c in scenario_df.columns]
    if not cols:
        return

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(cols):
        ax = axes[idx]
        vals = scenario_df[col].dropna()
        ax.hist(vals, bins=30, color="#2c7fb8", edgecolor="white", alpha=0.85)
        ax.axvline(vals.median(), color="red", linestyle="--",
                   label=f"Median={vals.median():.0f}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Scenario Metadata Distribution (Common Set, n={len(scenario_df)})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_and_close(fig, output_path, dpi)


def plot_scenario_vs_performance_scatter(
    perf_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Scatter plots: key scenario metrics vs exact_match rate, per model."""
    perf_df = perf_df.copy()
    scenario_df = scenario_df.copy()
    perf_df["id"] = perf_df["id"].astype(str)
    scenario_df["id"] = scenario_df["id"].astype(str)

    merged = perf_df.merge(scenario_df, on="id", how="left", suffixes=("", "_scen"))

    scen_cols = [c for c in ["n_conflict_files", "n_total_conflicts", "repo_code_lines"]
                 if c in merged.columns]
    if not scen_cols or "exact_match" not in merged.columns:
        return

    for method in EVAL_METHODS:
        method_sub = merged[merged["eval_method"] == method]
        if method_sub.empty:
            continue

        n_scen = len(scen_cols)
        n_models = len(method_sub["model_name"].dropna().unique())
        fig, axes = plt.subplots(n_scen, n_models, figsize=(5 * n_models, 4 * n_scen))
        if n_scen == 1 and n_models == 1:
            axes = np.array([[axes]])
        elif n_scen == 1:
            axes = axes.reshape(1, -1)
        elif n_models == 1:
            axes = axes.reshape(-1, 1)

        for j, model in enumerate(sorted(method_sub["model_name"].dropna().unique())):
            model_sub = method_sub[method_sub["model_name"] == model]
            short = MODEL_SHORT_NAMES.get(model, model)
            color = MODEL_COLORS.get(model, "#333333")

            for i, scen_col in enumerate(scen_cols):
                ax = axes[i, j]

                # Bin by scenario metric and compute exact_match rate
                valid = model_sub[[scen_col, "exact_match"]].dropna()
                if len(valid) < 20:
                    ax.text(0.5, 0.5, "Insufficient data",
                            ha="center", va="center", transform=ax.transAxes)
                    continue

                ax.scatter(valid[scen_col], valid["exact_match"],
                           alpha=0.15, s=15, color=color, edgecolor="none")

                # Trend line
                try:
                    z = np.polyfit(valid[scen_col], valid["exact_match"], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid[scen_col].min(), valid[scen_col].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

                    r, pval = stats.spearmanr(valid[scen_col], valid["exact_match"])
                    ax.annotate(
                        f"r={r:.3f}{_significance(pval)}\nn={len(valid)}",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                except Exception:
                    pass

                ax.set_xlabel(scen_col.replace("_", " ").title())
                if j == 0:
                    ax.set_ylabel("Exact Match")
                if i == 0:
                    ax.set_title(short, fontsize=11, fontweight="bold")

        method_display = METHOD_DISPLAY.get(method, method)
        method_clean = method_display.replace(" ", "_").replace("(", "").replace(")", "")
        plt.suptitle(
            f"Scenario Metrics vs Exact Match – {method_display}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        path = output_path.parent / f"{output_path.stem}_{method_clean}{output_path.suffix}"
        _save_and_close(fig, path, dpi)


def plot_difficulty_distribution(
    scenario_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Bar chart of difficulty and project_size distribution on common set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (col, order) in enumerate([("difficulty", DIFF_ORDER), ("project_size", SIZE_ORDER)]):
        ax = axes[idx]
        if col not in scenario_df.columns:
            ax.text(0.5, 0.5, f"No {col} data", ha="center", va="center")
            continue
        counts = scenario_df[col].value_counts()
        vals = [counts.get(v, 0) for v in order]
        colors = ["#2ca02c", "#ff7f0e", "#d62728"] if col == "difficulty" else ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        ax.bar(order, vals, color=colors[:len(order)], alpha=0.85, edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, v + 5, str(v), ha="center", fontsize=10)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(f"{col.replace('_', ' ').title()} Distribution")

    plt.suptitle(
        f"Common Set Composition (n={len(scenario_df)})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_and_close(fig, output_path, dpi)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def run_common_set_analysis(
    results_csv: str | Path,
    dataset_csv: str | Path,
    output_dir: str | Path = "results/rq_quantitative_common",
    dpi: int = 150,
) -> dict[str, Path]:
    """Run the full common-set analysis pipeline.

    Parameters
    ----------
    results_csv : Path
        Path to the results CSV (all models)
    dataset_csv : Path
        Path to the GitGoodBench dataset CSV
    output_dir : Path
        Output directory
    dpi : int
        Figure DPI

    Returns
    -------
    dict[str, Path]
        Mapping of output names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # ── Step 1: Load results and find common IDs ──
    logger.info("=" * 60)
    logger.info("STEP 1: Finding common IDs across all models")
    logger.info("=" * 60)

    results_df = pd.read_csv(results_csv)
    logger.info(f"  Loaded {len(results_df)} rows from {results_csv}")

    common_ids = find_common_ids(results_df)
    if not common_ids:
        logger.error("No common IDs found. Exiting.")
        return outputs

    # Save common IDs
    p = output_path / "common_ids.txt"
    with open(p, "w") as f:
        for sid in sorted(common_ids):
            f.write(sid + "\n")
    outputs["common_ids"] = p
    logger.info(f"  Saved {len(common_ids)} common IDs to {p}")

    # ── Step 2: Load scenario metadata ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Loading scenario metadata")
    logger.info("=" * 60)

    scenario_meta = load_scenario_metadata(Path(dataset_csv))
    scenario_meta["id"] = scenario_meta["id"].astype(str)
    scenario_common = scenario_meta[scenario_meta["id"].isin(common_ids)].copy()
    logger.info(f"  Scenario metadata for common set: {len(scenario_common)} rows")

    p = output_path / "common_scenario_metadata.csv"
    scenario_common.to_csv(p, index=False)
    outputs["scenario_metadata"] = p

    # ── Step 3: Build common performance DataFrame ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Building common performance DataFrame")
    logger.info("=" * 60)

    perf_df = build_common_performance_df(results_df, common_ids)

    p = output_path / "common_performance.csv"
    perf_df.to_csv(p, index=False)
    outputs["common_performance"] = p

    # ── Step 4: Scenario → Performance correlations ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Computing scenario → performance correlations")
    logger.info("=" * 60)

    corr_df = compute_scenario_performance_correlations(perf_df, scenario_common)
    if not corr_df.empty:
        p = output_path / "common_scenario_perf_correlation.csv"
        corr_df.to_csv(p, index=False)
        outputs["scenario_perf_correlation"] = p

    # ── Step 5: Performance by difficulty / project_size ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Performance by difficulty and project_size")
    logger.info("=" * 60)

    diff_df, size_df = compute_performance_by_category(perf_df)
    if not diff_df.empty:
        p = output_path / "common_perf_by_difficulty.csv"
        diff_df.to_csv(p, index=False)
        outputs["perf_by_difficulty"] = p
    if not size_df.empty:
        p = output_path / "common_perf_by_project_size.csv"
        size_df.to_csv(p, index=False)
        outputs["perf_by_project_size"] = p

    # ── Step 6: Model comparison ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Computing model comparison")
    logger.info("=" * 60)

    model_df = compute_model_comparison(perf_df)
    if not model_df.empty:
        p = output_path / "common_model_comparison.csv"
        model_df.to_csv(p, index=False)
        outputs["model_comparison"] = p
        logger.info(f"  Model comparison:\n{model_df.to_string()}")

    # ── Step 7: Scenario distributions ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7: Scenario distributions")
    logger.info("=" * 60)

    scenario_dist = compute_scenario_distribution(scenario_common)
    if not scenario_dist.empty:
        p = output_path / "common_scenario_distribution.csv"
        scenario_dist.to_csv(p, index=False)
        outputs["scenario_distribution"] = p
        logger.info(f"  Scenario distribution:\n{scenario_dist.to_string()}")

    # ── Step 8: Generate plots ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 8: Generating plots")
    logger.info("=" * 60)

    # 8a. Correlation heatmap
    if not corr_df.empty:
        plot_scenario_correlation_heatmap(
            corr_df,
            output_path / "common_correlation_heatmap.png",
            dpi,
        )

    # 8b. Performance by difficulty
    if not diff_df.empty:
        plot_performance_by_difficulty(
            diff_df,
            output_path / "common_perf_by_difficulty.png",
            dpi,
        )

    # 8c. Model comparison
    if not model_df.empty:
        plot_model_comparison(
            model_df,
            output_path / "common_model_comparison.png",
            dpi,
        )

    # 8d. Scenario distributions
    plot_scenario_distributions(
        scenario_common,
        output_path / "common_scenario_distributions.png",
        dpi,
    )

    # 8e. Scenario vs performance scatter
    plot_scenario_vs_performance_scatter(
        perf_df,
        scenario_common,
        output_path / "common_scenario_vs_perf.png",
        dpi,
    )

    # 8f. Difficulty / project_size distribution
    plot_difficulty_distribution(
        scenario_common,
        output_path / "common_difficulty_distribution.png",
        dpi,
    )

    # ── DONE ──
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"Common-set analysis complete. Generated {len(outputs)} outputs in {output_path}"
    )
    logger.info("=" * 60)

    return outputs


# ── CLI entry point ──────────────────────────────────────────────────────


def main() -> None:
    import tyro
    flags = tyro.cli(CommonSetFlags)
    run_common_set_analysis(
        results_csv=flags.results_csv,
        dataset_csv=flags.dataset_csv,
        output_dir=flags.output_dir,
        dpi=flags.dpi,
    )


if __name__ == "__main__":
    main()
