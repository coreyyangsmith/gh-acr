"""Generate publication-ready figures for RQ1 and RQ2.

Figures
-------
RQ1/
  Figure_A_method_comparison.pdf   – Agent vs Bypass vs Base-A vs Base-B
  Figure_B_bypass_advantage.pdf    – Bypass-minus-Agent distribution per model
  Figure_C_decision_outcomes.pdf   – A/B/MIX selection rates per model

RQ2/
  Figure_D_advantage_by_buckets.pdf – EM + similarity advantage by difficulty & conflict count
  Figure_E_repo_complexity.pdf      – Scatter: similarity advantage vs repo commits/LOC

Usage::

    python -m src.results.quantitative.paper_figures \\
        --results-csv data/2026_01_results_final.csv \\
        --dataset-csv data/git_good_bench_merge_commits_all.csv \\
        --output-dir results/paper_ready
"""

from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_palette("colorblind")

# ── Constants ────────────────────────────────────────────────────────────

MODELS_FULL = [
    "openai/gpt-5-nano",
    "groq:qwen/qwen3-32b",
    "local:meta-llama/Llama-3.1-8B-Instruct",
]

MODEL_SHORT = {
    "openai/gpt-5-nano": "GPT-5-nano",
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
}

MODEL_ORDER = ["GPT-5-nano", "Qwen3-32B", "LLaMA-3.1-8B"]

METHOD_ORDER = ["Base A", "Base B", "Agent", "Bypass"]
METHOD_COLORS = {
    "Base A": "#a6cee3",
    "Base B": "#b2df8a",
    "Agent": "#ff7f00",
    "Bypass": "#1f78b4",
}

EVAL_MAP = {
    "base_a": "Base A",
    "base_b": "Base B",
    "agent": "Agent",
    "bypass7": "Bypass",
}

PERF = ["exact_match", "similarity", "bleu3", "rouge_l"]
PERF_LABELS = {
    "exact_match": "Exact Match",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}

DIFF_ORDER = ["easy", "medium", "hard"]
SIZE_ORDER = ["small", "medium", "large", "huge"]
CONFLICT_BUCKETS = [(1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, 999, "11+")]


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def _coerce_em(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0.5).astype(float)
    return s.astype(str).str.lower().str.strip().isin(
        ["true", "1", "1.0", "yes"]
    ).astype(float)


def _save(fig, path):
    for ext in [".pdf", ".png"]:
        fig.savefig(str(path).replace(".pdf", ext))
    plt.close(fig)
    logger.info(f"  Saved {path.name}")


# ── Data loading ─────────────────────────────────────────────────────────

def _load_results(path):
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    if "exact_match" in df.columns:
        df["exact_match"] = _coerce_em(df["exact_match"])
    df["model"] = df["model_name"].map(MODEL_SHORT)
    df["method"] = df["eval_method"].map(EVAL_MAP)
    return df


def _load_scenario(path):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        e = {"id": str(r["Unnamed: 0"]) if "Unnamed: 0" in df.columns else str(r.name)}
        if "scenario" in df.columns:
            try:
                sc = ast.literal_eval(str(r["scenario"]))
                e["n_conflict_files"] = sc.get("number_of_files_with_merge_conflict", 0)
                e["n_total_conflicts"] = sc.get("total_number_of_merge_conflicts", 0)
            except (ValueError, SyntaxError):
                e["n_conflict_files"] = 0
                e["n_total_conflicts"] = 0
        for src, dst in [("commits", "repo_commits"), ("code_lines", "repo_code_lines"),
                         ("contributors", "repo_contributors")]:
            if src in df.columns:
                try: e[dst] = int(r[src])
                except: e[dst] = 0
        for c in ["difficulty", "project_size"]:
            if c in df.columns:
                e[c] = str(r[c])
        rows.append(e)
    return pd.DataFrame(rows)


def _common_ids(df):
    ab = df[df["eval_method"].isin(["agent", "bypass7"])]
    per_model = {m: set(ab[ab["model_name"] == m]["id"].unique()) for m in ab["model_name"].dropna().unique()}
    return set.intersection(*per_model.values()) if per_model else set()


def _instance_agg(df, common):
    """Aggregate per-file rows to per-instance (min EM, mean others)."""
    df = df[df["id"].isin(common)].copy()
    agg = {}
    for m in PERF:
        if m in df.columns:
            agg[m] = "min" if m == "exact_match" else "mean"
    for c in ["difficulty", "project_size"]:
        if c in df.columns:
            agg[c] = "first"
    return df.groupby(["id", "model_name", "eval_method", "model", "method"],
                       as_index=False).agg(agg)


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE A – Overall method comparison
# ═══════════════════════════════════════════════════════════════════════

def figure_a(df, common, out):
    """Grouped bars: 4 methods x 3 models, panels for EM + Similarity."""
    logger.info("Figure A: Method comparison")

    inst = _instance_agg(df, common)

    # Compute means
    rows = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            sub = inst[(inst["model"] == model) & (inst["method"] == method)]
            if sub.empty:
                continue
            row = {"model": model, "method": method, "n": len(sub)}
            for m in PERF:
                if m in sub.columns:
                    row[f"{m}_mean"] = sub[m].mean()
                    row[f"{m}_std"] = sub[m].std()
            rows.append(row)
    summary = pd.DataFrame(rows)

    metrics = [("exact_match", "Exact Match Rate"), ("similarity", "Similarity Score"),
               ("bleu3", "BLEU-3"), ("rouge_l", "ROUGE-L")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for idx, (metric, label) in enumerate(metrics):
        ax = axes.flatten()[idx]
        x = np.arange(len(MODEL_ORDER))
        n_methods = len(METHOD_ORDER)
        w = 0.18

        for j, method in enumerate(METHOD_ORDER):
            vals = []
            for model in MODEL_ORDER:
                row = summary[(summary["model"] == model) & (summary["method"] == method)]
                vals.append(row[f"{metric}_mean"].values[0] if len(row) > 0 else 0)
            offset = (j - (n_methods - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=method if idx == 0 else "",
                          color=METHOD_COLORS[method], edgecolor="white", linewidth=0.5)
            # Value labels on EM bars
            if metric == "exact_match":
                for bar, v in zip(bars, vals):
                    if v > 0.005:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                                f"{v:.1%}", ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        if metric == "exact_match":
            ax.set_ylim(0, max(0.2, ax.get_ylim()[1] * 1.15))
        else:
            ax.set_ylim(0, 1.05)

    axes[0, 0].legend(loc="upper left", framealpha=0.9, fontsize=8)

    fig.suptitle(
        f"Figure A: Method Comparison on Common Set (n = {len(common):,} scenarios)",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, out / "RQ1" / "Figure_A_method_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE B – Bypass advantage distribution
# ═══════════════════════════════════════════════════════════════════════

def figure_b(df, common, out, *, slim=False):
    """Violin + box: Bypass-minus-Agent per metric, faceted by model.

    Args:
        slim: If True, exclude BLEU-3 and ROUGE-L panels (show only
              Similarity and Exact Match).  Default False.
    """
    logger.info("Figure B: Bypass advantage distribution")

    inst = _instance_agg(df, common)
    agent = inst[inst["method"] == "Agent"].copy()
    bypass = inst[inst["method"] == "Bypass"].copy()
    merged = agent.merge(bypass, on=["id", "model"], suffixes=("_ag", "_by"))

    for m in PERF:
        ac, bc = f"{m}_ag", f"{m}_by"
        if ac in merged.columns and bc in merged.columns:
            merged[f"adv_{m}"] = pd.to_numeric(merged[bc], errors="coerce") - pd.to_numeric(merged[ac], errors="coerce")

    all_metrics = [("adv_similarity", "Similarity"), ("adv_bleu3", "BLEU-3"),
                   ("adv_rouge_l", "ROUGE-L"), ("adv_exact_match", "Exact Match")]
    if slim:
        metrics = [m for m in all_metrics if m[0] not in ("adv_bleu3", "adv_rouge_l")]
    else:
        metrics = all_metrics

    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        if col not in merged.columns:
            continue

        parts = ax.violinplot(
            [merged[merged["model"] == m][col].dropna().values for m in MODEL_ORDER],
            positions=range(len(MODEL_ORDER)),
            showmeans=True, showmedians=True, showextrema=False,
        )
        # Color violins
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.35)
        parts["cmeans"].set_color("red")
        parts["cmeans"].set_linewidth(1.5)
        parts["cmedians"].set_color("black")

        ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)

        # Annotate win rates
        for i, model in enumerate(MODEL_ORDER):
            vals = merged[merged["model"] == model][col].dropna()
            pct_better = (vals > 0).mean() * 100
            mean_val = vals.mean()
            ax.annotate(
                f"{pct_better:.0f}% better\n$\\mu$={mean_val:+.3f}",
                xy=(i, ax.get_ylim()[1] * 0.85 if col != "adv_exact_match" else 0.85),
                ha="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8, linewidth=0.5),
            )

        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, fontsize=8, fontweight="bold")
        ax.set_ylabel(f"Bypass - Agent ({label})", fontweight="bold")
        ax.set_title(label, fontweight="bold")

    fig.suptitle(
        f"Bypass Advantage Distribution (n = {len(common):,} per model)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, out / "RQ1" / "Figure_B_bypass_advantage.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE C – Decision outcomes (A/B/MIX selection)
# ═══════════════════════════════════════════════════════════════════════

def figure_c(df, common, out):
    """Stacked bar: A/B/MIX per model with annotated percentages."""
    logger.info("Figure C: Decision outcomes")

    bypass = df[(df["eval_method"] == "bypass7") & (df["id"].isin(common))].copy()
    bypass["model"] = bypass["model_name"].map(MODEL_SHORT)

    decision_colors = {"A": "#a6cee3", "B": "#b2df8a", "MIX": "#fb9a99"}
    decision_order = ["A", "B", "MIX"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"width_ratios": [3, 2]})

    # Panel 1: Stacked bar
    ax = axes[0]
    model_pcts = {}
    for model in MODEL_ORDER:
        sub = bypass[bypass["model"] == model]
        total = len(sub)
        if total == 0:
            continue
        counts = sub["bypass_method"].value_counts()
        model_pcts[model] = {d: counts.get(d, 0) / total * 100 for d in decision_order}

    x = np.arange(len(MODEL_ORDER))
    bottoms = np.zeros(len(MODEL_ORDER))
    for decision in decision_order:
        vals = [model_pcts.get(m, {}).get(decision, 0) for m in MODEL_ORDER]
        bars = ax.bar(x, vals, 0.55, bottom=bottoms, label=decision,
                      color=decision_colors[decision], edgecolor="white", linewidth=0.5)
        # Annotate percentages
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(x[i], b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=9)
    ax.set_ylabel("Percentage of Files (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="Selected Version", loc="upper right", fontsize=8)
    ax.set_title("(a) Selection Distribution", fontweight="bold")

    # Panel 2: Performance by decision (EM rate for A vs B selected files)
    ax2 = axes[1]
    rows = []
    for model in MODEL_ORDER:
        sub = bypass[bypass["model"] == model]
        for decision in ["A", "B"]:
            dsub = sub[sub["bypass_method"] == decision]
            if dsub.empty:
                continue
            rows.append({
                "model": model,
                "decision": f"Selected {decision}",
                "exact_match": dsub["exact_match"].mean(),
                "similarity": dsub["similarity"].mean() if "similarity" in dsub.columns else 0,
                "n": len(dsub),
            })
    perf_df = pd.DataFrame(rows)

    x2 = np.arange(len(MODEL_ORDER))
    w = 0.3
    for j, decision in enumerate(["Selected A", "Selected B"]):
        vals = [perf_df[(perf_df["model"] == m) & (perf_df["decision"] == decision)]["similarity"].values[0]
                if len(perf_df[(perf_df["model"] == m) & (perf_df["decision"] == decision)]) > 0 else 0
                for m in MODEL_ORDER]
        color = decision_colors["A"] if "A" in decision else decision_colors["B"]
        bars = ax2.bar(x2 + (j - 0.5) * w, vals, w, label=decision,
                       color=color, edgecolor="gray", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(MODEL_ORDER, fontsize=9)
    ax2.set_ylabel("Similarity Score")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)
    ax2.set_title("(b) Quality by Selected Version", fontweight="bold")

    fig.suptitle(
        "Figure C: Bypass Decision Outcomes and Selection Bias",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, out / "RQ1" / "Figure_C_decision_outcomes.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE D – Advantage by difficulty and conflict-count buckets
# ═══════════════════════════════════════════════════════════════════════

def figure_d(df, common, scenario, out):
    """Two-row panel: difficulty + conflict buckets, EM bars + similarity line."""
    logger.info("Figure D: Advantage by buckets")

    inst = _instance_agg(df, common)
    agent = inst[inst["method"] == "Agent"].copy()
    bypass = inst[inst["method"] == "Bypass"].copy()
    merged = agent.merge(bypass, on=["id", "model"], suffixes=("_ag", "_by"))

    for m in PERF:
        ac, bc = f"{m}_ag", f"{m}_by"
        if ac in merged.columns and bc in merged.columns:
            merged[f"adv_{m}"] = pd.to_numeric(merged[bc], errors="coerce") - pd.to_numeric(merged[ac], errors="coerce")

    # Get difficulty from agent side
    if "difficulty_ag" in merged.columns:
        merged["difficulty"] = merged["difficulty_ag"]

    # Enrich with scenario metadata
    scenario["id"] = scenario["id"].astype(str)
    merged = merged.merge(scenario[["id", "n_total_conflicts"]].drop_duplicates(),
                          on="id", how="left")

    # Assign conflict bucket
    def _bucket(n):
        for lo, hi, label in CONFLICT_BUCKETS:
            if lo <= n <= hi:
                return label
        return "11+"
    merged["conflict_bucket"] = merged["n_total_conflicts"].apply(_bucket)

    # Also get project_size from agent side
    if "project_size_ag" in merged.columns:
        merged["project_size"] = merged["project_size_ag"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    model_colors = {"GPT-5-nano": "#1f77b4", "Qwen3-32B": "#ff7f0e", "LLaMA-3.1-8B": "#2ca02c"}

    em_lo, em_hi = -0.03, 0.22
    sim_lo, sim_hi = -0.1, 0.55

    def _draw_row(row_idx, sub_merged, categories, cat_col, xlabel, model_colors):
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = sub_merged[sub_merged["model"] == model]

            x = np.arange(len(categories))
            em_vals = [sub[sub[cat_col] == c]["adv_exact_match"].mean() for c in categories]
            sim_vals = [sub[sub[cat_col] == c]["adv_similarity"].mean() for c in categories]
            ns = [len(sub[sub[cat_col] == c]) for c in categories]

            # Replace NaN with 0 for empty buckets
            em_vals = [v if not np.isnan(v) else 0 for v in em_vals]
            sim_vals = [v if not np.isnan(v) else 0 for v in sim_vals]

            bars = ax.bar(x, em_vals, 0.55, color=model_colors[model], alpha=0.8,
                           edgecolor="white", linewidth=0.5)
            for bar, v, n in zip(bars, em_vals, ns):
                ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.004,
                        f"{v:+.3f}\n(n={n})", ha="center", va="bottom", fontsize=6.5)

            ax2 = ax.twinx()
            ax2.plot(x, sim_vals, "s-", color="darkred", markersize=5, linewidth=1.5, zorder=5)
            for i, v in enumerate(sim_vals):
                ax2.annotate(f"{v:+.3f}", (x[i], v), textcoords="offset points",
                             xytext=(0, 8), fontsize=6.5, color="darkred", ha="center")

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            if row_idx == 2:
                ax.set_xlabel(xlabel)
            ax.set_ylabel("EM Advantage" if col_idx == 0 else "")
            if col_idx == len(MODEL_ORDER) - 1:
                ax2.set_ylabel("Similarity Advantage", color="darkred")
            else:
                ax2.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(model, fontweight="bold")

            ax.set_ylim(em_lo, em_hi)
            ax2.set_ylim(sim_lo, sim_hi)

    # --- Row 1: By difficulty ---
    _draw_row(0, merged, DIFF_ORDER, "difficulty", "Difficulty", model_colors)

    # --- Row 2: By conflict-count ---
    bucket_labels = [b[2] for b in CONFLICT_BUCKETS]
    _draw_row(1, merged, bucket_labels, "conflict_bucket", "Total Conflicts", model_colors)

    # --- Row 3: By project size ---
    available_sizes = [s for s in SIZE_ORDER if s in merged["project_size"].values]
    _draw_row(2, merged, available_sizes, "project_size", "Project Size", model_colors)

    # Row labels
    axes[0, 0].annotate("By Difficulty", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("By Conflict Count", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")
    axes[2, 0].annotate("By Project Size", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="gray", alpha=0.5, label="EM Advantage (bars)"),
        Line2D([0], [0], marker="s", color="darkred", linewidth=1.5,
               markersize=5, label="Similarity Advantage (line)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Figure D: Bypass Advantage by Difficulty, Conflict Count, and Project Size (n = {len(common):,})",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, out / "RQ2" / "Figure_D_advantage_by_buckets.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE E – Repo complexity vs advantage (scatter)
# ═══════════════════════════════════════════════════════════════════════

def figure_e(df, common, scenario, out):
    """Scatter: similarity advantage vs repo_commits and repo_code_lines, per model."""
    logger.info("Figure E: Repo complexity vs advantage")

    inst = _instance_agg(df, common)
    agent = inst[inst["method"] == "Agent"].copy()
    bypass = inst[inst["method"] == "Bypass"].copy()
    merged = agent.merge(bypass, on=["id", "model"], suffixes=("_ag", "_by"))

    for m in PERF:
        ac, bc = f"{m}_ag", f"{m}_by"
        if ac in merged.columns and bc in merged.columns:
            merged[f"adv_{m}"] = pd.to_numeric(merged[bc], errors="coerce") - pd.to_numeric(merged[ac], errors="coerce")

    scenario["id"] = scenario["id"].astype(str)
    merged = merged.merge(scenario[["id", "repo_commits", "repo_code_lines"]].drop_duplicates(),
                          on="id", how="left")

    model_colors = {"GPT-5-nano": "#1f77b4", "Qwen3-32B": "#ff7f0e", "LLaMA-3.1-8B": "#2ca02c"}

    repo_metrics = [
        ("repo_commits", "Repository Commits"),
        ("repo_code_lines", "Repository Lines of Code"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    for row_idx, (repo_col, repo_label) in enumerate(repo_metrics):
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = merged[merged["model"] == model]
            valid = sub[[repo_col, "adv_similarity"]].dropna()

            if len(valid) < 20:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            ax.scatter(valid[repo_col], valid["adv_similarity"],
                       alpha=0.12, s=12, color=model_colors[model], edgecolor="none")

            # Trend line
            z = np.polyfit(valid[repo_col], valid["adv_similarity"], 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(valid[repo_col].min(), valid[repo_col].max(), 100)
            ax.plot(x_range, p_line(x_range), color="red", linewidth=2, linestyle="--")

            # Correlation
            r, pval = stats.spearmanr(valid[repo_col], valid["adv_similarity"])
            ax.annotate(
                f"$r_s$ = {r:+.3f}{_sig(pval)}\nn = {len(valid)}",
                xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, linewidth=0.5),
            )

            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.set_xlabel(repo_label if row_idx == 1 else "")
            ax.set_ylabel("Bypass - Agent (Similarity)" if col_idx == 0 else "")
            if row_idx == 0:
                ax.set_title(model, fontweight="bold")

    # Row labels
    for row_idx, (_, label) in enumerate(repo_metrics):
        axes[row_idx, 0].annotate(
            label, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", rotation=90, va="center"
        )

    fig.suptitle(
        f"Figure E: Repository Complexity vs Bypass Advantage (n = {len(common):,})",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, out / "RQ2" / "Figure_E_repo_complexity.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PaperFlags:
    results_csv: Path = Path("data/2026_01_results_final.csv")
    dataset_csv: Path = Path("data/git_good_bench_merge_commits_all.csv")
    output_dir: Path = Path("results/paper_ready")
    slim_b: bool = False  # If True, Figure B excludes BLEU-3 and ROUGE-L


def generate_all(results_csv, dataset_csv, output_dir, *, slim_b=False):
    out = Path(output_dir)
    (out / "RQ1").mkdir(parents=True, exist_ok=True)
    (out / "RQ2").mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df = _load_results(results_csv)
    scenario = _load_scenario(Path(dataset_csv))
    common = _common_ids(df)
    logger.info(f"Common IDs: {len(common)}")

    figure_a(df, common, out)
    figure_b(df, common, out, slim=slim_b)
    figure_c(df, common, out)
    figure_d(df, common, scenario, out)
    figure_e(df, common, scenario, out)

    logger.info(f"\nAll figures saved to {out}")


if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    flags = PaperFlags()
    # Simple arg parsing
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--results-csv" and i + 1 < len(args):
            flags.results_csv = Path(args[i + 1]); i += 2
        elif args[i] == "--dataset-csv" and i + 1 < len(args):
            flags.dataset_csv = Path(args[i + 1]); i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            flags.output_dir = Path(args[i + 1]); i += 2
        elif args[i] == "--slim-b":
            flags.slim_b = True; i += 1
        else:
            i += 1

    generate_all(flags.results_csv, flags.dataset_csv, flags.output_dir,
                 slim_b=flags.slim_b)
