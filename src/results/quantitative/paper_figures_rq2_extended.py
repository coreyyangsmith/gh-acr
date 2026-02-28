"""Extended RQ2 analysis: labels, code metrics, and complexity vs bypass advantage.

Merges:
  - paired_data.csv        (manual labels + performance per scenario)
  - quantitative_deltas_enriched.csv  (LOC/SLOC/diff metrics from case folders)
  - complexity_metrics.csv (CC, Halstead, MI from case folders)

Produces paper-ready figures and tables in results/paper_ready/RQ2/.

Usage::

    python -m src.results.quantitative.paper_figures_rq2_extended
"""

from __future__ import annotations

import logging
import warnings
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


def _save(fig, path):
    for ext in [".pdf", ".png"]:
        fig.savefig(str(path).replace(".pdf", ext))
    plt.close(fig)
    logger.info(f"  Saved {path.name}")


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ── Label columns ────────────────────────────────────────────────────────

LABEL_COLS = [
    "favored_simplicity", "favored_complexity",
    "lost_information_compression",
    "feature_oriented", "fix_oriented", "refactor_oriented",
    "structural_change_bias", "modification_bias",
    "accurate",
    "vague_commit_message", "simple_commit_message", "detailed_commit_message",
    "test_oriented",
]

LABEL_DISPLAY = {
    "favored_simplicity": "Favored Simplicity",
    "favored_complexity": "Favored Complexity",
    "lost_information_compression": "Lost Info / Compression",
    "feature_oriented": "Feature-Oriented",
    "fix_oriented": "Fix-Oriented",
    "refactor_oriented": "Refactor-Oriented",
    "structural_change_bias": "Structural Change Bias",
    "modification_bias": "Modification Bias",
    "accurate": "Accurate",
    "vague_commit_message": "Vague Commit Msg",
    "simple_commit_message": "Simple Commit Msg",
    "detailed_commit_message": "Detailed Commit Msg",
    "test_oriented": "Test-Oriented",
}

# Code-level quantitative metrics (from deltas)
CODE_METRICS = [
    "gt_loc", "gt_sloc", "gt_diff_total_change", "gt_diff_hunks",
    "gt_diff_lines_added", "gt_diff_lines_removed",
    "n_commits_a", "n_commits_b", "n_commits_total",
]

CODE_DISPLAY = {
    "gt_loc": "GT Lines of Code",
    "gt_sloc": "GT Source LOC",
    "gt_diff_total_change": "GT Change Magnitude",
    "gt_diff_hunks": "GT Diff Hunks",
    "gt_diff_lines_added": "GT Lines Added",
    "gt_diff_lines_removed": "GT Lines Removed",
    "n_commits_a": "Commits (Branch A)",
    "n_commits_b": "Commits (Branch B)",
    "n_commits_total": "Total Commits",
}

# Complexity metrics
COMPLEXITY_METRICS = [
    "sloc", "cc_avg", "cc_max", "cc_total",
    "mi_score", "h_difficulty", "h_bugs",
]

COMPLEXITY_DISPLAY = {
    "sloc": "SLOC (radon)",
    "cc_avg": "Cyclomatic Complexity (avg)",
    "cc_max": "Cyclomatic Complexity (max)",
    "cc_total": "Cyclomatic Complexity (total)",
    "mi_score": "Maintainability Index",
    "h_difficulty": "Halstead Difficulty",
    "h_bugs": "Halstead Est. Bugs",
}

ADVANTAGE_COLS = [
    "delta_exact_match", "delta_similarity", "delta_bleu3", "delta_rouge_l",
]

ADV_DISPLAY = {
    "delta_exact_match": "EM Advantage",
    "delta_similarity": "Similarity Advantage",
    "delta_bleu3": "BLEU-3 Advantage",
    "delta_rouge_l": "ROUGE-L Advantage",
}


# ═══════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

def load_all():
    paired = pd.read_csv("results/rq3/paired_data.csv")
    paired["id"] = paired["id"].astype(str)
    logger.info(f"paired_data: {len(paired)} rows")

    deltas = pd.read_csv("results/rq_quantitative/quantitative_deltas_enriched.csv")
    deltas["sample_id"] = deltas["sample_id"].astype(str)
    deltas["_base_id"] = deltas["sample_id"].str.replace(r"-\d+$", "", regex=True)
    logger.info(f"quantitative_deltas: {len(deltas)} rows")

    complexity = pd.read_csv("results/rq3/complexity_metrics.csv")
    complexity["sample_id"] = complexity["sample_id"].astype(str)
    logger.info(f"complexity_metrics: {len(complexity)} rows")

    return paired, deltas, complexity


def merge_paired_deltas(paired, deltas):
    """Merge paired_data (labels + performance) with quantitative deltas (code metrics)."""
    merged = paired.merge(
        deltas, left_on="id", right_on="_base_id", how="inner"
    )
    merged = merged.drop(columns=["_base_id"], errors="ignore")
    logger.info(f"paired + deltas merged: {len(merged)} rows")
    return merged


def merge_with_complexity(merged, complexity):
    """Add ground_truth complexity to the merged DF."""
    gt_comp = complexity[complexity["method"] == "ground_truth"].copy()
    gt_comp["_base_id"] = gt_comp["sample_id"].str.replace(r"-\d+$", "", regex=True)
    gt_comp = gt_comp.drop_duplicates(subset=["_base_id"], keep="first")

    # Prefix complexity columns to avoid clashes
    comp_cols = [c for c in COMPLEXITY_METRICS if c in gt_comp.columns]
    rename = {c: f"gt_complexity_{c}" for c in comp_cols}
    gt_comp = gt_comp[["_base_id"] + comp_cols].rename(columns=rename)

    result = merged.merge(gt_comp, left_on="id", right_on="_base_id", how="left")
    result = result.drop(columns=["_base_id"], errors="ignore")
    logger.info(f"After complexity merge: {len(result)} rows, complexity columns matched: {result['gt_complexity_sloc'].notna().sum()}")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Labels vs Bypass Advantage
# ═══════════════════════════════════════════════════════════════════════

def analyze_labels_vs_advantage(df):
    """Mann-Whitney U: bypass advantage for samples WITH vs WITHOUT each label."""
    rows = []
    for label in LABEL_COLS:
        if label not in df.columns:
            continue
        with_label = df[df[label] == 1]
        without_label = df[df[label] == 0]
        if len(with_label) < 5 or len(without_label) < 5:
            continue

        for adv_col in ADVANTAGE_COLS:
            if adv_col not in df.columns:
                continue
            v_with = with_label[adv_col].dropna()
            v_without = without_label[adv_col].dropna()
            if len(v_with) < 5 or len(v_without) < 5:
                continue

            try:
                u_stat, u_p = stats.mannwhitneyu(v_with, v_without, alternative="two-sided")
            except Exception:
                u_stat, u_p = np.nan, np.nan

            rows.append({
                "label": label,
                "advantage_metric": adv_col,
                "mean_with": v_with.mean(),
                "mean_without": v_without.mean(),
                "diff": v_with.mean() - v_without.mean(),
                "n_with": len(v_with),
                "n_without": len(v_without),
                "mann_whitney_u": u_stat,
                "mann_whitney_p": u_p,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mann_whitney_p")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: Code metrics vs Bypass Advantage
# ═══════════════════════════════════════════════════════════════════════

def analyze_code_vs_advantage(df):
    """Spearman correlation: code-level metrics vs bypass advantage."""
    rows = []
    for code_col in CODE_METRICS:
        if code_col not in df.columns:
            continue
        for adv_col in ADVANTAGE_COLS:
            if adv_col not in df.columns:
                continue
            valid = df[[code_col, adv_col]].dropna()
            if len(valid) < 20:
                continue
            try:
                sp_r, sp_p = stats.spearmanr(valid[code_col], valid[adv_col])
                rows.append({
                    "code_metric": code_col,
                    "advantage_metric": adv_col,
                    "n": len(valid),
                    "spearman_r": sp_r,
                    "spearman_p": sp_p,
                })
            except Exception:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_r"] = result["spearman_r"].abs()
        result = result.sort_values("abs_r", ascending=False).drop(columns=["abs_r"])
    return result


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Complexity metrics vs Bypass Advantage
# ═══════════════════════════════════════════════════════════════════════

def analyze_complexity_vs_advantage(df):
    """Spearman correlation: complexity metrics vs bypass advantage."""
    rows = []
    comp_cols = [f"gt_complexity_{c}" for c in COMPLEXITY_METRICS]
    for comp_col in comp_cols:
        if comp_col not in df.columns:
            continue
        for adv_col in ADVANTAGE_COLS:
            if adv_col not in df.columns:
                continue
            valid = df[[comp_col, adv_col]].dropna()
            if len(valid) < 20:
                continue
            try:
                sp_r, sp_p = stats.spearmanr(valid[comp_col], valid[adv_col])
                rows.append({
                    "complexity_metric": comp_col.replace("gt_complexity_", ""),
                    "advantage_metric": adv_col,
                    "n": len(valid),
                    "spearman_r": sp_r,
                    "spearman_p": sp_p,
                })
            except Exception:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_r"] = result["spearman_r"].abs()
        result = result.sort_values("abs_r", ascending=False).drop(columns=["abs_r"])
    return result


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE F: Labels vs Bypass Advantage
# ═══════════════════════════════════════════════════════════════════════

def figure_f(label_results, out):
    """Horizontal bar: labels ranked by effect size on bypass similarity advantage."""
    logger.info("Figure F: Labels vs bypass advantage")

    sim = label_results[label_results["advantage_metric"] == "delta_similarity"].copy()
    if sim.empty:
        logger.warning("  No label-similarity data")
        return

    sim["label_display"] = sim["label"].map(LABEL_DISPLAY).fillna(sim["label"])
    sim["significant"] = sim["mann_whitney_p"] < 0.05
    sim = sim.sort_values("diff")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw={"width_ratios": [3, 2]})

    # Panel (a): Effect size bars (similarity advantage delta)
    ax = axes[0]
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in sim["diff"]]
    edge = ["black" if s else "gray" for s in sim["significant"]]
    lw = [1.5 if s else 0.5 for s in sim["significant"]]

    y = np.arange(len(sim))
    bars = ax.barh(y, sim["diff"], color=colors, edgecolor=edge, linewidth=lw, alpha=0.8)

    for i, (_, row) in enumerate(sim.iterrows()):
        sig = _sig(row["mann_whitney_p"])
        ax.text(
            row["diff"] + (0.003 if row["diff"] >= 0 else -0.003),
            i,
            f'{row["diff"]:+.3f}{sig}\n(n={row["n_with"]:.0f})',
            va="center", ha="left" if row["diff"] >= 0 else "right",
            fontsize=7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sim["label_display"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Bypass Advantage Difference (with label - without)")
    ax.set_title("(a) Similarity Advantage by Label", fontweight="bold")

    # Panel (b): Same for exact match
    em = label_results[label_results["advantage_metric"] == "delta_exact_match"].copy()
    if not em.empty:
        ax2 = axes[1]
        em["label_display"] = em["label"].map(LABEL_DISPLAY).fillna(em["label"])
        em = em.set_index("label").loc[sim["label"].values].reset_index()
        colors2 = ["#d62728" if d < 0 else "#2ca02c" for d in em["diff"]]
        edge2 = ["black" if p < 0.05 else "gray" for p in em["mann_whitney_p"]]
        lw2 = [1.5 if p < 0.05 else 0.5 for p in em["mann_whitney_p"]]

        ax2.barh(y, em["diff"], color=colors2, edgecolor=edge2, linewidth=lw2, alpha=0.8)
        for i, (_, row) in enumerate(em.iterrows()):
            sig = _sig(row["mann_whitney_p"])
            ax2.text(
                row["diff"] + (0.005 if row["diff"] >= 0 else -0.005),
                i,
                f'{row["diff"]:+.3f}{sig}',
                va="center", ha="left" if row["diff"] >= 0 else "right",
                fontsize=7,
            )
        ax2.set_yticks(y)
        ax2.set_yticklabels([])
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Mean EM Advantage Difference")
        ax2.set_title("(b) EM Advantage by Label", fontweight="bold")

    fig.suptitle(
        "Figure F: Classification Labels vs Bypass Advantage\n"
        "(green = label associated with larger Bypass benefit; bold border = p < 0.05)",
        fontsize=11, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    _save(fig, out / "Figure_F_labels_vs_advantage.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE G: Code metrics & complexity vs Bypass Advantage (heatmap)
# ═══════════════════════════════════════════════════════════════════════

def figure_g(code_results, complexity_results, out):
    """Combined heatmap: code metrics + complexity vs all advantage metrics."""
    logger.info("Figure G: Code + complexity heatmap")

    # Combine code and complexity results
    code_results = code_results.copy()
    code_results["metric"] = code_results["code_metric"]
    code_results["category"] = "Code / Diff"

    complexity_results = complexity_results.copy()
    complexity_results["metric"] = complexity_results["complexity_metric"]
    complexity_results["category"] = "Complexity"

    combined = pd.concat([
        code_results[["metric", "advantage_metric", "spearman_r", "spearman_p", "n", "category"]],
        complexity_results[["metric", "advantage_metric", "spearman_r", "spearman_p", "n", "category"]],
    ], ignore_index=True)

    if combined.empty:
        logger.warning("  No combined data")
        return

    # Build display names
    display_map = {**CODE_DISPLAY, **COMPLEXITY_DISPLAY}

    # Pivot
    pivot = combined.pivot_table(index="metric", columns="advantage_metric", values="spearman_r")
    p_pivot = combined.pivot_table(index="metric", columns="advantage_metric", values="spearman_p")

    # Order: code metrics first, then complexity
    code_order = [c for c in CODE_METRICS if c in pivot.index]
    comp_order = [c for c in COMPLEXITY_METRICS if c in pivot.index]
    row_order = code_order + comp_order
    pivot = pivot.reindex(row_order)
    p_pivot = p_pivot.reindex(row_order)

    # Column order
    col_order = [c for c in ADVANTAGE_COLS if c in pivot.columns]
    pivot = pivot[col_order]
    p_pivot = p_pivot[col_order]

    # Annotation
    annot = pivot.copy().astype(str)
    for row in pivot.index:
        for col in pivot.columns:
            r_val = pivot.loc[row, col]
            p_val = p_pivot.loc[row, col] if row in p_pivot.index and col in p_pivot.columns else 1.0
            if pd.isna(r_val):
                annot.loc[row, col] = ""
            else:
                annot.loc[row, col] = f"{r_val:.2f}{_sig(p_val)}"

    # Display names for axes
    y_labels = [display_map.get(m, m) for m in pivot.index]
    x_labels = [ADV_DISPLAY.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Add category prefix to labels for clarity
    prefixed_labels = []
    for i, m in enumerate(pivot.index):
        disp = display_map.get(m, m)
        prefixed_labels.append(disp)

    sns.heatmap(
        pivot.values.astype(float), annot=annot.values, fmt="",
        cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        ax=ax, linewidths=0.5, annot_kws={"fontsize": 9},
        xticklabels=x_labels, yticklabels=prefixed_labels,
    )

    # Add category divider and bracket labels
    if code_order and comp_order:
        ax.axhline(len(code_order), color="black", linewidth=2.5)
        # Use figure-level text for category labels (avoids overlap with y-ticks)
        bbox = dict(boxstyle="round,pad=0.2", facecolor="#ddeeff", edgecolor="#888", linewidth=0.5)
        ax.annotate("Code / Diff", xy=(-0.01, len(code_order) / 2),
                     xycoords=("axes fraction", "data"),
                     fontsize=8, fontweight="bold", ha="right", va="center",
                     bbox=bbox, rotation=90)
        ax.annotate("Complexity", xy=(-0.01, len(code_order) + len(comp_order) / 2),
                     xycoords=("axes fraction", "data"),
                     fontsize=8, fontweight="bold", ha="right", va="center",
                     bbox=bbox, rotation=90)

    ax.set_title(
        "Figure G: Code Metrics & Complexity vs Bypass Advantage\n"
        f"(Spearman r, n = {combined['n'].max():.0f} labeled scenarios)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    _save(fig, out / "Figure_G_code_complexity_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE H: Key scatter plots (GT change magnitude & CC vs advantage)
# ═══════════════════════════════════════════════════════════════════════

def figure_h(df, out):
    """2x2 scatter: key code/complexity metrics vs similarity advantage."""
    logger.info("Figure H: Key scatter plots")

    candidates = [
        ("gt_diff_total_change", "GT Change Magnitude\n(lines added + removed)"),
        ("n_commits_total", "Total Commits\n(Branch A + B)"),
        ("gt_complexity_cc_avg", "Cyclomatic Complexity (avg)"),
        ("gt_complexity_mi_score", "Maintainability Index"),
    ]

    adv_col = "delta_similarity"
    if adv_col not in df.columns:
        logger.warning("  No delta_similarity column")
        return

    available = [(c, l) for c, l in candidates if c in df.columns]
    if len(available) < 2:
        logger.warning("  Too few metrics available")
        return

    n = len(available)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (col, label) in enumerate(available):
        ax = axes[idx]
        valid = df[[col, adv_col]].dropna()
        if len(valid) < 20:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        ax.scatter(valid[col], valid[adv_col], alpha=0.25, s=15, color="#1f78b4", edgecolor="none")

        # Trend line
        z = np.polyfit(valid[col], valid[adv_col], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(valid[col].quantile(0.01), valid[col].quantile(0.99), 100)
        ax.plot(x_range, p_line(x_range), color="red", linewidth=2, linestyle="--")

        # Correlation
        r, pval = stats.spearmanr(valid[col], valid[adv_col])
        ax.annotate(
            f"$r_s$ = {r:+.3f}{_sig(pval)}\nn = {len(valid)}",
            xy=(0.05, 0.95), xycoords="axes fraction", fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9, linewidth=0.5),
        )

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Bypass - Agent (Similarity)")

    # Hide unused
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Figure H: Code / Complexity Metrics vs Bypass Similarity Advantage\n"
        "(labeled subset)",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, out / "Figure_H_code_complexity_scatter.pdf")


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def summary_table(label_results, code_results, complexity_results, out):
    """Export a combined summary CSV of all RQ2 correlations."""
    parts = []

    if not label_results.empty:
        lr = label_results.copy()
        lr["analysis"] = "Label vs Advantage"
        lr["metric"] = lr["label"]
        lr["statistic"] = "Mann-Whitney U"
        lr["value"] = lr["diff"]
        lr["p_value"] = lr["mann_whitney_p"]
        parts.append(lr[["analysis", "metric", "advantage_metric", "statistic",
                          "value", "p_value", "n_with", "n_without"]])

    if not code_results.empty:
        cr = code_results.copy()
        cr["analysis"] = "Code Metric vs Advantage"
        cr["metric"] = cr["code_metric"]
        cr["statistic"] = "Spearman r"
        cr["value"] = cr["spearman_r"]
        cr["p_value"] = cr["spearman_p"]
        parts.append(cr[["analysis", "metric", "advantage_metric", "statistic",
                          "value", "p_value", "n"]])

    if not complexity_results.empty:
        xr = complexity_results.copy()
        xr["analysis"] = "Complexity vs Advantage"
        xr["metric"] = xr["complexity_metric"]
        xr["statistic"] = "Spearman r"
        xr["value"] = xr["spearman_r"]
        xr["p_value"] = xr["spearman_p"]
        parts.append(xr[["analysis", "metric", "advantage_metric", "statistic",
                          "value", "p_value", "n"]])

    if parts:
        summary = pd.concat(parts, ignore_index=True)
        p = out / "Table_RQ2_extended_correlations.csv"
        summary.to_csv(p, index=False)
        logger.info(f"  Saved {p.name} ({len(summary)} rows)")
        return summary
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    out = Path("results/paper_ready/RQ2")
    out.mkdir(parents=True, exist_ok=True)

    paired, deltas, complexity = load_all()

    # Merge all data
    merged = merge_paired_deltas(paired, deltas)
    full = merge_with_complexity(merged, complexity)

    # Run analyses
    label_results = analyze_labels_vs_advantage(full)
    code_results = analyze_code_vs_advantage(full)
    complexity_results = analyze_complexity_vs_advantage(full)

    # Log key findings
    logger.info("\n=== TOP LABEL EFFECTS (similarity advantage) ===")
    sim_labels = label_results[label_results["advantage_metric"] == "delta_similarity"]
    for _, r in sim_labels.head(10).iterrows():
        logger.info(
            f"  {LABEL_DISPLAY.get(r['label'], r['label']):>30s}: "
            f"diff={r['diff']:+.4f}{_sig(r['mann_whitney_p'])} "
            f"(with={r['mean_with']:.3f}, without={r['mean_without']:.3f}, "
            f"n_with={r['n_with']:.0f})"
        )

    logger.info("\n=== TOP CODE METRIC CORRELATIONS (vs similarity advantage) ===")
    sim_code = code_results[code_results["advantage_metric"] == "delta_similarity"]
    for _, r in sim_code.head(10).iterrows():
        logger.info(
            f"  {CODE_DISPLAY.get(r['code_metric'], r['code_metric']):>30s}: "
            f"r={r['spearman_r']:+.3f}{_sig(r['spearman_p'])} (n={r['n']})"
        )

    logger.info("\n=== TOP COMPLEXITY CORRELATIONS (vs similarity advantage) ===")
    sim_comp = complexity_results[complexity_results["advantage_metric"] == "delta_similarity"]
    for _, r in sim_comp.head(10).iterrows():
        logger.info(
            f"  {COMPLEXITY_DISPLAY.get(r['complexity_metric'], r['complexity_metric']):>30s}: "
            f"r={r['spearman_r']:+.3f}{_sig(r['spearman_p'])} (n={r['n']})"
        )

    # Export tables
    label_p = out / "Table_labels_vs_advantage.csv"
    label_results.to_csv(label_p, index=False)
    logger.info(f"  Saved {label_p.name}")

    code_p = out / "Table_code_vs_advantage.csv"
    code_results.to_csv(code_p, index=False)
    logger.info(f"  Saved {code_p.name}")

    comp_p = out / "Table_complexity_vs_advantage.csv"
    complexity_results.to_csv(comp_p, index=False)
    logger.info(f"  Saved {comp_p.name}")

    summary_table(label_results, code_results, complexity_results, out)

    # Generate figures
    figure_f(label_results, out)
    figure_g(code_results, complexity_results, out)
    figure_h(full, out)

    logger.info("\nAll extended RQ2 outputs saved.")


if __name__ == "__main__":
    main()
