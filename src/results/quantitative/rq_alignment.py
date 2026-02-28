"""Compute per-scenario Bypass-vs-Agent advantage and correlate with characteristics.

Directly addresses:
  RQ1: Does multi-agent improve quality?
  RQ2: Under what characteristics does it help or hurt?

Outputs bypass_advantage CSVs, correlation tables, and figures.
"""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path

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

PERF_METRICS = ["exact_match", "similarity", "bleu3", "rouge_l"]

MODEL_SHORT = {
    "openai/gpt-5-nano": "GPT-5-nano",
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
}

MODEL_COLORS = {
    "GPT-5-nano": "#1f77b4",
    "Qwen3-32B": "#ff7f0e",
    "LLaMA-3.1-8B": "#2ca02c",
}

SCENARIO_METRICS = [
    "n_conflict_files", "n_total_conflicts",
    "repo_commits", "repo_code_lines", "repo_contributors",
]

DIFF_ORDER = ["easy", "medium", "hard"]
SIZE_ORDER = ["small", "medium", "large", "huge"]


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


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


# ── Load scenario metadata ──────────────────────────────────────────────

def _load_scenario(dataset_csv):
    df = pd.read_csv(dataset_csv)
    rows = []
    for _, row in df.iterrows():
        e = {}
        e["id"] = str(row["Unnamed: 0"]) if "Unnamed: 0" in df.columns else str(row.name)
        if "scenario" in df.columns:
            try:
                sc = ast.literal_eval(str(row["scenario"]))
                e["n_conflict_files"] = sc.get("number_of_files_with_merge_conflict", 0)
                e["n_total_conflicts"] = sc.get("total_number_of_merge_conflicts", 0)
            except (ValueError, SyntaxError):
                e["n_conflict_files"] = 0
                e["n_total_conflicts"] = 0
        for src, dst in [("commits", "repo_commits"), ("code_lines", "repo_code_lines"),
                         ("contributors", "repo_contributors")]:
            if src in df.columns:
                try: e[dst] = int(row[src])
                except: e[dst] = 0
        for c in ["difficulty", "project_size"]:
            if c in df.columns:
                e[c] = str(row[c])
        rows.append(e)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run(results_csv, dataset_csv, output_dir="results/rq_quantitative_common"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    results = pd.read_csv(results_csv)
    results["id"] = results["id"].astype(str)
    if "exact_match" in results.columns:
        results["exact_match"] = _coerce_em(results["exact_match"])

    scenario = _load_scenario(Path(dataset_csv))
    scenario["id"] = scenario["id"].astype(str)

    # ── Find common IDs ──
    ab = results[results["eval_method"].isin(["agent", "bypass7"])]
    model_ids = {}
    for m in ab["model_name"].dropna().unique():
        model_ids[m] = set(ab[ab["model_name"] == m]["id"].unique())
    common = set.intersection(*model_ids.values())
    logger.info(f"Common IDs: {len(common)}")

    ab = ab[ab["id"].isin(common)].copy()

    # ── Aggregate multi-file to instance level ──
    agg_dict = {}
    for m in PERF_METRICS:
        if m in ab.columns:
            agg_dict[m] = "min" if m == "exact_match" else "mean"
    for c in ["difficulty", "project_size"]:
        if c in ab.columns:
            agg_dict[c] = "first"

    inst = ab.groupby(["id", "model_name", "eval_method"], as_index=False).agg(agg_dict)
    inst["model"] = inst["model_name"].map(MODEL_SHORT)

    # ── Compute Bypass advantage (Bypass - Agent) per (id, model) ──
    agent = inst[inst["eval_method"] == "agent"].copy()
    bypass = inst[inst["eval_method"] == "bypass7"].copy()

    merged = agent.merge(bypass, on=["id", "model"], suffixes=("_agent", "_bypass"))

    for m in PERF_METRICS:
        ac = f"{m}_agent"
        bc = f"{m}_bypass"
        if ac in merged.columns and bc in merged.columns:
            merged[f"advantage_{m}"] = (
                pd.to_numeric(merged[bc], errors="coerce")
                - pd.to_numeric(merged[ac], errors="coerce")
            )

    # Keep useful columns
    keep = ["id", "model"]
    keep += [f"advantage_{m}" for m in PERF_METRICS if f"advantage_{m}" in merged.columns]
    keep += [f"{m}_agent" for m in PERF_METRICS if f"{m}_agent" in merged.columns]
    keep += [f"{m}_bypass" for m in PERF_METRICS if f"{m}_bypass" in merged.columns]
    for c in ["difficulty_agent", "project_size_agent"]:
        if c in merged.columns:
            merged[c.replace("_agent", "")] = merged[c]
            keep.append(c.replace("_agent", ""))
    adv = merged[[c for c in keep if c in merged.columns]].copy()

    # Enrich with scenario metadata
    adv = adv.merge(scenario, on="id", how="left", suffixes=("", "_scen"))
    # Resolve duplicate difficulty/project_size columns
    for c in ["difficulty", "project_size"]:
        if f"{c}_scen" in adv.columns:
            adv[c] = adv[c].fillna(adv[f"{c}_scen"])
            adv = adv.drop(columns=[f"{c}_scen"])

    p = out / "bypass_advantage.csv"
    adv.to_csv(p, index=False)
    logger.info(f"Saved bypass advantage: {p} ({len(adv)} rows)")

    # ═══════════════════════════════════════════════════════════════════
    # RQ1: Overall bypass advantage summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n=== RQ1: Multi-agent improvement summary ===")
    rq1_rows = []
    for model in sorted(adv["model"].unique()):
        sub = adv[adv["model"] == model]
        row = {"model": model, "n": len(sub)}
        for m in PERF_METRICS:
            ac = f"advantage_{m}"
            if ac in sub.columns:
                vals = sub[ac].dropna()
                row[f"mean_advantage_{m}"] = vals.mean()
                row[f"median_advantage_{m}"] = vals.median()
                row[f"std_advantage_{m}"] = vals.std()
                row[f"pct_bypass_better_{m}"] = (vals > 0).mean() * 100
                row[f"pct_agent_better_{m}"] = (vals < 0).mean() * 100
                row[f"pct_tied_{m}"] = (vals == 0).mean() * 100
                # Wilcoxon signed-rank test
                try:
                    stat, pval = stats.wilcoxon(vals, alternative="two-sided")
                    row[f"wilcoxon_p_{m}"] = pval
                except Exception:
                    row[f"wilcoxon_p_{m}"] = np.nan
        rq1_rows.append(row)

    rq1_df = pd.DataFrame(rq1_rows)
    p = out / "rq1_bypass_advantage_summary.csv"
    rq1_df.to_csv(p, index=False)
    logger.info(f"Saved: {p}")

    # Print key RQ1 numbers
    for _, r in rq1_df.iterrows():
        logger.info(
            f"  {r['model']}: Bypass better on similarity {r.get('pct_bypass_better_similarity', 0):.1f}% of cases, "
            f"mean advantage = {r.get('mean_advantage_similarity', 0):+.3f}, "
            f"EM advantage = {r.get('mean_advantage_exact_match', 0):+.3f}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # RQ2: Bypass advantage vs scenario characteristics
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n=== RQ2: Characteristics that modulate bypass advantage ===")

    # 2a. Correlations: advantage vs scenario metadata
    corr_rows = []
    for model in sorted(adv["model"].unique()):
        sub = adv[adv["model"] == model]
        for sc in SCENARIO_METRICS:
            if sc not in sub.columns:
                continue
            for m in PERF_METRICS:
                ac = f"advantage_{m}"
                if ac not in sub.columns:
                    continue
                valid = sub[[sc, ac]].dropna()
                if len(valid) < 20:
                    continue
                try:
                    sp_r, sp_p = stats.spearmanr(valid[sc], valid[ac])
                    corr_rows.append({
                        "model": model,
                        "scenario_metric": sc,
                        "advantage_metric": ac,
                        "n": len(valid),
                        "spearman_r": sp_r,
                        "spearman_p": sp_p,
                    })
                except Exception:
                    pass

    # Aggregated (all models)
    for sc in SCENARIO_METRICS:
        if sc not in adv.columns:
            continue
        for m in PERF_METRICS:
            ac = f"advantage_{m}"
            if ac not in adv.columns:
                continue
            valid = adv[[sc, ac]].dropna()
            if len(valid) < 20:
                continue
            try:
                sp_r, sp_p = stats.spearmanr(valid[sc], valid[ac])
                corr_rows.append({
                    "model": "All Models",
                    "scenario_metric": sc,
                    "advantage_metric": ac,
                    "n": len(valid),
                    "spearman_r": sp_r,
                    "spearman_p": sp_p,
                })
            except Exception:
                pass

    rq2_corr = pd.DataFrame(corr_rows)
    if not rq2_corr.empty:
        rq2_corr["abs_r"] = rq2_corr["spearman_r"].abs()
        rq2_corr = rq2_corr.sort_values("abs_r", ascending=False).drop(columns=["abs_r"])
    p = out / "rq2_advantage_vs_characteristics.csv"
    rq2_corr.to_csv(p, index=False)
    logger.info(f"Saved: {p}")

    # Print top correlations
    for _, r in rq2_corr.head(20).iterrows():
        logger.info(
            f"  {r['model']:>15s} | {r['scenario_metric']:>20s} vs {r['advantage_metric']:<25s} "
            f"r={r['spearman_r']:+.3f}{_sig(r['spearman_p'])} (n={r['n']})"
        )

    # 2b. Bypass advantage by difficulty
    diff_rows = []
    for model in sorted(adv["model"].unique()):
        sub = adv[adv["model"] == model]
        for d in DIFF_ORDER:
            dsub = sub[sub["difficulty"] == d]
            if dsub.empty:
                continue
            row = {"model": model, "difficulty": d, "n": len(dsub)}
            for m in PERF_METRICS:
                ac = f"advantage_{m}"
                if ac in dsub.columns:
                    vals = dsub[ac].dropna()
                    row[f"mean_advantage_{m}"] = vals.mean()
                    row[f"pct_bypass_better_{m}"] = (vals > 0).mean() * 100
            diff_rows.append(row)

    rq2_diff = pd.DataFrame(diff_rows)
    p = out / "rq2_advantage_by_difficulty.csv"
    rq2_diff.to_csv(p, index=False)
    logger.info(f"Saved: {p}")

    # 2c. Bypass advantage by project_size
    size_rows = []
    for model in sorted(adv["model"].unique()):
        sub = adv[adv["model"] == model]
        for sz in SIZE_ORDER:
            ssub = sub[sub["project_size"] == sz]
            if ssub.empty:
                continue
            row = {"model": model, "project_size": sz, "n": len(ssub)}
            for m in PERF_METRICS:
                ac = f"advantage_{m}"
                if ac in ssub.columns:
                    vals = ssub[ac].dropna()
                    row[f"mean_advantage_{m}"] = vals.mean()
                    row[f"pct_bypass_better_{m}"] = (vals > 0).mean() * 100
            size_rows.append(row)

    rq2_size = pd.DataFrame(size_rows)
    p = out / "rq2_advantage_by_project_size.csv"
    rq2_size.to_csv(p, index=False)
    logger.info(f"Saved: {p}")

    # 2d. Bypass advantage by conflict count buckets
    bucket_rows = []
    conflict_buckets = [(1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, 100, "11+")]
    for model in sorted(adv["model"].unique()):
        sub = adv[adv["model"] == model]
        for lo, hi, label in conflict_buckets:
            bsub = sub[(sub["n_total_conflicts"] >= lo) & (sub["n_total_conflicts"] <= hi)]
            if bsub.empty:
                continue
            row = {"model": model, "conflict_bucket": label, "n": len(bsub)}
            for m in PERF_METRICS:
                ac = f"advantage_{m}"
                if ac in bsub.columns:
                    vals = bsub[ac].dropna()
                    row[f"mean_advantage_{m}"] = vals.mean()
                    row[f"pct_bypass_better_{m}"] = (vals > 0).mean() * 100
            bucket_rows.append(row)

    rq2_buckets = pd.DataFrame(bucket_rows)
    p = out / "rq2_advantage_by_conflict_count.csv"
    rq2_buckets.to_csv(p, index=False)
    logger.info(f"Saved: {p}")

    # ═══════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════

    # Plot 1: RQ1 - Bypass advantage distribution per model
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, m in enumerate(PERF_METRICS):
        ax = axes[idx]
        ac = f"advantage_{m}"
        if ac not in adv.columns:
            continue
        for model in sorted(adv["model"].unique()):
            vals = adv[adv["model"] == model][ac].dropna()
            ax.hist(vals, bins=50, alpha=0.5, label=model,
                    color=MODEL_COLORS.get(model, "#333"))
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"Bypass - Agent ({m.replace('_', ' ').title()})")
        ax.set_ylabel("Count")
        ax.set_title(m.replace("_", " ").title())
        if idx == 0:
            ax.legend(fontsize=8)
    plt.suptitle(
        "RQ1: Distribution of Bypass Advantage (Bypass - Agent)\n"
        f"Common Set, n={len(common)} per model",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out / "rq1_bypass_advantage_distribution.png")

    # Plot 2: RQ2 - Bypass advantage by difficulty
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, m in enumerate(PERF_METRICS):
        ax = axes[idx]
        col = f"mean_advantage_{m}"
        if col not in rq2_diff.columns:
            continue
        models = sorted(rq2_diff["model"].unique())
        x = np.arange(len(DIFF_ORDER))
        w = 0.25
        for i, model in enumerate(models):
            msub = rq2_diff[rq2_diff["model"] == model]
            vals = [msub[msub["difficulty"] == d][col].values[0]
                    if len(msub[msub["difficulty"] == d]) > 0 else 0
                    for d in DIFF_ORDER]
            ax.bar(x + i * w, vals, w, label=model if idx == 0 else "",
                   color=MODEL_COLORS.get(model, "#333"), alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x + w)
        ax.set_xticklabels(DIFF_ORDER)
        ax.set_ylabel(f"Mean Advantage ({m.replace('_', ' ').title()})")
        ax.set_title(m.replace("_", " ").title())
    axes[0].legend(fontsize=8)
    plt.suptitle(
        "RQ2: Bypass Advantage by Difficulty\n"
        "(Positive = Bypass better, Negative = Agent better)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out / "rq2_advantage_by_difficulty.png")

    # Plot 3: RQ2 - Bypass advantage by conflict count
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    bucket_labels = [b[2] for b in conflict_buckets]
    for idx, m in enumerate(PERF_METRICS):
        ax = axes[idx]
        col = f"mean_advantage_{m}"
        if col not in rq2_buckets.columns:
            continue
        models = sorted(rq2_buckets["model"].unique())
        x = np.arange(len(bucket_labels))
        w = 0.25
        for i, model in enumerate(models):
            msub = rq2_buckets[rq2_buckets["model"] == model]
            vals = [msub[msub["conflict_bucket"] == b][col].values[0]
                    if len(msub[msub["conflict_bucket"] == b]) > 0 else 0
                    for b in bucket_labels]
            ax.bar(x + i * w, vals, w, label=model if idx == 0 else "",
                   color=MODEL_COLORS.get(model, "#333"), alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x + w)
        ax.set_xticklabels(bucket_labels)
        ax.set_xlabel("Total Conflicts")
        ax.set_ylabel(f"Mean Advantage")
        ax.set_title(m.replace("_", " ").title())
    axes[0].legend(fontsize=8)
    plt.suptitle(
        "RQ2: Bypass Advantage by Conflict Count\n"
        "(Positive = Bypass better, Negative = Agent better)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out / "rq2_advantage_by_conflict_count.png")

    # Plot 4: RQ2 - Correlation heatmap of advantage vs characteristics
    if not rq2_corr.empty:
        # Filter to all models aggregate + per-model for advantage_similarity
        for metric_label, adv_col in [("similarity", "advantage_similarity"),
                                       ("exact_match", "advantage_exact_match")]:
            sub = rq2_corr[rq2_corr["advantage_metric"] == adv_col]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index="scenario_metric", columns="model", values="spearman_r"
            )
            p_pivot = sub.pivot_table(
                index="scenario_metric", columns="model", values="spearman_p"
            )
            annot = pivot.copy().astype(str)
            for row in pivot.index:
                for col in pivot.columns:
                    r_val = pivot.loc[row, col]
                    p_val = p_pivot.loc[row, col] if row in p_pivot.index and col in p_pivot.columns else 1.0
                    annot.loc[row, col] = f"{r_val:.3f}{_sig(p_val)}" if not pd.isna(r_val) else ""

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(pivot, annot=annot, fmt="", cmap="RdBu_r", center=0,
                        vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.5)
            ax.set_title(
                f"RQ2: Scenario Characteristics vs Bypass Advantage ({metric_label.replace('_', ' ').title()})\n"
                f"(Spearman r; positive = more conflicts -> bigger Bypass advantage)",
                fontsize=12, fontweight="bold"
            )
            plt.tight_layout()
            _save(fig, out / f"rq2_advantage_correlation_{metric_label}.png")

    # Plot 5: RQ2 - Where bypass HURTS (GPT-5-nano similarity)
    gpt_adv = adv[adv["model"] == "GPT-5-nano"]["advantage_similarity"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gpt_adv, bins=60, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=2)
    ax.axvline(gpt_adv.mean(), color="red", linestyle="-", linewidth=2,
               label=f"Mean = {gpt_adv.mean():+.3f}")
    ax.axvline(gpt_adv.median(), color="orange", linestyle="-", linewidth=2,
               label=f"Median = {gpt_adv.median():+.3f}")
    pct_worse = (gpt_adv < 0).mean() * 100
    pct_better = (gpt_adv > 0).mean() * 100
    ax.annotate(
        f"Bypass better: {pct_better:.1f}%\nBypass worse: {pct_worse:.1f}%",
        xy=(0.05, 0.95), xycoords="axes fraction", fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9)
    )
    ax.set_xlabel("Bypass - Agent (Similarity)")
    ax.set_ylabel("Count")
    ax.set_title(
        "RQ2: GPT-5-nano Bypass Advantage Distribution (Similarity)\n"
        "Where Multi-Agent Helps vs Hurts",
        fontsize=12, fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, out / "rq2_gpt5nano_bypass_hurts.png")

    logger.info("\nDone.")
    return {
        "bypass_advantage": adv,
        "rq1_summary": rq1_df,
        "rq2_correlations": rq2_corr,
        "rq2_by_difficulty": rq2_diff,
        "rq2_by_conflict": rq2_buckets,
    }


if __name__ == "__main__":
    run(
        results_csv="data/2026_01_results_final.csv",
        dataset_csv="data/git_good_bench_merge_commits_all.csv",
        output_dir="results/rq_quantitative_common",
    )
