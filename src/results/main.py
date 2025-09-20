from __future__ import annotations

"""One-stop CLI to generate tables and plots for results analyses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tyro

from .data_loader import load_results
from .tables import method_summary, by_difficulty_leaderboard, pairwise_win_matrix, pairwise_cost_win_matrix
from .plots_tradeoffs import (
    pareto_scatter,
    quality_vs_time,
    cost_time_per_success_bars,
    slope_chart_by_difficulty,
    tokens_to_quality_curve,
    pareto_heatmap_by_repo,
)
from .plots_distributions import ecdf_or_violin, token_composition_stacks, cost_breakdown
from .diagnostics import head_to_head_paired_plot, bland_altman, compute_paired_tests, outlier_table, correlation_metrics_table, _build_key


@dataclass
class Flags:
    """CLI switches to control which artifacts are generated and saved."""
    results_csv: Optional[Path] = None
    output_dir: Path = Path("results")
    show: bool = True

    tables: bool = True
    pairwise: bool = True
    by_difficulty: bool = True

    pareto_scatter: bool = True
    quality_vs_time: bool = True
    cost_time_per_success: bool = True
    slope_chart: bool = True
    tokens_quality_curve: bool = True
    pareto_heatmap: bool = True

    distributions_violin: bool = True
    distributions_ecdf: bool = False
    token_composition: bool = True
    cost_breakdown: bool = True

    head_to_head: bool = False
    bland_altman: bool = False
    paired_tests: bool = False
    outliers: bool = True

    # Random instance deltas plot
    random_instance_plot: bool = True
    random_seed: Optional[int] = None

    method_a: Optional[str] = None
    method_b: Optional[str] = None


def main(flags: Flags) -> None:
    """Load results and generate selected tables/plots into `output_dir`."""
    flags.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(flags.results_csv)
    df = data.dataframe

    # Tables
    if flags.tables:
        tbl = method_summary(df)
        tbl.to_csv(flags.output_dir / "method_summary.csv", index=False)

    if flags.pairwise:
        win_mat = pairwise_win_matrix(df, metric="similarity")
        win_mat.to_csv(flags.output_dir / "pairwise_win_similarity.csv")
        cost_mat = pairwise_cost_win_matrix(df)
        cost_mat.to_csv(flags.output_dir / "pairwise_win_cost.csv")

    if flags.by_difficulty and "difficulty" in df.columns:
        per_diff = by_difficulty_leaderboard(df)
        for diff, t in per_diff.items():
            t.to_csv(flags.output_dir / f"leaderboard_{diff}.csv", index=False)

    # Tradeoff plots
    if flags.pareto_scatter:
        pareto_scatter(df, save_path=flags.output_dir / "pareto_scatter.png", show=flags.show)
    if flags.quality_vs_time:
        quality_vs_time(df, save_path=flags.output_dir / "quality_vs_time.png", show=flags.show)
    if flags.cost_time_per_success:
        cost_time_per_success_bars(df, save_prefix=flags.output_dir / "cost_time_per_success", show=flags.show)
    if flags.slope_chart:
        slope_chart_by_difficulty(df, metric="exact_match", save_path=flags.output_dir / "slope_chart_exact_match.png", show=flags.show)
    if flags.tokens_quality_curve:
        tokens_to_quality_curve(df, save_path=flags.output_dir / "tokens_to_quality_curve.png", show=flags.show)
    if flags.pareto_heatmap:
        pareto_heatmap_by_repo(df, save_path=flags.output_dir / "pareto_heatmap_by_repo.png", show=flags.show)

    # Distributions
    if flags.distributions_violin:
        ecdf_or_violin(df, metrics=[m for m in ["similarity", "bleu3", "rouge_l", "total_cost", "processing_time_s"] if m in df.columns], kind="violin", save_prefix=flags.output_dir / "dist", show=flags.show)
    if flags.distributions_ecdf:
        ecdf_or_violin(df, metrics=[m for m in ["similarity", "bleu3", "rouge_l", "total_cost", "processing_time_s"] if m in df.columns], kind="ecdf", save_prefix=flags.output_dir / "ecdf", show=flags.show)
    if flags.token_composition:
        token_composition_stacks(df, save_path=flags.output_dir / "token_composition.png", show=flags.show)
    if flags.cost_breakdown:
        cost_breakdown(df, save_path=flags.output_dir / "cost_breakdown.png", show=flags.show)

    # Random instance deltas (per-method differences vs a baseline on one randomly chosen instance)
    if flags.random_instance_plot:
        random_instance_delta_plot(
            df,
            save_path=flags.output_dir / "random_instance_deltas.png",
            show=flags.show,
            seed=flags.random_seed,
        )

    # Diagnostics
    if flags.head_to_head and flags.method_a and flags.method_b:
        head_to_head_paired_plot(df, method_a=flags.method_a, method_b=flags.method_b, save_path=flags.output_dir / f"head_to_head_{flags.method_a}_vs_{flags.method_b}.png", show=flags.show)
    if flags.bland_altman:
        bland_altman(df, save_path=flags.output_dir / "bland_altman.png", show=flags.show)
    if flags.paired_tests and flags.method_a and flags.method_b:
        tests = compute_paired_tests(df, method_a=flags.method_a, method_b=flags.method_b)
        pd.DataFrame(tests).to_csv(flags.output_dir / f"paired_tests_{flags.method_a}_vs_{flags.method_b}.csv")
    if flags.outliers:
        outliers_df = outlier_table(df)
        outliers_df.to_csv(flags.output_dir / "outliers.csv", index=False)


def random_instance_delta_plot(
    df: pd.DataFrame,
    *,
    save_path: Optional[Path] = None,
    show: bool = True,
    seed: Optional[int] = None,
) -> None:
    """Pick a random instance and plot per-method differences vs a baseline.

    Metrics compared:
    - tokens_total (derived if needed)
    - total_cost
    - available eval metrics: similarity, bleu3, rouge_l, exact_match
    Baseline method defaults to highest similarity (then highest exact_match, else lowest total_cost).
    """
    if "eval_method" not in df.columns:
        return

    key = _build_key(df)
    work = df.assign(_key=key)

    # Choose an instance with results from at least 2 methods
    counts = work.groupby("_key")["eval_method"].nunique()
    eligible_keys = counts[counts >= 2].index
    if len(eligible_keys) == 0:
        return

    rng = np.random.default_rng(seed)
    chosen_key = rng.choice(eligible_keys)
    g = work[work["_key"] == chosen_key].copy()

    # Derive tokens_total if needed
    if "tokens_total" in g.columns:
        g["_tokens_total"] = g["tokens_total"]
    elif {"tokens_in", "tokens_out"}.issubset(g.columns):
        g["_tokens_total"] = g["tokens_in"].astype(float) + g["tokens_out"].astype(float)
    elif {"tokens_total_input", "tokens_output"}.issubset(g.columns):
        g["_tokens_total"] = g["tokens_total_input"].astype(float) + g["tokens_output"].astype(float)
    else:
        g["_tokens_total"] = np.nan

    metrics_eval = [m for m in ["similarity", "bleu3", "rouge_l", "exact_match"] if m in g.columns]
    value_cols = ["_tokens_total"] + (["total_cost"] if "total_cost" in g.columns else []) + metrics_eval
    if not value_cols:
        return

    wide = g.pivot_table(index="eval_method", values=value_cols, aggfunc="first")
    # Normalize types
    if "exact_match" in wide.columns:
        wide["exact_match"] = wide["exact_match"].astype(float)

    # Pick baseline
    if "similarity" in wide.columns and wide["similarity"].notna().any():
        baseline_method = wide["similarity"].idxmax()
        baseline_desc = "highest similarity"
    elif "exact_match" in wide.columns and wide["exact_match"].notna().any():
        baseline_method = wide["exact_match"].idxmax()
        baseline_desc = "highest exact_match"
    elif "total_cost" in wide.columns and wide["total_cost"].notna().any():
        baseline_method = wide["total_cost"].idxmin()
        baseline_desc = "lowest total_cost"
    else:
        baseline_method = wide.index[0]
        baseline_desc = "first available"

    baseline_vals = wide.loc[baseline_method]
    delta = wide.subtract(baseline_vals, axis=1)
    delta = delta.rename(columns={"_tokens_total": "tokens_total"})

    long = (
        delta.reset_index()
        .melt(id_vars="eval_method", var_name="metric", value_name="delta")
        .dropna(subset=["delta"])  # drop metrics that are all NaN
    )

    # Keep only intended metrics
    intended_metrics = ["tokens_total"] + (["total_cost"] if "total_cost" in delta.columns else []) + [m for m in ["similarity", "bleu3", "rouge_l", "exact_match"] if m in delta.columns]
    long = long[long["metric"].isin(intended_metrics)]

    # Split panels to avoid scale conflicts: eval metrics separate from tokens/cost
    eval_metric_names = [m for m in ["similarity", "bleu3", "rouge_l", "exact_match"] if m in delta.columns]
    long_eval = long[long["metric"].isin(eval_metric_names)]
    long_tokens = long[long["metric"] == "tokens_total"]
    long_cost = long[long["metric"] == "total_cost"] if "total_cost" in delta.columns else long.iloc[0:0]

    panels: list[tuple[str, pd.DataFrame]] = []
    if not long_eval.empty:
        panels.append(("Eval metrics (Δ vs baseline)", long_eval))
    if not long_tokens.empty:
        panels.append(("Tokens (Δ vs baseline)", long_tokens))
    if not long_cost.empty:
        panels.append(("Cost (Δ vs baseline)", long_cost))
    if not panels:
        return

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5 + 4 * ncols, 6), squeeze=False)
    axes_flat = axes[0]

    order = list(wide.index)
    for idx, (title, data_panel) in enumerate(panels):
        ax = axes_flat[idx]
        if data_panel["metric"].nunique() > 1:
            sns.barplot(data=data_panel, x="eval_method", y="delta", hue="metric", order=order, ax=ax)
        else:
            sns.barplot(data=data_panel, x="eval_method", y="delta", color=sns.color_palette()[0], order=order, ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("method")
        ax.set_ylabel("difference")
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
        # Tight y-limits for eval metrics in [−1, 1]
        if title.startswith("Eval metrics"):
            ax.set_ylim(-1.0, 1.0)
        # Simplify legends for single-metric panels
        if data_panel["metric"].nunique() <= 1 and ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.tight_layout()

    # Title/labels include brief identifier of the chosen instance
    title_parts: list[str] = []
    if "id" in g.columns and pd.notna(g["id"].iloc[0]):
        try:
            title_parts.append(f"id={int(g['id'].iloc[0])}")
        except Exception:
            title_parts.append(f"id={g['id'].iloc[0]}")
    if "repo" in g.columns and pd.notna(g["repo"].iloc[0]):
        title_parts.append(str(g["repo"].iloc[0]))
    if "file_name" in g.columns and pd.notna(g["file_name"].iloc[0]):
        title_parts.append(str(g["file_name"].iloc[0]))
    subtitle = " | ".join(title_parts) if title_parts else str(chosen_key)
    plt.title(f"Per-method deltas vs {baseline_method} ({baseline_desc})\n{subtitle}")
    plt.ylabel("difference vs baseline")
    plt.xlabel("method")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parsed_flags = tyro.cli(Flags)
    main(parsed_flags)

