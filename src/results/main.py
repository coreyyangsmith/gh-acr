from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
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
from .diagnostics import head_to_head_paired_plot, bland_altman, compute_paired_tests, outlier_table, correlation_metrics_table


@dataclass
class Flags:
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

    method_a: Optional[str] = None
    method_b: Optional[str] = None


def main(flags: Flags) -> None:
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


if __name__ == "__main__":
    parsed_flags = tyro.cli(Flags)
    main(parsed_flags)

