"""Main orchestrator script for RQ1 visualizations.

Generates all RQ1 figures with a single command:

    python -m src.results.rq1.main --input-csv data/results.csv --output-dir results/rq1

Or programmatically:

    from src.results.rq1 import generate_all_rq1_figures
    generate_all_rq1_figures("data/results.csv", "results/rq1")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import tyro

from .config import RQ1Config, DEFAULT_CONFIG
from .data import (
    compute_model_metrics,
    compute_win_tie_loss,
    compute_all_methods_metrics,
    aggregate_to_instance_level,
    GranularityType,
    compute_paired_delta_statistics,
    common_agent_bypass_ids,
)
from .dumbbell_chart import render_dumbbell_chart, render_grouped_bar_chart, render_all_methods_comparison
from .scatter_plot import render_scatter_plot, render_scatter_plot_by_model
from .win_tie_loss import render_win_tie_loss_chart, render_win_tie_loss_summary
from .bypass_distribution import (
    compute_bypass_distribution,
    compute_bypass_distribution_per_instance,
    render_bypass_distribution_bars,
    render_bypass_distribution_bars_per_instance,
    render_bypass_pie_charts,
)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RQ1Flags:
    """CLI flags for RQ1 visualization generation.

    Attributes
    ----------
    input_csv : Path
        Path to the results CSV file
    output_dir : Path
        Directory to save generated figures
    show : bool
        Whether to display figures interactively
    single_agent_method : str
        Eval method for single-agent (default: "agent")
    multi_agent_method : str
        Eval method for multi-agent (default: "bypass7")
    dumbbell : bool
        Generate dumbbell chart
    grouped_bar : bool
        Generate grouped bar chart
    scatter : bool
        Generate scatter plot (overall)
    scatter_by_model : bool
        Generate scatter plots per model
    win_tie_loss : bool
        Generate win/tie/loss stacked bars
    win_tie_loss_summary : bool
        Generate horizontal win/tie/loss summary
    summary_csv : bool
        Export summary statistics to CSV
    """

    input_csv: Path
    output_dir: Path = Path("results/rq1")
    show: bool = False

    single_agent_method: str = "agent"
    multi_agent_method: str = "bypass7"

    # Visualization toggles
    dumbbell: bool = True
    grouped_bar: bool = True
    all_methods: bool = True  # Include baselines comparison
    include_baselines: bool = True  # Include base_a/base_b in all_methods plot
    scatter: bool = True
    scatter_by_model: bool = True
    win_tie_loss: bool = True
    win_tie_loss_summary: bool = True
    bypass_distribution: bool = True  # Show A/B/MIX bypass method distribution
    summary_csv: bool = True


def generate_all_rq1_figures(
    input_csv: str | Path,
    output_dir: str | Path = "results/rq1",
    *,
    show: bool = False,
    config: Optional[RQ1Config] = None,
    dumbbell: bool = True,
    grouped_bar: bool = True,
    all_methods: bool = True,
    include_baselines: bool = True,
    scatter: bool = True,
    scatter_by_model: bool = True,
    win_tie_loss: bool = True,
    win_tie_loss_summary: bool = True,
    bypass_distribution: bool = True,
    summary_csv: bool = True,
) -> dict[str, Path]:
    """Generate all RQ1 visualizations.

    Parameters
    ----------
    input_csv : str | Path
        Path to results CSV
    output_dir : str | Path
        Output directory for figures
    show : bool
        Display figures interactively
    config : RQ1Config, optional
        Custom configuration
    dumbbell : bool
        Generate dumbbell chart
    grouped_bar : bool
        Generate grouped bar chart
    scatter : bool
        Generate scatter plot
    scatter_by_model : bool
        Generate per-model scatter plots
    win_tie_loss : bool
        Generate win/tie/loss chart
    win_tie_loss_summary : bool
        Generate summary chart
    summary_csv : bool
        Export summary to CSV

    Returns
    -------
    dict[str, Path]
        Mapping of figure names to output paths
    """
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = RQ1Config(include_baselines=include_baselines)
    else:
        config.include_baselines = include_baselines

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    # Validate required columns
    if "eval_method" not in df.columns:
        raise ValueError("Input CSV must have 'eval_method' column")

    outputs: dict[str, Path] = {}

    # 0. All Methods Comparison (with baselines)
    if all_methods:
        logger.info("Generating all methods comparison chart (with baselines)...")
        path = output_path / "rq1_all_methods_comparison.png"
        render_all_methods_comparison(df, config, output_path=path, show=show)
        outputs["all_methods"] = path
        logger.info(f"  Saved: {path}")

    # 1. Dumbbell Chart
    if dumbbell:
        logger.info("Generating dumbbell chart...")
        path = output_path / "rq1_dumbbell_chart.png"
        render_dumbbell_chart(df, config, output_path=path, show=show)
        outputs["dumbbell"] = path
        logger.info(f"  Saved: {path}")

    # 2. Grouped Bar Chart
    if grouped_bar:
        logger.info("Generating grouped bar chart...")
        path = output_path / "rq1_grouped_bar_chart.png"
        render_grouped_bar_chart(df, config, output_path=path, show=show)
        outputs["grouped_bar"] = path
        logger.info(f"  Saved: {path}")

    # 3. Scatter Plots
    if scatter:
        for metric in config.metrics:
            if metric not in df.columns:
                continue
            logger.info(f"Generating scatter plot for {metric}...")
            path = output_path / f"rq1_scatter_{metric}.png"
            render_scatter_plot(df, metric, config, output_path=path, show=show)
            outputs[f"scatter_{metric}"] = path
            logger.info(f"  Saved: {path}")

    # 4. Scatter Plots by Model
    if scatter_by_model and "model_name" in df.columns:
        for metric in config.metrics:
            if metric not in df.columns:
                continue
            logger.info(f"Generating scatter plot by model for {metric}...")
            path = output_path / f"rq1_scatter_{metric}_by_model.png"
            render_scatter_plot_by_model(df, metric, config, output_path=path, show=show)
            outputs[f"scatter_{metric}_by_model"] = path
            logger.info(f"  Saved: {path}")

    # 5. Win/Tie/Loss Chart
    if win_tie_loss:
        logger.info("Generating win/tie/loss chart...")
        path = output_path / "rq1_win_tie_loss.png"
        render_win_tie_loss_chart(df, config, output_path=path, show=show)
        outputs["win_tie_loss"] = path
        logger.info(f"  Saved: {path}")

    # 6. Win/Tie/Loss Summary (horizontal bars)
    if win_tie_loss_summary:
        for metric in ["exact_match", "similarity"]:
            if metric not in df.columns:
                continue
            logger.info(f"Generating win/tie/loss summary for {metric}...")
            path = output_path / f"rq1_win_tie_loss_summary_{metric}.png"
            render_win_tie_loss_summary(df, metric, config, output_path=path, show=show)
            outputs[f"win_tie_loss_summary_{metric}"] = path
            logger.info(f"  Saved: {path}")

    # 7. Bypass Method Distribution (A/B/MIX)
    if bypass_distribution and "bypass_method" in df.columns:
        logger.info("Generating bypass method distribution charts...")
        
        # Stacked bar chart
        path = output_path / "rq1_bypass_distribution.png"
        render_bypass_distribution_bars(df, config, output_path=path, show=show)
        outputs["bypass_distribution"] = path
        logger.info(f"  Saved: {path}")
        
        # Pie charts
        path = output_path / "rq1_bypass_pie_charts.png"
        render_bypass_pie_charts(df, config, output_path=path, show=show)
        outputs["bypass_pie_charts"] = path
        logger.info(f"  Saved: {path}")
        
        # Export bypass distribution to CSV
        bypass_dist = compute_bypass_distribution(df, config)
        if bypass_dist:
            rows = []
            for bd in bypass_dist:
                row = {
                    "model_name": bd.model_name,
                    "total": bd.total,
                }
                for method, count in bd.counts.items():
                    row[f"count_{method}"] = count
                    row[f"pct_{method}"] = bd.percentages.get(method, 0)
                rows.append(row)
            bypass_df = pd.DataFrame(rows)
            path = output_path / "rq1_bypass_distribution.csv"
            bypass_df.to_csv(path, index=False)
            outputs["bypass_distribution_csv"] = path
            logger.info(f"  Saved: {path}")
        
        # Instance-level bypass distribution
        if "id" in df.columns:
            logger.info("Generating instance-level bypass distribution...")
            
            # Stacked bar chart (per instance)
            path = output_path / "rq1_bypass_distribution_per_instance.png"
            render_bypass_distribution_bars_per_instance(df, config, output_path=path, show=show)
            outputs["bypass_distribution_per_instance"] = path
            logger.info(f"  Saved: {path}")
            
            # Export instance-level bypass distribution to CSV
            bypass_dist_inst = compute_bypass_distribution_per_instance(df, config)
            if bypass_dist_inst:
                rows = []
                for bd in bypass_dist_inst:
                    row = {
                        "model_name": bd.model_name,
                        "total_instances": bd.total,
                    }
                    for method, count in bd.counts.items():
                        row[f"count_{method}"] = count
                        row[f"pct_{method}"] = bd.percentages.get(method, 0)
                    rows.append(row)
                bypass_inst_df = pd.DataFrame(rows)
                path = output_path / "rq1_bypass_distribution_per_instance.csv"
                bypass_inst_df.to_csv(path, index=False)
                outputs["bypass_distribution_per_instance_csv"] = path
                logger.info(f"  Saved: {path}")

    # 8. Export Summary Statistics
    if summary_csv:
        logger.info("Exporting summary statistics...")

        # ========== PER-FILE (OVERALL) METRICS ==========
        logger.info("  Computing per-file (overall) metrics...")

        # Model metrics summary (per-file)
        model_metrics = compute_model_metrics(df, config, granularity="file")
        if model_metrics:
            rows = []
            for m in model_metrics:
                row = {
                    "model_name": m.model_name,
                    "n_scenarios": m.n_scenarios,
                }
                for metric in config.metrics:
                    if metric in m.single_agent:
                        row[f"single_{metric}"] = m.single_agent[metric]
                        row[f"single_{metric}_ci_low"] = m.single_agent_ci[metric][0]
                        row[f"single_{metric}_ci_high"] = m.single_agent_ci[metric][1]
                    if metric in m.multi_agent:
                        row[f"multi_{metric}"] = m.multi_agent[metric]
                        row[f"multi_{metric}_ci_low"] = m.multi_agent_ci[metric][0]
                        row[f"multi_{metric}_ci_high"] = m.multi_agent_ci[metric][1]
                    # Compute delta
                    if metric in m.single_agent and metric in m.multi_agent:
                        delta = m.multi_agent[metric] - m.single_agent[metric]
                        row[f"delta_{metric}"] = delta
                        row[f"delta_{metric}_pct"] = 100 * delta / m.single_agent[metric] if m.single_agent[metric] != 0 else 0
                rows.append(row)

            summary_df = pd.DataFrame(rows)
            path = output_path / "rq1_model_summary.csv"
            summary_df.to_csv(path, index=False)
            outputs["model_summary_csv"] = path
            logger.info(f"  Saved: {path}")

        # Win/Tie/Loss summary (per-file)
        wtl_all = []
        for metric in config.metrics:
            if metric not in df.columns:
                continue
            wtl = compute_win_tie_loss(df, metric, config, granularity="file")
            if not wtl.empty:
                wtl["metric"] = metric
                wtl_all.append(wtl)

        if wtl_all:
            wtl_df = pd.concat(wtl_all, ignore_index=True)
            path = output_path / "rq1_win_tie_loss.csv"
            wtl_df.to_csv(path, index=False)
            outputs["win_tie_loss_csv"] = path
            logger.info(f"  Saved: {path}")

        # ========== PER-INSTANCE METRICS ==========
        # For instances with multiple files:
        # - Exact Match: True only if ALL files have exact match
        # - Soft metrics: Averaged across files
        if "id" in df.columns:
            logger.info("  Computing per-instance metrics...")
            logger.info("    (EM = all files must match, soft metrics = averaged)")

            # Model metrics summary (per-instance)
            instance_metrics = compute_model_metrics(df, config, granularity="instance")
            if instance_metrics:
                rows = []
                for m in instance_metrics:
                    row = {
                        "model_name": m.model_name,
                        "n_instances": m.n_scenarios,
                    }
                    for metric in config.metrics:
                        if metric in m.single_agent:
                            row[f"single_{metric}"] = m.single_agent[metric]
                            row[f"single_{metric}_ci_low"] = m.single_agent_ci[metric][0]
                            row[f"single_{metric}_ci_high"] = m.single_agent_ci[metric][1]
                        if metric in m.multi_agent:
                            row[f"multi_{metric}"] = m.multi_agent[metric]
                            row[f"multi_{metric}_ci_low"] = m.multi_agent_ci[metric][0]
                            row[f"multi_{metric}_ci_high"] = m.multi_agent_ci[metric][1]
                        # Compute delta
                        if metric in m.single_agent and metric in m.multi_agent:
                            delta = m.multi_agent[metric] - m.single_agent[metric]
                            row[f"delta_{metric}"] = delta
                            row[f"delta_{metric}_pct"] = 100 * delta / m.single_agent[metric] if m.single_agent[metric] != 0 else 0
                    rows.append(row)

                instance_summary_df = pd.DataFrame(rows)
                path = output_path / "rq1_model_summary_per_instance.csv"
                instance_summary_df.to_csv(path, index=False)
                outputs["model_summary_instance_csv"] = path
                logger.info(f"  Saved: {path}")

            # Win/Tie/Loss summary (per-instance)
            wtl_instance_all = []
            for metric in config.metrics:
                if metric not in df.columns:
                    continue
                wtl = compute_win_tie_loss(df, metric, config, granularity="instance")
                if not wtl.empty:
                    wtl["metric"] = metric
                    wtl_instance_all.append(wtl)

            if wtl_instance_all:
                wtl_instance_df = pd.concat(wtl_instance_all, ignore_index=True)
                path = output_path / "rq1_win_tie_loss_per_instance.csv"
                wtl_instance_df.to_csv(path, index=False)
                outputs["win_tie_loss_instance_csv"] = path
                logger.info(f"  Saved: {path}")

            # All methods comparison (per-instance)
            all_methods_instance = compute_all_methods_metrics(df, config, granularity="instance")
            if all_methods_instance:
                rows = []
                for am in all_methods_instance:
                    for method, metrics_dict in am.methods.items():
                        row = {
                            "model_name": am.model_name,
                            "eval_method": method,
                            "n_instances": am.n_scenarios.get(method, 0),
                        }
                        for metric_name, value in metrics_dict.items():
                            row[metric_name] = value
                            if method in am.methods_ci and metric_name in am.methods_ci[method]:
                                ci = am.methods_ci[method][metric_name]
                                row[f"{metric_name}_ci_low"] = ci[0]
                                row[f"{metric_name}_ci_high"] = ci[1]
                        rows.append(row)

                all_methods_instance_df = pd.DataFrame(rows)
                path = output_path / "rq1_all_methods_per_instance.csv"
                all_methods_instance_df.to_csv(path, index=False)
                outputs["all_methods_instance_csv"] = path
                logger.info(f"  Saved: {path}")

            # Paired delta statistics (common agent/bypass IDs across all models)
            logger.info("  Computing paired delta statistics (common set)...")
            common = common_agent_bypass_ids(df)
            if common:
                df_common = df[df["id"].astype(str).isin(common)].copy()
                paired_stats = compute_paired_delta_statistics(
                    df_common, config, granularity="instance"
                )
                if paired_stats:
                    ps_rows = []
                    for ps in paired_stats:
                        ps_rows.append(
                            {
                                "model_name": ps.model_name,
                                "metric": ps.metric,
                                "granularity": ps.granularity,
                                "n_common_instances": ps.n_pairs,
                                "mean_delta": ps.mean_delta,
                                "mean_delta_ci_low": ps.ci_low,
                                "mean_delta_ci_high": ps.ci_high,
                                "p_value": ps.p_value,
                                "test": ps.test,
                                "wins": ps.wins,
                                "ties": ps.ties,
                                "losses": ps.losses,
                                "n_discordant": ps.n_discordant,
                            }
                        )
                    ps_df = pd.DataFrame(ps_rows)
                    path = output_path / "rq1_paired_delta_stats_per_instance.csv"
                    ps_df.to_csv(path, index=False)
                    outputs["paired_delta_stats_instance_csv"] = path
                    logger.info(f"  Saved: {path}")
            else:
                logger.warning("  Could not compute common agent/bypass ID set for paired deltas")
        else:
            logger.warning("  No 'id' column found - skipping per-instance metrics")

    logger.info(f"RQ1 analysis complete. Generated {len(outputs)} outputs in {output_path}")
    return outputs


def main(flags: RQ1Flags) -> None:
    """CLI entry point."""
    config = RQ1Config(
        single_agent_method=flags.single_agent_method,
        multi_agent_method=flags.multi_agent_method,
        include_baselines=flags.include_baselines,
    )

    generate_all_rq1_figures(
        input_csv=flags.input_csv,
        output_dir=flags.output_dir,
        show=flags.show,
        config=config,
        dumbbell=flags.dumbbell,
        grouped_bar=flags.grouped_bar,
        all_methods=flags.all_methods,
        include_baselines=flags.include_baselines,
        scatter=flags.scatter,
        scatter_by_model=flags.scatter_by_model,
        win_tie_loss=flags.win_tie_loss,
        win_tie_loss_summary=flags.win_tie_loss_summary,
        bypass_distribution=flags.bypass_distribution,
        summary_csv=flags.summary_csv,
    )


if __name__ == "__main__":
    parsed_flags = tyro.cli(RQ1Flags)
    main(parsed_flags)
