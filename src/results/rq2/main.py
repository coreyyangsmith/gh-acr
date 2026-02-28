"""Main orchestrator script for RQ2 visualizations.

Generates all RQ2 figures with a single command:

    python -m src.results.rq2.main --input-csv data/results.csv --output-dir results/rq2

Or programmatically:

    from src.results.rq2 import generate_all_rq2_figures
    generate_all_rq2_figures("data/results.csv", "results/rq2")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import tyro

from .config import RQ2Config, DEFAULT_CONFIG, get_short_model_name
from .data import (
    prepare_improvement_data,
    create_buckets,
    compute_stratified_metrics,
    aggregate_to_instance_level,
    GranularityType,
)
from .stratified_lift import (
    render_forest_plot,
    render_stratified_lift_by_characteristic,
    render_single_characteristic_forest,
)
from .heatmap import (
    render_difficulty_size_heatmap,
    render_interaction_heatmap,
    render_win_rate_heatmap,
)
from .regression import render_odds_ratio_plot, render_coefficient_plot, fit_logistic_model
from .distribution import (
    render_violin_by_bucket,
    render_improvement_distributions,
    render_bimodality_analysis,
)
from .method_comparison import (
    render_method_comparison_by_difficulty,
    render_method_comparison_by_project_size,
    render_method_comparison_heatmap,
    export_method_comparison_table,
)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _filter_by_model(df: pd.DataFrame, model_name: str | None) -> pd.DataFrame:
    """Filter dataframe to a specific model, or return all if None."""
    if model_name is None:
        return df
    if "model_name" not in df.columns:
        logger.warning("No 'model_name' column found - returning all data")
        return df
    # Also include baselines (which have NaN model_name)
    mask = (df["model_name"] == model_name) | (df["eval_method"].isin(["base_a", "base_b"]))
    return df[mask].copy()


def _get_unique_models(df: pd.DataFrame) -> list[str]:
    """Get unique model names from dataframe, excluding baselines."""
    if "model_name" not in df.columns:
        return []
    models = df["model_name"].dropna().unique().tolist()
    return sorted(models)


@dataclass
class RQ2Flags:
    """CLI flags for RQ2 visualization generation.

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
    model_filter : str | None
        If set, only analyze this model. If None, analyze all models.
    per_model_analysis : bool
        Generate separate outputs for each model
    stratified_lift : bool
        Generate stratified lift / forest plots
    heatmaps : bool
        Generate interaction heatmaps
    regression : bool
        Generate logistic regression plots
    distributions : bool
        Generate distribution (violin/box) plots
    bimodality : bool
        Generate bimodality analysis plots
    summary_csv : bool
        Export summary statistics to CSV
    """

    input_csv: Path
    output_dir: Path = Path("results/rq2")
    show: bool = False

    single_agent_method: str = "agent"
    multi_agent_method: str = "bypass7"
    
    # Model filtering
    model_filter: str | None = None
    per_model_analysis: bool = True

    # Visualization toggles
    stratified_lift: bool = True
    heatmaps: bool = True
    regression: bool = True
    distributions: bool = True
    bimodality: bool = True
    summary_csv: bool = True


def _generate_rq2_for_subset(
    df: pd.DataFrame,
    output_path: Path,
    model_label: str,
    *,
    show: bool = False,
    config: RQ2Config = DEFAULT_CONFIG,
    stratified_lift: bool = True,
    heatmaps: bool = True,
    regression: bool = True,
    distributions: bool = True,
    bimodality: bool = True,
    summary_csv: bool = True,
) -> dict[str, Path]:
    """Generate RQ2 outputs for a specific data subset (model or all).
    
    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered dataframe
    output_path : Path
        Output directory for this subset
    model_label : str
        Label for this subset (e.g., "Qwen3-32B" or "all_models")
    show : bool
        Display figures interactively
    config : RQ2Config
        Configuration
    stratified_lift, heatmaps, regression, distributions, bimodality, summary_csv : bool
        Toggle for each output type
        
    Returns
    -------
    dict[str, Path]
        Mapping of figure names to output paths
    """
    output_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    
    prefix = f"{model_label}_" if model_label != "all_models" else ""
    
    logger.info(f"  Generating outputs for {model_label} ({len(df)} rows)...")

    # 1. Stratified Lift / Forest Plots
    if stratified_lift:
        for metric in ["exact_match", "similarity"]:
            if metric not in df.columns:
                continue
            path = output_path / f"{prefix}rq2_forest_overview_{metric}.png"
            render_stratified_lift_by_characteristic(
                df,
                characteristics=["difficulty", "project_size", "conflict_size", "file_type"],
                metric=metric,
                config=config,
                output_path=path,
                show=show,
            )
            outputs[f"{prefix}forest_overview_{metric}"] = path

        for char in ["difficulty", "project_size", "conflict_size_bucket"]:
            path = output_path / f"{prefix}rq2_forest_{char}.png"
            render_single_characteristic_forest(
                df, char,
                metric="exact_match",
                config=config,
                output_path=path,
                show=show,
            )
            outputs[f"{prefix}forest_{char}"] = path

    # 2. Heatmaps
    if heatmaps:
        for metric in ["exact_match", "similarity"]:
            if metric not in df.columns:
                continue
            path = output_path / f"{prefix}rq2_heatmap_difficulty_conflict_{metric}.png"
            render_difficulty_size_heatmap(
                df, metric=metric, config=config, output_path=path, show=show,
            )
            outputs[f"{prefix}heatmap_difficulty_conflict_{metric}"] = path

        path = output_path / f"{prefix}rq2_heatmap_win_rate.png"
        render_win_rate_heatmap(
            df,
            row_characteristic="difficulty",
            col_characteristic="conflict_size",
            metric="exact_match",
            config=config,
            output_path=path,
            show=show,
        )
        outputs[f"{prefix}heatmap_win_rate"] = path

        if "project_size" in df.columns:
            path = output_path / f"{prefix}rq2_heatmap_difficulty_project.png"
            render_interaction_heatmap(
                df,
                row_characteristic="difficulty",
                col_characteristic="project_size",
                metric="exact_match",
                config=config,
                output_path=path,
                show=show,
            )
            outputs[f"{prefix}heatmap_difficulty_project"] = path

    # 3. Logistic Regression
    if regression:
        for metric in ["exact_match"]:
            if metric not in df.columns:
                continue

            path = output_path / f"{prefix}rq2_odds_ratio_{metric}.png"
            render_odds_ratio_plot(df, metric=metric, config=config, output_path=path, show=show)
            outputs[f"{prefix}odds_ratio_{metric}"] = path

            path = output_path / f"{prefix}rq2_coefficients_{metric}.png"
            render_coefficient_plot(df, metric=metric, config=config, output_path=path, show=show)
            outputs[f"{prefix}coefficients_{metric}"] = path

            model_result = fit_logistic_model(df, metric=metric, config=config)
            if model_result is not None:
                coef_path = output_path / f"{prefix}rq2_logistic_coefficients_{metric}.csv"
                model_result.coefficients.to_csv(coef_path, index=False)
                outputs[f"{prefix}logistic_coef_csv_{metric}"] = coef_path

    # 4. Distribution Plots
    if distributions:
        for metric in ["similarity", "exact_match"]:
            if metric not in df.columns:
                continue

            path = output_path / f"{prefix}rq2_distributions_{metric}.png"
            render_improvement_distributions(
                df,
                characteristics=["difficulty", "project_size", "conflict_size", "file_type"],
                metric=metric,
                config=config,
                output_path=path,
                show=show,
                kind="violin",
            )
            outputs[f"{prefix}distributions_{metric}"] = path

            for char in ["difficulty", "conflict_size"]:
                path = output_path / f"{prefix}rq2_violin_{char}_{metric}.png"
                render_violin_by_bucket(
                    df, char, metric=metric, config=config,
                    output_path=path, show=show, kind="violin",
                )
                outputs[f"{prefix}violin_{char}_{metric}"] = path

    # 5. Bimodality Analysis
    if bimodality:
        for char in ["difficulty", "conflict_size"]:
            path = output_path / f"{prefix}rq2_bimodality_{char}.png"
            render_bimodality_analysis(
                df, char, metric="similarity", config=config,
                output_path=path, show=show,
            )
            outputs[f"{prefix}bimodality_{char}"] = path

    # 6. Method Comparison (Baselines vs Single vs Multi)
    # Only generate for combined analysis (all_models) since it needs all methods
    if model_label == "all_models":
        logger.info("  Generating method comparison plots...")
        
        for metric in ["exact_match", "similarity"]:
            if metric not in df.columns:
                continue
            
            # By difficulty
            path = output_path / f"rq2_method_comparison_difficulty_{metric}.png"
            render_method_comparison_by_difficulty(df, metric=metric, config=config, output_path=path, show=show)
            outputs[f"method_comparison_difficulty_{metric}"] = path
            
            # By project size
            path = output_path / f"rq2_method_comparison_project_size_{metric}.png"
            render_method_comparison_by_project_size(df, metric=metric, config=config, output_path=path, show=show)
            outputs[f"method_comparison_project_size_{metric}"] = path
        
        # Heatmap comparison
        path = output_path / "rq2_method_comparison_heatmap.png"
        render_method_comparison_heatmap(df, metric="exact_match", config=config, output_path=path, show=show)
        outputs["method_comparison_heatmap"] = path
        
        # Export table
        path = output_path / "rq2_method_comparison_table.csv"
        export_method_comparison_table(df, config=config, output_path=path)
        outputs["method_comparison_table"] = path
        logger.info(f"    Saved method comparison outputs")

    # 7. Export Summary Statistics
    if summary_csv:
        # Per-file stratified metrics
        improvement_data = prepare_improvement_data(df, config, granularity="file")
        if improvement_data.n_pairs > 0:
            work = create_buckets(improvement_data.dataframe, config)

            all_stratified = []
            for char in ["difficulty", "project_size", "file_type", "conflict_size_bucket", "context_size_bucket"]:
                if char not in work.columns:
                    continue
                for metric in config.metrics:
                    result = compute_stratified_metrics(work, char, metric, config, granularity="file")
                    if not result.data.empty:
                        result_df = result.data.copy()
                        result_df["characteristic"] = char
                        result_df["metric"] = metric
                        result_df["granularity"] = "file"
                        result_df["model"] = model_label
                        all_stratified.append(result_df)

            if all_stratified:
                combined = pd.concat(all_stratified, ignore_index=True)
                path = output_path / f"{prefix}rq2_stratified_summary.csv"
                combined.to_csv(path, index=False)
                outputs[f"{prefix}stratified_summary_csv"] = path

        # Per-instance stratified metrics
        if "id" in df.columns:
            improvement_data_instance = prepare_improvement_data(df, config, granularity="instance")
            if improvement_data_instance.n_pairs > 0:
                work_instance = create_buckets(improvement_data_instance.dataframe, config)

                all_stratified_instance = []
                for char in ["difficulty", "project_size"]:
                    if char not in work_instance.columns:
                        continue
                    for metric in config.metrics:
                        result = compute_stratified_metrics(
                            work_instance, char, metric, config, granularity="instance"
                        )
                        if not result.data.empty:
                            result_df = result.data.copy()
                            result_df["characteristic"] = char
                            result_df["metric"] = metric
                            result_df["granularity"] = "instance"
                            result_df["model"] = model_label
                            all_stratified_instance.append(result_df)

                if all_stratified_instance:
                    combined_instance = pd.concat(all_stratified_instance, ignore_index=True)
                    path = output_path / f"{prefix}rq2_stratified_per_instance.csv"
                    combined_instance.to_csv(path, index=False)
                    outputs[f"{prefix}stratified_instance_csv"] = path

                # Overall per-instance summary
                instance_summary = {
                    "model": model_label,
                    "n_instances": improvement_data_instance.n_pairs,
                }
                for metric in config.metrics:
                    delta_col = f"delta_{metric}"
                    if delta_col in work_instance.columns:
                        deltas = work_instance[delta_col].dropna()
                        instance_summary[f"mean_delta_{metric}"] = deltas.mean()
                        instance_summary[f"std_delta_{metric}"] = deltas.std()
                        win_col = f"win_{metric}"
                        if win_col in work_instance.columns:
                            instance_summary[f"win_rate_{metric}"] = work_instance[win_col].mean()

                instance_summary_df = pd.DataFrame([instance_summary])
                path = output_path / f"{prefix}rq2_overall_per_instance.csv"
                instance_summary_df.to_csv(path, index=False)
                outputs[f"{prefix}overall_instance_csv"] = path

    return outputs


def generate_all_rq2_figures(
    input_csv: str | Path,
    output_dir: str | Path = "results/rq2",
    *,
    show: bool = False,
    config: Optional[RQ2Config] = None,
    stratified_lift: bool = True,
    heatmaps: bool = True,
    regression: bool = True,
    distributions: bool = True,
    bimodality: bool = True,
    summary_csv: bool = True,
) -> dict[str, Path]:
    """Generate all RQ2 visualizations.

    Parameters
    ----------
    input_csv : str | Path
        Path to results CSV
    output_dir : str | Path
        Output directory for figures
    show : bool
        Display figures interactively
    config : RQ2Config, optional
        Custom configuration
    stratified_lift : bool
        Generate stratified lift / forest plots
    heatmaps : bool
        Generate interaction heatmaps
    regression : bool
        Generate logistic regression plots
    distributions : bool
        Generate distribution (violin/box) plots
    bimodality : bool
        Generate bimodality analysis plots
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
        config = DEFAULT_CONFIG

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    if "eval_method" not in df.columns:
        raise ValueError("Input CSV must have 'eval_method' column")

    outputs: dict[str, Path] = {}
    
    # Get list of models to process
    models = _get_unique_models(df)
    logger.info(f"Found {len(models)} unique models: {[get_short_model_name(m) for m in models]}")
    
    # If model_filter is set, only process that model
    if config.model_filter:
        logger.info(f"Filtering to model: {config.model_filter}")
        df_filtered = _filter_by_model(df, config.model_filter)
        model_label = get_short_model_name(config.model_filter)
        model_outputs = _generate_rq2_for_subset(
            df_filtered, output_path, model_label,
            show=show, config=config,
            stratified_lift=stratified_lift, heatmaps=heatmaps,
            regression=regression, distributions=distributions,
            bimodality=bimodality, summary_csv=summary_csv,
        )
        outputs.update(model_outputs)
    else:
        # Generate per-model outputs if enabled
        if config.per_model_analysis and models:
            logger.info("Generating per-model analyses...")
            for model in models:
                df_model = _filter_by_model(df, model)
                model_label = get_short_model_name(model)
                model_outputs = _generate_rq2_for_subset(
                    df_model, output_path, model_label,
                    show=show, config=config,
                    stratified_lift=stratified_lift, heatmaps=heatmaps,
                    regression=regression, distributions=distributions,
                    bimodality=bimodality, summary_csv=summary_csv,
                )
                outputs.update(model_outputs)
        
        # Generate combined all-models outputs
        logger.info("Generating combined all-models analysis...")
        combined_outputs = _generate_rq2_for_subset(
            df, output_path, "all_models",
            show=show, config=config,
            stratified_lift=stratified_lift, heatmaps=heatmaps,
            regression=regression, distributions=distributions,
            bimodality=bimodality, summary_csv=summary_csv,
        )
        outputs.update(combined_outputs)
        
        # Generate cross-model comparison summary
        if summary_csv and models:
            logger.info("Generating cross-model comparison summary...")
            _generate_cross_model_summary(df, output_path, models, config)

    logger.info(f"RQ2 analysis complete. Generated {len(outputs)} outputs in {output_path}")
    return outputs


def _generate_cross_model_summary(
    df: pd.DataFrame,
    output_path: Path,
    models: list[str],
    config: RQ2Config,
) -> None:
    """Generate a cross-model comparison summary CSV."""
    rows = []
    
    for model in models:
        df_model = _filter_by_model(df, model)
        model_label = get_short_model_name(model)
        
        # Per-instance metrics
        if "id" in df_model.columns:
            improvement_data = prepare_improvement_data(df_model, config, granularity="instance")
            if improvement_data.n_pairs > 0:
                work = create_buckets(improvement_data.dataframe, config)
                row = {
                    "model": model_label,
                    "n_instances": improvement_data.n_pairs,
                }
                for metric in config.metrics:
                    delta_col = f"delta_{metric}"
                    win_col = f"win_{metric}"
                    if delta_col in work.columns:
                        deltas = work[delta_col].dropna()
                        row[f"mean_delta_{metric}"] = deltas.mean()
                        row[f"std_delta_{metric}"] = deltas.std()
                    if win_col in work.columns:
                        row[f"win_rate_{metric}"] = work[win_col].mean()
                rows.append(row)
    
    if rows:
        summary_df = pd.DataFrame(rows)
        path = output_path / "rq2_cross_model_comparison.csv"
        summary_df.to_csv(path, index=False)
        logger.info(f"  Saved: {path}")


def main(flags: RQ2Flags) -> None:
    """CLI entry point."""
    config = RQ2Config(
        single_agent_method=flags.single_agent_method,
        multi_agent_method=flags.multi_agent_method,
        model_filter=flags.model_filter,
        per_model_analysis=flags.per_model_analysis,
    )

    generate_all_rq2_figures(
        input_csv=flags.input_csv,
        output_dir=flags.output_dir,
        show=flags.show,
        config=config,
        stratified_lift=flags.stratified_lift,
        heatmaps=flags.heatmaps,
        regression=flags.regression,
        distributions=flags.distributions,
        bimodality=flags.bimodality,
        summary_csv=flags.summary_csv,
    )


if __name__ == "__main__":
    parsed_flags = tyro.cli(RQ2Flags)
    main(parsed_flags)
