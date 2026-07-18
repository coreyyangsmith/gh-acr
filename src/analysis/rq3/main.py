"""Main orchestrator script for RQ3 classification analyses.

Generates all RQ3 figures and statistics with a single command:

    python -m src.analysis.rq3.main \\
        --classification-jsons data/labeled/file1.json data/labeled/file2.json \\
        --case-folders data/labeled/folder1-cases data/labeled/folder2-cases \\
        --results-csv data/2026_01_results_final.csv \\
        --output-dir results/rq3

Each JSON file is processed separately into its own subfolder within output_dir.
A combined analysis aggregating all JSON files is created in the parent output_dir.

If case_folders are provided, complexity analysis is also performed.

Output structure:
    results/rq3/
    ├── file1/                    # Subfolder for first JSON
    │   ├── aggregate.csv
    │   ├── complexity_metrics.csv
    │   ├── rq3_label_distribution.png
    │   └── ...
    ├── file2/                    # Subfolder for second JSON
    │   └── ...
    ├── aggregate_combined.csv    # Combined analysis in parent
    ├── complexity_metrics.csv    # Combined complexity metrics
    ├── rq3_label_distribution.png
    └── rq3_summary.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import tyro

from .config import RQ3Config, DEFAULT_CONFIG
from .data import (
    parse_classification_json,
    parse_multiple_json_files,
    group_by_base_id,
    grouped_to_dataframe,
    load_results_csv,
    merge_with_results,
    compute_method_pairs,
)
from .aggregation import (
    create_aggregate_dataframe,
    export_aggregate_csv,
    create_combined_aggregate,
    compute_label_summary,
    compute_co_occurrence_matrix,
    export_label_summary,
    export_co_occurrence_matrix,
)
from .statistics import (
    compute_performance_by_label,
    compute_stratified_analysis,
    compute_statistical_tests,
    compute_label_winner_correlation,
    compute_mcnemar_test,
    compute_label_improvement_tests,
    compute_selector_mcnemar_test,
    generate_summary_report,
    compute_complexity_performance_correlation,
    compute_complexity_by_label,
    compute_gt_complexity_vs_performance,
)
from .plots import (
    plot_label_distribution,
    plot_co_occurrence_heatmap,
    plot_method_comparison_by_label,
    plot_performance_delta_by_label,
    plot_label_improvement_forest,
    plot_difficulty_interaction,
    plot_project_size_interaction,
    plot_all_labels_violin,
    plot_complexity_by_method,
    plot_mi_distribution,
    plot_complexity_correlation_heatmap,
)
from .complexity_loader import (
    process_all_samples,
    aggregate_metrics_by_method,
    compute_complexity_deltas,
)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RQ3Flags:
    """CLI flags for RQ3 analysis generation.

    Attributes
    ----------
    classification_jsons : list[Path]
        Paths to JSON classification files
    results_csv : Path
        Path to the results CSV file with performance metrics
    output_dir : Path
        Directory to save generated outputs (default: results/rq3)
    case_folders : list[Path], optional
        Paths to case folders containing sample code (1:1 with classification_jsons).
        If provided, complexity analysis will be performed.
    show : bool
        Whether to display figures interactively
    
    Toggles for specific outputs:
    ----------------------------
    aggregate_csv : bool
        Export aggregate CSVs per JSON file
    combined_csv : bool
        Export combined aggregate CSV
    label_summary : bool
        Export label distribution summary
    co_occurrence : bool
        Export and plot label co-occurrence
    performance_by_label : bool
        Compute and plot performance by label
    stratified_analysis : bool
        Compute stratified analysis by difficulty/project_size
    statistical_tests : bool
        Compute statistical significance tests
    plots : bool
        Generate all visualization plots
    summary_report : bool
        Generate text summary report
    complexity_analysis : bool
        Compute code complexity metrics (requires case_folders)
    """

    classification_jsons: list[Path]
    results_csv: Path
    output_dir: Path = Path("results/rq3")
    case_folders: Optional[list[Path]] = None
    show: bool = False

    # Output toggles
    aggregate_csv: bool = True
    combined_csv: bool = True
    label_summary: bool = True
    co_occurrence: bool = True
    performance_by_label: bool = True
    stratified_analysis: bool = True
    statistical_tests: bool = True
    plots: bool = True
    summary_report: bool = True
    complexity_analysis: bool = True


def _run_single_analysis(
    aggregate_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_path: Path,
    source_name: str,
    *,
    show: bool = False,
    config: RQ3Config = DEFAULT_CONFIG,
    label_summary: bool = True,
    co_occurrence: bool = True,
    performance_by_label: bool = True,
    stratified_analysis: bool = True,
    statistical_tests: bool = True,
    plots: bool = True,
    summary_report: bool = True,
) -> dict[str, Path]:
    """Run full analysis for a single aggregate DataFrame.
    
    This is the core analysis function that processes one dataset
    (either a single JSON or the combined data).
    
    Parameters
    ----------
    aggregate_df : pd.DataFrame
        Aggregate DataFrame with label columns
    results_df : pd.DataFrame
        Results DataFrame with performance metrics
    output_path : Path
        Output directory for this analysis
    source_name : str
        Name of the source (for logging)
    show : bool
        Display figures interactively
    config : RQ3Config
        Configuration
    label_summary : bool
        Export label distribution summary
    co_occurrence : bool
        Export and plot label co-occurrence
    performance_by_label : bool
        Compute and plot performance by label
    stratified_analysis : bool
        Compute stratified analysis
    statistical_tests : bool
        Compute statistical tests
    plots : bool
        Generate visualization plots
    summary_report : bool
        Generate text summary report
        
    Returns
    -------
    dict[str, Path]
        Mapping of output names to paths
    """
    output_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    
    logger.info(f"  Processing: {source_name} -> {output_path}")
    
    # Export aggregate CSV
    csv_path = output_path / "aggregate.csv"
    export_aggregate_csv(aggregate_df, csv_path)
    outputs["aggregate"] = csv_path
    
    # Compute label summary
    label_summary_df = pd.DataFrame()
    co_occurrence_df = pd.DataFrame()
    
    if not aggregate_df.empty:
        if label_summary:
            label_summary_df = compute_label_summary(aggregate_df, config)
            summary_path = output_path / "label_distribution.csv"
            label_summary_df.to_csv(summary_path, index=False)
            outputs["label_distribution"] = summary_path
        
        if co_occurrence:
            co_occurrence_df = compute_co_occurrence_matrix(aggregate_df, config)
            cooc_path = output_path / "co_occurrence_matrix.csv"
            co_occurrence_df.to_csv(cooc_path)
            outputs["co_occurrence_matrix"] = cooc_path
    
    # Merge with results and compute pairs
    merged_df = merge_with_results(aggregate_df, results_df, config)
    paired_df = pd.DataFrame()
    
    if not merged_df.empty:
        paired_df = compute_method_pairs(merged_df, config)
        paired_path = output_path / "paired_data.csv"
        paired_df.to_csv(paired_path, index=False)
        outputs["paired_data"] = paired_path
        logger.info(f"    Paired data: {len(paired_df)} samples")
    
    # Performance by label
    perf_by_label_df = pd.DataFrame()
    if performance_by_label and not paired_df.empty:
        perf_by_label_df = compute_performance_by_label(paired_df, config)
        perf_path = output_path / "performance_by_label.csv"
        perf_by_label_df.to_csv(perf_path, index=False)
        outputs["performance_by_label"] = perf_path
    
    # Stratified analysis
    stratified_difficulty_df = pd.DataFrame()
    stratified_project_size_df = pd.DataFrame()
    
    if stratified_analysis and not paired_df.empty:
        stratified_difficulty_df = compute_stratified_analysis(paired_df, "difficulty", config)
        if not stratified_difficulty_df.empty:
            diff_path = output_path / "stratified_difficulty.csv"
            stratified_difficulty_df.to_csv(diff_path, index=False)
            outputs["stratified_difficulty"] = diff_path
        
        stratified_project_size_df = compute_stratified_analysis(paired_df, "project_size", config)
        if not stratified_project_size_df.empty:
            size_path = output_path / "stratified_project_size.csv"
            stratified_project_size_df.to_csv(size_path, index=False)
            outputs["stratified_project_size"] = size_path
    
    # Statistical tests
    stats_tests_df = pd.DataFrame()
    if statistical_tests and not paired_df.empty:
        stats_tests_df = compute_statistical_tests(paired_df, config)
        if not stats_tests_df.empty:
            tests_path = output_path / "statistical_tests.csv"
            stats_tests_df.to_csv(tests_path, index=False)
            outputs["statistical_tests"] = tests_path
    
    # Label-winner correlation analysis
    label_winner_corr_df = pd.DataFrame()
    if statistical_tests and not paired_df.empty:
        label_winner_corr_df = compute_label_winner_correlation(paired_df, config, metric="exact_match")
        if not label_winner_corr_df.empty:
            corr_path = output_path / "label_winner_correlation.csv"
            label_winner_corr_df.to_csv(corr_path, index=False)
            outputs["label_winner_correlation"] = corr_path
    
    # McNemar test (global paired method difference)
    mcnemar_result = {}
    if statistical_tests and not paired_df.empty:
        mcnemar_result = compute_mcnemar_test(paired_df, config)
        if mcnemar_result:
            mcnemar_path = output_path / "mcnemar_test.csv"
            pd.DataFrame([mcnemar_result]).to_csv(mcnemar_path, index=False)
            outputs["mcnemar_test"] = mcnemar_path
    
    # Per-label improvement tests (Fisher's exact on improve = Bypass > Agent)
    label_improvement_tests_df = pd.DataFrame()
    if statistical_tests and not paired_df.empty:
        label_improvement_tests_df = compute_label_improvement_tests(paired_df, config, metric="exact_match")
        if not label_improvement_tests_df.empty:
            imp_path = output_path / "label_improvement_tests.csv"
            label_improvement_tests_df.to_csv(imp_path, index=False)
            outputs["label_improvement_tests"] = imp_path

    # Selector McNemar test (chosen vs rejected diff — evaluates selector quality)
    selector_mcnemar_result = {}
    if statistical_tests and not results_df.empty:
        selector_mcnemar_result = compute_selector_mcnemar_test(results_df, metric="exact_match")
        if selector_mcnemar_result:
            sel_path = output_path / "selector_mcnemar_test.csv"
            pd.DataFrame([selector_mcnemar_result]).to_csv(sel_path, index=False)
            outputs["selector_mcnemar_test"] = sel_path

    # Generate plots
    if plots:
        if not label_summary_df.empty:
            plot_path = output_path / "rq3_label_distribution.png"
            plot_label_distribution(label_summary_df, config, output_path=plot_path, show=show)
            outputs["plot_label_distribution"] = plot_path
        
        if not co_occurrence_df.empty:
            plot_path = output_path / "rq3_co_occurrence_heatmap.png"
            plot_co_occurrence_heatmap(co_occurrence_df, config, output_path=plot_path, show=show)
            outputs["plot_co_occurrence"] = plot_path
        
        if not perf_by_label_df.empty:
            for metric in config.metrics:
                plot_path = output_path / f"rq3_method_comparison_{metric}.png"
                plot_method_comparison_by_label(perf_by_label_df, metric, config, output_path=plot_path, show=show)
                outputs[f"plot_method_comparison_{metric}"] = plot_path
                
                plot_path = output_path / f"rq3_delta_by_label_{metric}.png"
                plot_performance_delta_by_label(perf_by_label_df, metric, config, output_path=plot_path, show=show)
                outputs[f"plot_delta_{metric}"] = plot_path
        
        if not stratified_difficulty_df.empty:
            for metric in ["exact_match", "similarity"]:
                plot_path = output_path / f"rq3_difficulty_interaction_{metric}.png"
                plot_difficulty_interaction(stratified_difficulty_df, metric, config, output_path=plot_path, show=show)
                outputs[f"plot_difficulty_interaction_{metric}"] = plot_path
        
        if not stratified_project_size_df.empty:
            for metric in ["exact_match", "similarity"]:
                plot_path = output_path / f"rq3_project_size_interaction_{metric}.png"
                plot_project_size_interaction(stratified_project_size_df, metric, config, output_path=plot_path, show=show)
                outputs[f"plot_project_size_interaction_{metric}"] = plot_path
        
        if not paired_df.empty:
            for metric in ["similarity", "exact_match"]:
                plot_path = output_path / f"rq3_violin_all_labels_{metric}.png"
                plot_all_labels_violin(paired_df, metric, config, output_path=plot_path, show=show)
                outputs[f"plot_violin_{metric}"] = plot_path
        
        if not label_improvement_tests_df.empty:
            plot_path = output_path / "rq3_label_improvement_forest.png"
            plot_label_improvement_forest(label_improvement_tests_df, config, output_path=plot_path, show=show)
            outputs["plot_label_improvement_forest"] = plot_path
    
    # Summary report
    if summary_report:
        report = generate_summary_report(
            label_summary_df,
            perf_by_label_df,
            stats_tests_df,
            stratified_difficulty_df,
            stratified_project_size_df,
            label_winner_corr_df,
            mcnemar_result if mcnemar_result else None,
            label_improvement_tests_df if not label_improvement_tests_df.empty else None,
            selector_mcnemar_result if selector_mcnemar_result else None,
            config,
        )
        report_path = output_path / "rq3_summary.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        outputs["summary_report"] = report_path
    
    logger.info(f"    Generated {len(outputs)} outputs")
    return outputs


def generate_all_rq3_figures(
    classification_jsons: list[str | Path],
    results_csv: str | Path,
    output_dir: str | Path = "results/rq3",
    case_folders: Optional[list[str | Path]] = None,
    *,
    show: bool = False,
    config: Optional[RQ3Config] = None,
    aggregate_csv: bool = True,
    combined_csv: bool = True,
    label_summary: bool = True,
    co_occurrence: bool = True,
    performance_by_label: bool = True,
    stratified_analysis: bool = True,
    statistical_tests: bool = True,
    plots: bool = True,
    summary_report: bool = True,
    complexity_analysis: bool = True,
) -> dict[str, Path]:
    """Generate all RQ3 analyses and visualizations.
    
    Processes each JSON file separately into subfolders, then creates
    a combined analysis in the parent output directory.

    Parameters
    ----------
    classification_jsons : list[str | Path]
        Paths to JSON classification files
    results_csv : str | Path
        Path to results CSV with performance metrics
    output_dir : str | Path
        Output directory for figures and CSVs
    case_folders : list[str | Path], optional
        Paths to case folders (1:1 with classification_jsons) for complexity analysis
    show : bool
        Display figures interactively
    config : RQ3Config, optional
        Custom configuration
    aggregate_csv : bool
        Export aggregate CSVs per JSON file
    combined_csv : bool
        Export combined aggregate CSV
    label_summary : bool
        Export label distribution summary
    co_occurrence : bool
        Export and plot label co-occurrence
    performance_by_label : bool
        Compute and plot performance by label
    stratified_analysis : bool
        Compute stratified analysis
    statistical_tests : bool
        Compute statistical tests
    plots : bool
        Generate visualization plots
    summary_report : bool
        Generate text summary report
    complexity_analysis : bool
        Compute code complexity metrics (requires case_folders)

    Returns
    -------
    dict[str, Path]
        Mapping of output names to paths
    """
    json_paths = [Path(p) for p in classification_jsons]
    results_path = Path(results_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = DEFAULT_CONFIG

    outputs: dict[str, Path] = {}

    # =========================================================================
    # STEP 1: Load results CSV (shared across all analyses)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading results CSV")
    logger.info("=" * 60)
    
    results_df = load_results_csv(results_path)

    # =========================================================================
    # STEP 2: Parse JSON files and create aggregate DataFrames
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Parsing classification JSON files")
    logger.info("=" * 60)
    
    all_entries_by_source = parse_multiple_json_files(json_paths, config)
    aggregate_dfs: dict[str, pd.DataFrame] = {}
    
    for source_name, entries in all_entries_by_source.items():
        grouped = group_by_base_id(entries)
        df = grouped_to_dataframe(grouped, config)
        aggregate_dfs[source_name] = df
        logger.info(f"  {source_name}: {len(df)} unique samples")
    
    total_entries = sum(len(entries) for entries in all_entries_by_source.values())
    logger.info(f"Total entries parsed: {total_entries} from {len(all_entries_by_source)} files")

    # =========================================================================
    # STEP 3: Process each JSON file separately into subfolders
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Processing each JSON file separately")
    logger.info("=" * 60)
    
    for source_name, aggregate_df in aggregate_dfs.items():
        # Create subfolder for this source
        # Clean up the source name for folder naming
        folder_name = source_name.replace("-classifications", "").replace("_classifications", "")
        subfolder = output_path / folder_name
        
        logger.info(f"\n--- {source_name} ---")
        
        sub_outputs = _run_single_analysis(
            aggregate_df=aggregate_df,
            results_df=results_df,
            output_path=subfolder,
            source_name=source_name,
            show=show,
            config=config,
            label_summary=label_summary,
            co_occurrence=co_occurrence,
            performance_by_label=performance_by_label,
            stratified_analysis=stratified_analysis,
            statistical_tests=statistical_tests,
            plots=plots,
            summary_report=summary_report,
        )
        
        # Add to outputs with prefix
        for key, path in sub_outputs.items():
            outputs[f"{folder_name}/{key}"] = path

    # =========================================================================
    # STEP 4: Create combined analysis in parent folder
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Creating combined analysis (all JSON files)")
    logger.info("=" * 60)
    
    if len(aggregate_dfs) > 0:
        combined_df = create_combined_aggregate(aggregate_dfs, config)
        
        logger.info(f"\n--- COMBINED ({len(combined_df)} total samples) ---")
        
        combined_outputs = _run_single_analysis(
            aggregate_df=combined_df,
            results_df=results_df,
            output_path=output_path,
            source_name="combined",
            show=show,
            config=config,
            label_summary=label_summary,
            co_occurrence=co_occurrence,
            performance_by_label=performance_by_label,
            stratified_analysis=stratified_analysis,
            statistical_tests=statistical_tests,
            plots=plots,
            summary_report=summary_report,
        )
        
        # Rename the aggregate file to aggregate_combined
        if "aggregate" in combined_outputs:
            old_path = combined_outputs["aggregate"]
            new_path = output_path / "aggregate_combined.csv"
            if old_path.exists():
                # Remove existing file if it exists (for re-runs)
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)
            combined_outputs["aggregate_combined"] = new_path
            del combined_outputs["aggregate"]
        
        outputs.update(combined_outputs)

    # =========================================================================
    # STEP 5: Complexity analysis (if case_folders provided)
    # =========================================================================
    if complexity_analysis and case_folders:
        case_folder_paths = [Path(f) for f in case_folders]
        
        if len(case_folder_paths) != len(json_paths):
            logger.warning(
                f"Mismatch: {len(json_paths)} JSONs but {len(case_folder_paths)} case folders. "
                "Skipping complexity analysis."
            )
        else:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 5: Computing code complexity metrics")
            logger.info("=" * 60)
            
            all_complexity_dfs = []
            source_names = list(aggregate_dfs.keys())
            
            for i, (source_name, case_folder) in enumerate(zip(source_names, case_folder_paths)):
                aggregate_df = aggregate_dfs[source_name]
                sample_ids = aggregate_df["sample_id"].tolist() if "sample_id" in aggregate_df.columns else []
                
                if not sample_ids:
                    logger.warning(f"  No sample IDs for {source_name}")
                    continue
                
                # Determine model name from source for think tag handling
                model_name = source_name.lower()
                
                logger.info(f"\n--- Complexity: {source_name} ({len(sample_ids)} samples) ---")
                
                complexity_df = process_all_samples(case_folder, sample_ids, model_name)
                
                if not complexity_df.empty:
                    complexity_df["source"] = source_name
                    all_complexity_dfs.append(complexity_df)
                    
                    # Export per-source complexity metrics
                    folder_name = source_name.replace("-classifications", "").replace("_classifications", "")
                    subfolder = output_path / folder_name
                    complexity_path = subfolder / "complexity_metrics.csv"
                    complexity_df.to_csv(complexity_path, index=False)
                    outputs[f"{folder_name}/complexity_metrics"] = complexity_path
                    logger.info(f"  Saved: {complexity_path}")
                    
                    # Export aggregated by method
                    agg_df = aggregate_metrics_by_method(complexity_df)
                    if not agg_df.empty:
                        agg_path = subfolder / "complexity_by_method.csv"
                        agg_df.to_csv(agg_path, index=False)
                        outputs[f"{folder_name}/complexity_by_method"] = agg_path
            
            # Combined complexity metrics
            if all_complexity_dfs:
                combined_complexity = pd.concat(all_complexity_dfs, ignore_index=True)
                
                # Export combined complexity
                combined_complexity_path = output_path / "complexity_metrics.csv"
                combined_complexity.to_csv(combined_complexity_path, index=False)
                outputs["complexity_metrics"] = combined_complexity_path
                logger.info(f"\n  Combined complexity metrics: {combined_complexity_path}")
                
                # Export combined aggregation by method
                combined_agg = aggregate_metrics_by_method(combined_complexity)
                if not combined_agg.empty:
                    combined_agg_path = output_path / "complexity_by_method.csv"
                    combined_agg.to_csv(combined_agg_path, index=False)
                    outputs["complexity_by_method"] = combined_agg_path
                
                # Compute complexity deltas
                deltas_df = compute_complexity_deltas(combined_complexity)
                if not deltas_df.empty:
                    deltas_path = output_path / "complexity_deltas.csv"
                    deltas_df.to_csv(deltas_path, index=False)
                    outputs["complexity_deltas"] = deltas_path
                
                # Generate complexity plots
                if plots:
                    logger.info("\n  Generating complexity plots...")
                    
                    # Complexity by method boxplots
                    for metric in ["cc_avg", "mi_score", "sloc"]:
                        plot_path = output_path / f"complexity_{metric}_by_method.png"
                        plot_complexity_by_method(combined_complexity, metric, config, output_path=plot_path, show=show)
                        outputs[f"plot_complexity_{metric}"] = plot_path
                    
                    # MI distribution
                    mi_path = output_path / "complexity_mi_distribution.png"
                    plot_mi_distribution(combined_complexity, config, output_path=mi_path, show=show)
                    outputs["plot_mi_distribution"] = mi_path
                
                # Compute and export complexity correlations
                if statistical_tests:
                    logger.info("\n  Computing complexity correlations...")
                    
                    # Complexity vs performance correlation
                    corr_df = compute_complexity_performance_correlation(combined_complexity, results_df, config)
                    if not corr_df.empty:
                        corr_path = output_path / "complexity_performance_correlation.csv"
                        corr_df.to_csv(corr_path, index=False)
                        outputs["complexity_correlation"] = corr_path
                        
                        # Correlation heatmap
                        if plots:
                            heatmap_path = output_path / "complexity_correlation_heatmap.png"
                            plot_complexity_correlation_heatmap(corr_df, config, output_path=heatmap_path, show=show)
                            outputs["plot_complexity_correlation"] = heatmap_path
                    
                    # Complexity by label (using combined aggregate)
                    if len(aggregate_dfs) > 0:
                        combined_agg_df = create_combined_aggregate(aggregate_dfs, config)
                        complexity_by_label_df = compute_complexity_by_label(combined_complexity, combined_agg_df, config)
                        if not complexity_by_label_df.empty:
                            cbl_path = output_path / "complexity_by_label.csv"
                            complexity_by_label_df.to_csv(cbl_path, index=False)
                            outputs["complexity_by_label"] = cbl_path

    # =========================================================================
    # DONE
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RQ3 analysis complete. Generated {len(outputs)} outputs in {output_path}")
    logger.info("=" * 60)
    
    # Print summary of folder structure
    logger.info("")
    logger.info("Output structure:")
    logger.info(f"  {output_path}/")
    for source_name in aggregate_dfs.keys():
        folder_name = source_name.replace("-classifications", "").replace("_classifications", "")
        logger.info(f"    {folder_name}/  (individual analysis)")
    logger.info(f"    *.csv, *.png, *.md  (combined analysis)")
    
    return outputs


def main(flags: RQ3Flags) -> None:
    """CLI entry point."""
    config = RQ3Config()

    generate_all_rq3_figures(
        classification_jsons=flags.classification_jsons,
        results_csv=flags.results_csv,
        output_dir=flags.output_dir,
        case_folders=flags.case_folders,
        show=flags.show,
        config=config,
        aggregate_csv=flags.aggregate_csv,
        combined_csv=flags.combined_csv,
        label_summary=flags.label_summary,
        co_occurrence=flags.co_occurrence,
        performance_by_label=flags.performance_by_label,
        stratified_analysis=flags.stratified_analysis,
        statistical_tests=flags.statistical_tests,
        plots=flags.plots,
        summary_report=flags.summary_report,
        complexity_analysis=flags.complexity_analysis,
    )


if __name__ == "__main__":
    parsed_flags = tyro.cli(RQ3Flags)
    main(parsed_flags)
