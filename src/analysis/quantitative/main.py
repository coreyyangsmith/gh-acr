"""Main orchestrator for quantitative change metrics analysis.

Computes commit counts, PR length, LOC/SLOC changes, and diff-based
metrics, then runs correlation analysis and generates figures.

Usage::

    python -m src.analysis.quantitative.main \\
        --case-folders data/folder1 data/folder2 \\
        --classification-jsons data/labeled/file1.json data/labeled/file2.json \\
        --results-csv results/em_datasets/models_combined.csv \\
        --output-dir results/rq_quantitative

    # Optionally include dataset CSV for scenario metadata:
    python -m src.analysis.quantitative.main \\
        ... \\
        --dataset-csv data/git_good_bench_merge_commits_all_working.csv
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import QuantConfig, DEFAULT_CONFIG
from .loader import (
    process_all_samples,
    aggregate_metrics_by_version,
    compute_quantitative_deltas,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── CLI flags ─────────────────────────────────────────────────────────────


@dataclass
class QuantFlags:
    """CLI flags for quantitative analysis.

    Attributes
    ----------
    case_folders : list[Path]
        Paths to case folders containing sample subfolders
    results_csv : Path
        Path to the results CSV with performance metrics
    output_dir : Path
        Output directory for CSVs and figures
    classification_jsons : list[Path], optional
        Paths to RQ3 classification JSON files (for label correlation)
    dataset_csv : Path, optional
        Path to the GitGoodBench dataset CSV (for scenario metadata)
    rq3_paired_csv : Path, optional
        Path to RQ3 paired_data.csv (for label correlation)
    rq3_complexity_csv : Path, optional
        Path to RQ3 complexity_metrics.csv (for cross-analysis)
    plots : bool
        Generate visualization plots
    correlations : bool
        Compute correlation analyses
    """

    case_folders: list[Path]
    results_csv: Path
    output_dir: Path = Path("results/rq_quantitative")

    # Optional inputs for cross-analysis
    classification_jsons: Optional[list[Path]] = None
    dataset_csv: Optional[Path] = None
    rq3_paired_csv: Optional[Path] = None
    rq3_complexity_csv: Optional[Path] = None

    # Output toggles
    plots: bool = True
    correlations: bool = True


# ── Dataset metadata enrichment ───────────────────────────────────────────


def _load_scenario_metadata(dataset_csv: Path) -> pd.DataFrame:
    """Load scenario-level metadata from the GitGoodBench dataset CSV.

    Extracts from the ``scenario`` JSON column:
    - ``n_conflict_files``
    - ``n_total_conflicts``

    And from the CSV columns:
    - ``repo_commits``, ``repo_code_lines``, ``repo_contributors``

    Parameters
    ----------
    dataset_csv : Path
        Path to the benchmark CSV

    Returns
    -------
    pd.DataFrame
        DataFrame with ``id`` and metadata columns
    """
    logger.info(f"Loading scenario metadata from {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    rows = []
    for _, row in df.iterrows():
        entry: dict = {}

        # The numeric scenario ID is in the first column (Unnamed: 0),
        # which is the original pandas index stored when the CSV was saved.
        # The 'id' column contains a repo-name string, not the numeric ID.
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

        # Repository-level metadata
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

        rows.append(entry)

    return pd.DataFrame(rows)


# ── Sample ID extraction ─────────────────────────────────────────────────


def _extract_sample_ids_from_case_folder(case_folder: Path) -> list[str]:
    """List sample IDs from subfolders in a case folder."""
    ids = []
    if not case_folder.exists():
        logger.warning(f"Case folder does not exist: {case_folder}")
        return ids
    for sub in sorted(case_folder.iterdir()):
        if sub.is_dir():
            # Extract base ID (strip suffix like -1, -2)
            name = sub.name
            base_id = name.rsplit("-", 1)[0] if "-" in name else name
            if base_id not in ids:
                ids.append(base_id)
    return ids


def _extract_sample_ids_from_json(json_paths: list[Path]) -> list[str]:
    """Extract sample IDs from RQ3 classification JSON files."""
    import json
    import re

    ids = set()
    for json_path in json_paths:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            classifications = data.get("classifications", {})
            for full_id in classifications:
                # Extract base ID: "106151546310-1" → "106151546310"
                base_id = re.sub(r"-\d+$", "", str(full_id))
                ids.add(base_id)
        except Exception as e:
            logger.warning(f"Error reading {json_path}: {e}")
    return sorted(ids)


# ── Main pipeline ─────────────────────────────────────────────────────────


def generate_all_quantitative(
    case_folders: list[str | Path],
    results_csv: str | Path,
    output_dir: str | Path = "results/rq_quantitative",
    classification_jsons: Optional[list[str | Path]] = None,
    dataset_csv: Optional[str | Path] = None,
    rq3_paired_csv: Optional[str | Path] = None,
    rq3_complexity_csv: Optional[str | Path] = None,
    *,
    plots: bool = True,
    correlations: bool = True,
    config: Optional[QuantConfig] = None,
) -> dict[str, Path]:
    """Generate all quantitative analyses.

    Parameters
    ----------
    case_folders : list[str | Path]
        Paths to case folders containing sample subfolders
    results_csv : str | Path
        Path to results CSV with performance metrics
    output_dir : str | Path
        Output directory for CSVs and figures
    classification_jsons : list[str | Path], optional
        Paths to RQ3 classification JSON files
    dataset_csv : str | Path, optional
        Path to the GitGoodBench dataset CSV
    rq3_paired_csv : str | Path, optional
        Path to RQ3 paired_data.csv
    rq3_complexity_csv : str | Path, optional
        Path to RQ3 complexity_metrics.csv
    plots : bool
        Generate visualization plots
    correlations : bool
        Compute correlation analyses
    config : QuantConfig, optional
        Custom configuration

    Returns
    -------
    dict[str, Path]
        Mapping of output names to paths
    """
    case_paths = [Path(f) for f in case_folders]
    results_path = Path(results_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = DEFAULT_CONFIG

    outputs: dict[str, Path] = {}

    # =====================================================================
    # STEP 1: Gather sample IDs
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Gathering sample IDs")
    logger.info("=" * 60)

    all_sample_ids: list[str] = []

    if classification_jsons:
        json_paths = [Path(p) for p in classification_jsons]
        json_ids = _extract_sample_ids_from_json(json_paths)
        logger.info(f"  From JSON files: {len(json_ids)} unique sample IDs")
        all_sample_ids.extend(json_ids)

    for cf in case_paths:
        folder_ids = _extract_sample_ids_from_case_folder(cf)
        logger.info(f"  From {cf.name}: {len(folder_ids)} sample IDs")
        # Only add IDs not already present
        for sid in folder_ids:
            if sid not in all_sample_ids:
                all_sample_ids.append(sid)

    logger.info(f"  Total unique sample IDs: {len(all_sample_ids)}")

    # =====================================================================
    # STEP 2: Compute quantitative metrics from case folders
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Computing quantitative metrics")
    logger.info("=" * 60)

    all_metrics_dfs: list[pd.DataFrame] = []

    for case_folder in case_paths:
        logger.info(f"\n--- Processing: {case_folder.name} ---")

        # Use all sample IDs for this folder
        folder_ids = _extract_sample_ids_from_case_folder(case_folder)
        if not folder_ids:
            logger.warning(f"  No sample folders found in {case_folder}")
            continue

        metrics_df = process_all_samples(case_folder, folder_ids)
        if not metrics_df.empty:
            metrics_df["source"] = case_folder.name
            all_metrics_dfs.append(metrics_df)
            logger.info(f"  Computed {len(metrics_df)} metric rows")

    if not all_metrics_dfs:
        logger.error("No quantitative metrics computed. Exiting.")
        return outputs

    combined_metrics = pd.concat(all_metrics_dfs, ignore_index=True)

    # Deduplicate: keep first occurrence per (sample_id, version)
    combined_metrics = combined_metrics.drop_duplicates(
        subset=["sample_id", "version"], keep="first"
    )

    # Export raw metrics
    metrics_path = output_path / "quantitative_metrics.csv"
    combined_metrics.to_csv(metrics_path, index=False)
    outputs["quantitative_metrics"] = metrics_path
    logger.info(f"\n  Saved: {metrics_path} ({len(combined_metrics)} rows)")

    # =====================================================================
    # STEP 3: Aggregate summary by version
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Computing summary statistics")
    logger.info("=" * 60)

    summary_df = aggregate_metrics_by_version(combined_metrics)
    if not summary_df.empty:
        summary_path = output_path / "quantitative_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        outputs["quantitative_summary"] = summary_path
        logger.info(f"  Saved: {summary_path}")

    # =====================================================================
    # STEP 4: Compute pairwise deltas
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Computing pairwise deltas")
    logger.info("=" * 60)

    deltas_df = compute_quantitative_deltas(combined_metrics)
    if not deltas_df.empty:
        deltas_path = output_path / "quantitative_deltas.csv"
        deltas_df.to_csv(deltas_path, index=False)
        outputs["quantitative_deltas"] = deltas_path
        logger.info(f"  Saved: {deltas_path} ({len(deltas_df)} rows)")

    # =====================================================================
    # STEP 5: Enrich with scenario metadata (optional)
    # =====================================================================
    if dataset_csv:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 5: Enriching with scenario metadata")
        logger.info("=" * 60)

        ds_path = Path(dataset_csv)
        if ds_path.exists():
            scenario_meta = _load_scenario_metadata(ds_path)

            # Merge into deltas (on sample_id ↔ id, strip suffix)
            if not deltas_df.empty and not scenario_meta.empty:
                deltas_df["_base_id"] = (
                    deltas_df["sample_id"]
                    .astype(str)
                    .str.replace(r"-\d+$", "", regex=True)
                )
                deltas_enriched = deltas_df.merge(
                    scenario_meta,
                    left_on="_base_id",
                    right_on="id",
                    how="left",
                )
                deltas_enriched = deltas_enriched.drop(
                    columns=["_base_id"], errors="ignore"
                )
                # Drop duplicate id column
                if "id" in deltas_enriched.columns:
                    deltas_enriched = deltas_enriched.drop(columns=["id"])

                enriched_path = output_path / "quantitative_deltas_enriched.csv"
                deltas_enriched.to_csv(enriched_path, index=False)
                outputs["quantitative_deltas_enriched"] = enriched_path
                logger.info(f"  Saved: {enriched_path}")
                # Use enriched version going forward
                deltas_df = deltas_enriched
        else:
            logger.warning(f"  Dataset CSV not found: {ds_path}")

    # =====================================================================
    # STEP 6: Load results CSV for performance data
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Loading performance results")
    logger.info("=" * 60)

    results_df = pd.read_csv(results_path)
    logger.info(f"  Loaded {len(results_df)} rows from {results_path}")

    # =====================================================================
    # STEP 7: Correlation analysis (optional)
    # =====================================================================
    if correlations:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 7: Running correlation analysis")
        logger.info("=" * 60)

        from .correlations import (
            compute_performance_correlations,
            compute_label_correlations,
            compute_complexity_cross_correlations,
        )

        # 7a. Quantitative vs Performance
        perf_corr = compute_performance_correlations(
            deltas_df, results_df, config
        )
        if not perf_corr.empty:
            perf_path = output_path / "quantitative_performance_correlation.csv"
            perf_corr.to_csv(perf_path, index=False)
            outputs["performance_correlation"] = perf_path
            logger.info(f"  Saved: {perf_path}")

        # 7b. Quantitative vs Labels
        paired_path = Path(rq3_paired_csv) if rq3_paired_csv else None
        if paired_path and paired_path.exists():
            paired_df = pd.read_csv(paired_path)
            label_corr = compute_label_correlations(
                deltas_df, paired_df, config
            )
            if not label_corr.empty:
                label_path = output_path / "quantitative_label_correlation.csv"
                label_corr.to_csv(label_path, index=False)
                outputs["label_correlation"] = label_path
                logger.info(f"  Saved: {label_path}")

        # 7c. Quantitative vs Complexity
        complexity_path = Path(rq3_complexity_csv) if rq3_complexity_csv else None
        if complexity_path and complexity_path.exists():
            complexity_df = pd.read_csv(complexity_path)
            cross_corr = compute_complexity_cross_correlations(
                deltas_df, complexity_df, config
            )
            if not cross_corr.empty:
                cross_path = output_path / "quantitative_complexity_cross.csv"
                cross_corr.to_csv(cross_path, index=False)
                outputs["complexity_cross"] = cross_path
                logger.info(f"  Saved: {cross_path}")

    # =====================================================================
    # STEP 8: Generate plots (optional)
    # =====================================================================
    if plots:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 8: Generating plots")
        logger.info("=" * 60)

        from .plots import (
            plot_size_by_version,
            plot_change_magnitude_by_version,
            plot_commit_count_distribution,
            plot_change_by_difficulty,
            plot_correlation_heatmap,
            plot_metric_vs_performance_scatter,
            plot_metrics_by_label,
            plot_summary_table,
        )

        # 8a. Descriptive plots
        p = output_path / "quant_size_by_version.png"
        plot_size_by_version(combined_metrics, config, output_path=p)
        outputs["plot_size_by_version"] = p

        p = output_path / "quant_change_magnitude_by_version.png"
        plot_change_magnitude_by_version(combined_metrics, config, output_path=p)
        outputs["plot_change_magnitude"] = p

        p = output_path / "quant_commit_count_distribution.png"
        plot_commit_count_distribution(combined_metrics, config, output_path=p)
        outputs["plot_commit_distribution"] = p

        p = output_path / "quant_change_by_difficulty.png"
        plot_change_by_difficulty(
            combined_metrics, results_df, config, output_path=p
        )
        outputs["plot_change_by_difficulty"] = p

        # 8b. Correlation plots
        perf_corr_path = output_path / "quantitative_performance_correlation.csv"
        if perf_corr_path.exists():
            perf_corr_df = pd.read_csv(perf_corr_path)

            p = output_path / "quant_correlation_heatmap.png"
            plot_correlation_heatmap(perf_corr_df, config, output_path=p)
            outputs["plot_correlation_heatmap"] = p

            p = output_path / "quant_metric_vs_performance.png"
            plot_metric_vs_performance_scatter(
                deltas_df, results_df, config, output_path=p
            )
            outputs["plot_metric_vs_performance"] = p

        # 8c. Label interaction plots
        paired_path_obj = Path(rq3_paired_csv) if rq3_paired_csv else None
        if paired_path_obj and paired_path_obj.exists():
            paired_df = pd.read_csv(paired_path_obj)

            p = output_path / "quant_metrics_by_label.png"
            plot_metrics_by_label(deltas_df, paired_df, config, output_path=p)
            outputs["plot_metrics_by_label"] = p

        # 8d. Summary table
        p = output_path / "quant_summary_table.png"
        plot_summary_table(summary_df, config, output_path=p)
        outputs["plot_summary_table"] = p

    # =====================================================================
    # DONE
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"Quantitative analysis complete. Generated {len(outputs)} outputs "
        f"in {output_path}"
    )
    logger.info("=" * 60)

    return outputs


# ── CLI entry point ──────────────────────────────────────────────────────


def main(flags: QuantFlags) -> None:
    """CLI entry point."""
    config = QuantConfig()

    generate_all_quantitative(
        case_folders=flags.case_folders,
        results_csv=flags.results_csv,
        output_dir=flags.output_dir,
        classification_jsons=flags.classification_jsons,
        dataset_csv=flags.dataset_csv,
        rq3_paired_csv=flags.rq3_paired_csv,
        rq3_complexity_csv=flags.rq3_complexity_csv,
        plots=flags.plots,
        correlations=flags.correlations,
        config=config,
    )


if __name__ == "__main__":
    import tyro

    parsed_flags = tyro.cli(QuantFlags)
    main(parsed_flags)
