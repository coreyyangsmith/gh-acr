"""Orchestration script: generate all final paper figures and tables.

Produces outputs in results/final_paper_figs/:
  - Table_I_dataset_summary.csv
  - Table_III_fail_set_overview.csv
  - Table_IV_performance_metrics.csv
  - Figure_B_bypass_advantage.pdf / .png
  - Figure_C_decision_outcomes.pdf / .png
  - Figure_D_advantage_by_buckets.pdf / .png

Usage::

    python -m src.analysis.final_paper_figs.run_all

Or with custom paths::

    python -m src.analysis.final_paper_figs.run_all \\
        --results-csv data/2026_01_results_final.csv \\
        --dataset-csv data/git_good_bench_merge_commits_all.csv \\
        --output-dir  results/final_paper_figs
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.analysis.final_paper_figs.shared import (
    DATASET_CSV,
    FAIL_ONLY_AGGREGATE_CSV,
    OUTPUT_DIR,
    RESULTS_CSV,
    logger,
)
from src.analysis.final_paper_figs.table_1_dataset_summary import generate_table_1
from src.analysis.final_paper_figs.table_3_fail_set import generate_table_3
from src.analysis.final_paper_figs.table_4_performance import generate_table_4
from src.analysis.final_paper_figs.figure_b_bypass_advantage import generate_figure_b
from src.analysis.final_paper_figs.figure_c_decision_outcomes import generate_figure_c
from src.analysis.final_paper_figs.figure_d_advantage_by_buckets import generate_figure_d

def run_all(
    results_csv: Path | None = None,
    dataset_csv: Path | None = None,
    fail_aggregate_csv: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate all final paper tables and figures."""
    results_csv = results_csv or RESULTS_CSV
    dataset_csv = dataset_csv or DATASET_CSV
    fail_aggregate_csv = fail_aggregate_csv or FAIL_ONLY_AGGREGATE_CSV
    output_dir = output_dir or OUTPUT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Generating all final paper figures and tables")
    logger.info(f"  Results CSV  : {results_csv}")
    logger.info(f"  Dataset CSV  : {dataset_csv}")
    logger.info(f"  Fail agg CSV : {fail_aggregate_csv}")
    logger.info(f"  Output dir   : {output_dir}")
    logger.info("=" * 60)

    # ── Tables ────────────────────────────────────────────────────────
    logger.info("\n--- TABLES ---")
    generate_table_1(dataset_csv=dataset_csv, output_dir=output_dir)
    generate_table_3(
        results_csv=results_csv,
        output_dir=output_dir,
    )
    generate_table_4(results_csv=results_csv, output_dir=output_dir)

    # ── Figures ───────────────────────────────────────────────────────
    logger.info("\n--- FIGURES ---")
    generate_figure_b(results_csv=results_csv, output_dir=output_dir)
    generate_figure_c(results_csv=results_csv, output_dir=output_dir)
    generate_figure_d(
        results_csv=results_csv,
        dataset_csv=dataset_csv,
        output_dir=output_dir,
    )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"All outputs saved to {output_dir}  ({elapsed:.1f}s)")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Simple arg parsing
    args = sys.argv[1:]
    kwargs: dict[str, Path] = {}
    i = 0
    while i < len(args):
        if args[i] == "--results-csv" and i + 1 < len(args):
            kwargs["results_csv"] = Path(args[i + 1])
            i += 2
        elif args[i] == "--dataset-csv" and i + 1 < len(args):
            kwargs["dataset_csv"] = Path(args[i + 1])
            i += 2
        elif args[i] == "--fail-aggregate-csv" and i + 1 < len(args):
            kwargs["fail_aggregate_csv"] = Path(args[i + 1])
            i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            kwargs["output_dir"] = Path(args[i + 1])
            i += 2
        else:
            i += 1

    run_all(**kwargs)
