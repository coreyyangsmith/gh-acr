param(
    [string]$ResultsCsv = "data/2026_01_results_final.csv",
    [string]$DatasetCsv = "data/git_good_bench_merge_commits_all.csv",
    [string]$OutputDir = "results/final_paper_figs"
)

$ErrorActionPreference = "Stop"

$code = @"
from pathlib import Path
from src.analysis.final_paper_figs.figure_d_advantage_by_buckets import generate_figure_d

generate_figure_d(
    results_csv=Path(r'$ResultsCsv'),
    dataset_csv=Path(r'$DatasetCsv'),
    output_dir=Path(r'$OutputDir'),
)
"@

uv run python -c $code
