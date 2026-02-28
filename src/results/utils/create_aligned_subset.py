"""Create an aligned subset of results for fair comparison.

This script finds the intersection of scenarios across all methods
(base_a, base_b, agent, bypass7 for each model) and creates a filtered
dataset where all methods are evaluated on the exact same scenarios.

Usage:
    python -m src.results.utils.create_aligned_subset \
        --input-csv data/2026_01_results_combined.csv \
        --output-csv data/2026_01_results_aligned.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AlignmentReport:
    """Report on the alignment process."""
    original_rows: int
    aligned_rows: int
    original_scenarios: dict[str, int]  # method -> count
    aligned_scenarios: int
    models_found: list[str]
    overlap_stats: dict[str, float]


def get_scenario_keys(df: pd.DataFrame, eval_method: str, model_name: str | None = None) -> set[tuple]:
    """Get set of (id, file_name) tuples for a method."""
    mask = df["eval_method"] == eval_method
    if model_name:
        mask &= df["model_name"] == model_name
    subset = df[mask]
    return set(zip(subset["id"], subset["file_name"]))


def find_common_scenarios(
    df: pd.DataFrame,
    require_baselines: bool = True,
    require_all_models: bool = True,
) -> tuple[set[tuple], dict[str, set[tuple]]]:
    """Find scenarios that exist across all required methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    require_baselines : bool
        Require scenarios to have base_a and base_b results
    require_all_models : bool
        Require scenarios to have results for all models
        
    Returns
    -------
    common_scenarios : set[tuple]
        Set of (id, file_name) tuples present in all required methods
    all_scenario_sets : dict[str, set[tuple]]
        Individual scenario sets for each method
    """
    all_scenario_sets: dict[str, set[tuple]] = {}
    
    # Get baseline scenarios
    if require_baselines:
        all_scenario_sets["base_a"] = get_scenario_keys(df, "base_a")
        all_scenario_sets["base_b"] = get_scenario_keys(df, "base_b")
    
    # Get model scenarios
    models = df[df["model_name"].notna()]["model_name"].unique().tolist()
    
    for model in models:
        # Single-agent
        agent_key = f"agent:{model}"
        agent_scenarios = get_scenario_keys(df, "agent", model)
        if agent_scenarios:
            all_scenario_sets[agent_key] = agent_scenarios
        
        # Multi-agent (bypass7)
        bypass_key = f"bypass7:{model}"
        bypass_scenarios = get_scenario_keys(df, "bypass7", model)
        if bypass_scenarios:
            all_scenario_sets[bypass_key] = bypass_scenarios
    
    # Find intersection
    if not all_scenario_sets:
        return set(), all_scenario_sets
    
    sets_to_intersect = list(all_scenario_sets.values())
    common_scenarios = sets_to_intersect[0].copy()
    
    for s in sets_to_intersect[1:]:
        common_scenarios &= s
    
    return common_scenarios, all_scenario_sets


def create_aligned_dataset(
    df: pd.DataFrame,
    common_scenarios: set[tuple],
) -> pd.DataFrame:
    """Filter dataframe to only include common scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    common_scenarios : set[tuple]
        Set of (id, file_name) tuples to keep
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    # Create scenario key column
    df_copy = df.copy()
    df_copy["_scenario_key"] = list(zip(df_copy["id"], df_copy["file_name"]))
    
    # Filter to common scenarios
    aligned = df_copy[df_copy["_scenario_key"].isin(common_scenarios)].copy()
    
    # Remove temporary column
    aligned = aligned.drop(columns=["_scenario_key"])
    
    return aligned


def generate_alignment_report(
    original_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    common_scenarios: set[tuple],
    all_scenario_sets: dict[str, set[tuple]],
) -> AlignmentReport:
    """Generate a report on the alignment process."""
    # Original scenario counts
    original_scenarios = {}
    for key, scenarios in all_scenario_sets.items():
        original_scenarios[key] = len(scenarios)
    
    # Get models
    models = aligned_df[aligned_df["model_name"].notna()]["model_name"].unique().tolist()
    
    # Compute overlap statistics
    overlap_stats = {}
    for key, scenarios in all_scenario_sets.items():
        if scenarios:
            overlap = len(common_scenarios) / len(scenarios) * 100
            overlap_stats[key] = overlap
    
    return AlignmentReport(
        original_rows=len(original_df),
        aligned_rows=len(aligned_df),
        original_scenarios=original_scenarios,
        aligned_scenarios=len(common_scenarios),
        models_found=models,
        overlap_stats=overlap_stats,
    )


def print_report(report: AlignmentReport) -> None:
    """Print the alignment report."""
    print("=" * 70)
    print("ALIGNMENT REPORT")
    print("=" * 70)
    
    print(f"\nOriginal dataset: {report.original_rows:,} rows")
    print(f"Aligned dataset:  {report.aligned_rows:,} rows")
    
    print(f"\n{'Method':<50} {'Original':>10} {'Aligned':>10} {'Overlap %':>10}")
    print("-" * 80)
    
    for key, count in sorted(report.original_scenarios.items()):
        overlap = report.overlap_stats.get(key, 0)
        print(f"{key:<50} {count:>10,} {report.aligned_scenarios:>10,} {overlap:>9.1f}%")
    
    print(f"\nCommon scenarios across all methods: {report.aligned_scenarios:,}")
    print(f"Models in aligned dataset: {', '.join(report.models_found)}")
    
    # Compute reduction
    if report.original_rows > 0:
        reduction = (1 - report.aligned_rows / report.original_rows) * 100
        print(f"\nData reduction: {reduction:.1f}%")


def analyze_aligned_baseline_rates(aligned_df: pd.DataFrame) -> None:
    """Analyze baseline rates on the aligned dataset."""
    print("\n" + "=" * 70)
    print("BASELINE RATES ON ALIGNED DATASET")
    print("=" * 70)
    
    # Baseline rates
    for baseline in ["base_a", "base_b"]:
        baseline_df = aligned_df[aligned_df["eval_method"] == baseline]
        if len(baseline_df) > 0:
            em_rate = baseline_df["exact_match"].sum() / len(baseline_df) * 100
            sim_mean = baseline_df["similarity"].mean()
            print(f"\n{baseline}:")
            print(f"  Exact Match: {em_rate:.1f}%")
            print(f"  Similarity:  {sim_mean:.3f}")
    
    # Model rates
    print("\n" + "-" * 70)
    print("MODEL RATES ON ALIGNED DATASET")
    print("-" * 70)
    
    models = aligned_df[aligned_df["model_name"].notna()]["model_name"].unique()
    
    for model in sorted(models):
        print(f"\n{model}:")
        
        # Single-agent
        agent_df = aligned_df[(aligned_df["eval_method"] == "agent") & 
                              (aligned_df["model_name"] == model)]
        if len(agent_df) > 0:
            em_rate = agent_df["exact_match"].sum() / len(agent_df) * 100
            sim_mean = agent_df["similarity"].mean()
            print(f"  Single-Agent: {em_rate:.1f}% EM, {sim_mean:.3f} Sim (n={len(agent_df):,})")
        
        # Multi-agent
        bypass_df = aligned_df[(aligned_df["eval_method"] == "bypass7") & 
                               (aligned_df["model_name"] == model)]
        if len(bypass_df) > 0:
            em_rate = bypass_df["exact_match"].sum() / len(bypass_df) * 100
            sim_mean = bypass_df["similarity"].mean()
            
            # Bypass distribution
            a_pct = (bypass_df["bypass_method"] == "A").sum() / len(bypass_df) * 100
            b_pct = (bypass_df["bypass_method"] == "B").sum() / len(bypass_df) * 100
            mix_pct = (bypass_df["bypass_method"] == "MIX").sum() / len(bypass_df) * 100
            
            print(f"  Multi-Agent:  {em_rate:.1f}% EM, {sim_mean:.3f} Sim (n={len(bypass_df):,})")
            print(f"    Bypass: A={a_pct:.1f}%, B={b_pct:.1f}%, MIX={mix_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Create aligned subset of results for fair comparison"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to input results CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output aligned CSV (optional, skips if not provided)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report, don't save aligned dataset",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} rows")
    
    # Find common scenarios
    print("\nFinding common scenarios across all methods...")
    common_scenarios, all_scenario_sets = find_common_scenarios(df)
    
    if not common_scenarios:
        print("ERROR: No common scenarios found!")
        return
    
    # Create aligned dataset
    print(f"Creating aligned dataset with {len(common_scenarios):,} common scenarios...")
    aligned_df = create_aligned_dataset(df, common_scenarios)
    
    # Generate and print report
    report = generate_alignment_report(df, aligned_df, common_scenarios, all_scenario_sets)
    print_report(report)
    
    # Analyze rates on aligned dataset
    analyze_aligned_baseline_rates(aligned_df)
    
    # Save aligned dataset
    if args.output_csv and not args.report_only:
        print(f"\nSaving aligned dataset to {args.output_csv}...")
        aligned_df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(aligned_df):,} rows")
    elif not args.report_only:
        # Default output path
        output_path = args.input_csv.parent / f"{args.input_csv.stem}_aligned{args.input_csv.suffix}"
        print(f"\nSaving aligned dataset to {output_path}...")
        aligned_df.to_csv(output_path, index=False)
        print(f"Saved {len(aligned_df):,} rows")


if __name__ == "__main__":
    main()
