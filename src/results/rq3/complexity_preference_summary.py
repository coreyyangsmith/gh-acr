"""Summarize RQ3 complexity-preference labels.

This script resolves the contradictory "favored simplicity" vs
"favored complexity" labels at the instance level using a majority vote
over file-level label counts from ``aggregate_combined.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_AGG_CSV = Path("results/rq3/aggregate_combined.csv")
DEFAULT_CLASSIFICATION_JSON_DIR = Path("data/labeling_results")
DEFAULT_OUTPUT_CSV = Path("results/rq3/complexity_preference_summary.csv")

SIMPLICITY_COL = "favored_simplicity_count"
COMPLEXITY_COL = "favored_complexity_count"
LABEL_MAPPINGS = {
    "fewer changes": "favored simplicity",
    "fewer-changes": "favored simplicity",
}


def resolve_dominant_preference(row: pd.Series) -> str:
    """Resolve the dominant complexity preference for one instance."""
    simplicity_count = int(row["simplicity_files"])
    complexity_count = int(row["complexity_files"])

    if simplicity_count == 0 and complexity_count == 0:
        return "neither"
    if simplicity_count > complexity_count:
        return "favored simplicity"
    if complexity_count > simplicity_count:
        return "favored complexity"
    return "tie (ambiguous)"


def extract_base_id(full_id: str) -> str:
    """Extract the instance ID from a file-level classification ID."""
    parts = full_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return full_id


def normalize_label(label: str) -> str:
    """Normalize raw labels to the canonical labels used in RQ3."""
    normalized = label.lower().strip()
    return LABEL_MAPPINGS.get(normalized, normalized)


def load_aggregate_csv(path: Path) -> pd.DataFrame:
    """Load and validate the aggregate RQ3 CSV."""
    df = pd.read_csv(path)
    required_columns = {
        "sample_id",
        "file_count",
        "source_file",
        "favored_simplicity",
        "favored_complexity",
        SIMPLICITY_COL,
        COMPLEXITY_COL,
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required columns in {path}: {joined}")
    return df


def find_classification_jsons(directory: Path) -> list[Path]:
    """Find the top-level classification JSON files used by RQ3."""
    return sorted(directory.glob("*-classifications.json"))


def filter_jsons_to_aggregate_sources(
    paths: list[Path],
    aggregate_df: pd.DataFrame,
) -> list[Path]:
    """Keep only JSON files represented in the aggregate CSV."""
    source_files = set(aggregate_df["source_file"].astype(str))
    filtered = [path for path in paths if path.stem in source_files]
    missing_sources = source_files.difference(path.stem for path in filtered)
    if missing_sources:
        joined = ", ".join(sorted(missing_sources))
        raise ValueError(f"Could not find classification JSONs for aggregate sources: {joined}")
    return filtered


def build_instance_summary_from_jsons(paths: list[Path]) -> pd.DataFrame:
    """Build one row per instance from raw file-level classification JSONs."""
    rows: dict[tuple[str, str], dict[str, object]] = {}

    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        classifications = data.get("classifications", {})
        source_file = path.stem

        for full_id, labels in classifications.items():
            sample_id = extract_base_id(full_id)
            key = (source_file, sample_id)
            row = rows.setdefault(
                key,
                {
                    "sample_id": sample_id,
                    "file_count": 0,
                    "simplicity_files": 0,
                    "complexity_files": 0,
                    "complexity_preference_files": 0,
                    "both_preference_labels_files": 0,
                    "source_file": source_file,
                },
            )

            normalized_labels = {normalize_label(label) for label in labels}
            has_simplicity = "favored simplicity" in normalized_labels
            has_complexity = "favored complexity" in normalized_labels

            row["file_count"] = int(row["file_count"]) + 1
            row["simplicity_files"] = int(row["simplicity_files"]) + int(has_simplicity)
            row["complexity_files"] = int(row["complexity_files"]) + int(has_complexity)
            row["complexity_preference_files"] = int(row["complexity_preference_files"]) + int(
                has_simplicity or has_complexity
            )
            row["both_preference_labels_files"] = int(row["both_preference_labels_files"]) + int(
                has_simplicity and has_complexity
            )

    if not rows:
        raise ValueError("No classification entries found in the provided JSON files.")

    summary = pd.DataFrame(rows.values())
    summary["dominant_preference"] = summary.apply(resolve_dominant_preference, axis=1)
    return summary[
        [
            "sample_id",
            "file_count",
            "simplicity_files",
            "complexity_files",
            "complexity_preference_files",
            "both_preference_labels_files",
            "dominant_preference",
            "source_file",
        ]
    ]


def validate_against_aggregate(instance_summary: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    """Validate that raw JSON-derived rows align with the aggregate CSV."""
    if len(instance_summary) != len(aggregate_df):
        raise ValueError(
            "Classification JSON row count does not match aggregate CSV row count: "
            f"{len(instance_summary)} vs {len(aggregate_df)}"
        )

    raw_files = int(instance_summary["file_count"].sum())
    aggregate_files = int(aggregate_df["file_count"].sum())
    if raw_files != aggregate_files:
        raise ValueError(
            "Classification JSON file count does not match aggregate CSV file count: "
            f"{raw_files} vs {aggregate_files}"
        )


def format_count_pct(count: int, denominator: int) -> str:
    """Format a count and percentage against a denominator."""
    pct = 100 * count / denominator if denominator else 0
    return f"{count:,} ({pct:.1f}%)"


def print_summary(instance_summary: pd.DataFrame) -> None:
    """Print file-level and instance-level complexity-preference summaries."""
    total_instances = len(instance_summary)
    total_files = int(instance_summary["file_count"].sum())
    simplicity_files = int(instance_summary["simplicity_files"].sum())
    complexity_files = int(instance_summary["complexity_files"].sum())
    complexity_preference_files = int(instance_summary["complexity_preference_files"].sum())
    both_preference_files = int(instance_summary["both_preference_labels_files"].sum())
    neither_files = total_files - complexity_preference_files

    has_simplicity = int((instance_summary["simplicity_files"] > 0).sum())
    has_complexity = int((instance_summary["complexity_files"] > 0).sum())
    has_both = int(
        (
            (instance_summary["simplicity_files"] > 0)
            & (instance_summary["complexity_files"] > 0)
        ).sum()
    )
    has_neither = int(
        (
            (instance_summary["simplicity_files"] == 0)
            & (instance_summary["complexity_files"] == 0)
        ).sum()
    )

    dominant_counts = instance_summary["dominant_preference"].value_counts()
    dominant_simplicity = int(dominant_counts.get("favored simplicity", 0))
    dominant_complexity = int(dominant_counts.get("favored complexity", 0))
    tie_ambiguous = int(dominant_counts.get("tie (ambiguous)", 0))
    dominant_neither = int(dominant_counts.get("neither", 0))

    resolved_total = dominant_simplicity + dominant_complexity

    print("RQ3 Complexity Preference Summary")
    print("=" * 33)
    print(f"Aggregate input instances: {total_instances:,}")
    print()

    print("A. File-level summary")
    print(f"  Total files coded: {total_files:,}")
    print(f"  Files with favored simplicity label: {simplicity_files:,}")
    print(f"  Files with favored complexity label: {complexity_files:,}")
    print(f"  Files with either complexity-preference label: {complexity_preference_files:,}")
    print(f"  Files with both preference labels: {both_preference_files:,}")
    print(f"  Files with neither preference label: {neither_files:,}")
    print()

    print("B. Instance-level raw counts before majority vote")
    print(f"  Has simplicity label in any file: {format_count_pct(has_simplicity, total_instances)}")
    print(f"  Has complexity label in any file: {format_count_pct(has_complexity, total_instances)}")
    print(f"  Has both labels across files: {format_count_pct(has_both, total_instances)}")
    print(f"  Has neither label: {format_count_pct(has_neither, total_instances)}")
    print()

    print("C. Instance-level majority-vote resolution")
    print(f"  Dominant favored simplicity: {format_count_pct(dominant_simplicity, total_instances)}")
    print(f"  Dominant favored complexity: {format_count_pct(dominant_complexity, total_instances)}")
    print(f"  Tie / ambiguous: {format_count_pct(tie_ambiguous, total_instances)}")
    print(f"  Neither: {format_count_pct(dominant_neither, total_instances)}")
    print()

    print("D. Coverage among resolved classifiable instances")
    print(f"  Resolved classifiable instances: {resolved_total:,}")
    print(f"  Favored simplicity coverage: {format_count_pct(dominant_simplicity, resolved_total)}")
    print(f"  Favored complexity coverage: {format_count_pct(dominant_complexity, resolved_total)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize RQ3 favored simplicity vs favored complexity labels.",
    )
    parser.add_argument(
        "--agg-csv",
        type=Path,
        default=DEFAULT_AGG_CSV,
        help=f"Path to aggregate RQ3 CSV for validation. Default: {DEFAULT_AGG_CSV}",
    )
    parser.add_argument(
        "--classification-json-dir",
        type=Path,
        default=DEFAULT_CLASSIFICATION_JSON_DIR,
        help=(
            "Directory containing top-level *-classifications.json files. "
            f"Default: {DEFAULT_CLASSIFICATION_JSON_DIR}"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path for per-instance output CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the summary calculation."""
    args = parse_args()
    aggregate_df = load_aggregate_csv(args.agg_csv)
    classification_jsons = find_classification_jsons(args.classification_json_dir)
    if not classification_jsons:
        raise ValueError(f"No *-classifications.json files found in {args.classification_json_dir}")
    classification_jsons = filter_jsons_to_aggregate_sources(classification_jsons, aggregate_df)

    instance_summary = build_instance_summary_from_jsons(classification_jsons)
    validate_against_aggregate(instance_summary, aggregate_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    instance_summary.to_csv(args.output_csv, index=False)

    print_summary(instance_summary)
    print()
    print(f"Wrote per-instance summary: {args.output_csv}")


if __name__ == "__main__":
    main()
