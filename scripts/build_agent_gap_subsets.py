"""Build benchmark subset CSVs from agent coverage gap files.

By default uses the full gap list
(``data/agent_coverage_gaps/<model>_agent_missing_or_failed.csv``), which
includes:

- not_processed / empty_output / incomplete / invalid_output
- missing_results (artifacts present but absent from results CSV)

Writes filtered copies of ``data/git_good_bench_merge_commits_all.csv``
suitable for ``DATASET_CSV=... python -m src.cli.run_all``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENCH = Path("data/git_good_bench_merge_commits_all.csv")
GAPS_DIR = Path("data/agent_coverage_gaps")
OUT_DIR = Path("data/agent_coverage_gaps/subsets")

MODELS = {
    "gpt-5-nano": "gpt-5-nano",
    "llama-3.1-8b": "llama-3.1-8b",
    "qwen3-32b": "qwen3-32b",
}

# Canonical subset filename used by run scripts (always the full gap set).
CANONICAL_SUFFIX = "all_gaps"


def build_subset(
    *,
    model_label: str,
    source: str = "all_gaps",
) -> Path:
    """Filter the full benchmark to gap scenario IDs for one model."""
    if source == "needs_reprocess":
        gaps_path = GAPS_DIR / f"{model_label}_agent_needs_reprocess.csv"
    elif source == "all_gaps":
        gaps_path = GAPS_DIR / f"{model_label}_agent_missing_or_failed.csv"
    else:
        raise ValueError(f"Unknown source={source!r}")

    if not gaps_path.exists():
        raise FileNotFoundError(gaps_path)

    gaps = pd.read_csv(gaps_path)
    if gaps.empty:
        raise ValueError(f"No rows in {gaps_path}")

    ids = set(gaps["scenario_id"].astype(str))
    bench = pd.read_csv(BENCH, index_col=0)
    subset = bench.loc[bench.index.astype(str).isin(ids)].copy()

    missing = ids - set(subset.index.astype(str))
    if missing:
        raise RuntimeError(
            f"{model_label}: {len(missing)} gap IDs not found in benchmark "
            f"(examples: {sorted(missing)[:5]})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{model_label}_agent_gaps_{source}.csv"
    subset.to_csv(out)

    # Also write/refresh the canonical path run scripts expect.
    if source == CANONICAL_SUFFIX:
        canonical = OUT_DIR / f"{model_label}_agent_gaps.csv"
        subset.to_csv(canonical)
        print(
            f"{model_label}: wrote {len(subset)} scenarios -> {out} "
            f"(and {canonical.name}; from {gaps_path.name}, "
            f"categories={dict(gaps['gap_category'].value_counts())})"
        )
    else:
        print(
            f"{model_label}: wrote {len(subset)} scenarios -> {out} "
            f"(from {gaps_path.name}, categories={dict(gaps['gap_category'].value_counts())})"
        )
    return out


def main() -> None:
    for label in MODELS:
        build_subset(model_label=label, source="all_gaps")
        # Keep the narrower file around for cost estimates / comparison.
        build_subset(model_label=label, source="needs_reprocess")


if __name__ == "__main__":
    main()
