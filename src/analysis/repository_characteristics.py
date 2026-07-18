"""Summarize repository characteristics for evaluated repositories.

This module keeps the analysis offline and reproducible: it uses the final
results CSV to define the evaluated repository set, then joins against the
GitGoodBench benchmark CSV for repository metadata.

Usage::

    python -m src.analysis.repository_characteristics
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tyro


DEFAULT_RESULTS_CSV = Path("data/2026_01_results_final.csv")
DEFAULT_DATASET_CSV = Path("data/git_good_bench_merge_commits_all.csv")
DEFAULT_OUTPUT_DIR = Path("results/repository_characteristics")

REPO_METADATA_COLUMNS = [
    "default_branch",
    "license",
    "stargazers",
    "created_at",
    "topics",
    "programming_language",
    "commits",
    "branches",
    "releases",
    "forks",
    "watchers",
    "contributors",
    "blank_lines",
    "code_lines",
    "comment_lines",
    "last_commit",
]

NUMERIC_REPO_COLUMNS = [
    "stargazers",
    "commits",
    "branches",
    "releases",
    "forks",
    "watchers",
    "contributors",
    "blank_lines",
    "code_lines",
    "comment_lines",
]

COUNT_PREFIXES = {
    "difficulty": "difficulty",
    "project_size": "project_size",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepoCharacteristicsFlags:
    """CLI flags for repository-characteristics analysis."""

    results_csv: Path = DEFAULT_RESULTS_CSV
    dataset_csv: Path = DEFAULT_DATASET_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _normalize_repo(value: object) -> str:
    """Normalize repository slugs for joins while preserving owner/repo shape."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {path}: {missing}")


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _parse_scenario(raw: object) -> dict[str, Any]:
    if pd.isna(raw):
        return {}
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _mode_or_empty(series: pd.Series) -> str:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    return str(counts.index[0])


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _load_inputs(results_csv: Path, dataset_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(results_csv)
    dataset = pd.read_csv(dataset_csv)
    _require_columns(results, ["repo", "id"], results_csv)
    _require_columns(dataset, ["name"], dataset_csv)

    results = results.copy()
    dataset = dataset.copy()
    results["repo_norm"] = results["repo"].map(_normalize_repo)
    dataset["repo_norm"] = dataset["name"].map(_normalize_repo)

    results = results[results["repo_norm"] != ""]
    dataset = dataset[dataset["repo_norm"] != ""]
    dataset = _coerce_numeric(dataset, NUMERIC_REPO_COLUMNS)
    return results, dataset


def _coverage_by_repo(results: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        results.groupby("repo_norm", as_index=False)
        .agg(
            repo=("repo", _first_non_null),
            evaluated_row_count=("repo", "size"),
            evaluated_instance_count=("id", lambda s: s.astype(str).nunique()),
        )
    )
    return coverage


def _scenario_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    if "scenario" in out.columns:
        scenario = out["scenario"].map(_parse_scenario)
        out["n_conflict_files"] = scenario.map(
            lambda item: item.get("number_of_files_with_merge_conflict", 0)
        )
        out["n_total_conflicts"] = scenario.map(
            lambda item: item.get("total_number_of_merge_conflicts", 0)
        )
        out["merge_parent_count"] = scenario.map(
            lambda item: len(item.get("parents", []))
            if isinstance(item.get("parents", []), list)
            else 0
        )
    else:
        out["n_conflict_files"] = 0
        out["n_total_conflicts"] = 0
        out["merge_parent_count"] = 0

    for col in ["n_conflict_files", "n_total_conflicts", "merge_parent_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def _count_columns(grouped: pd.core.groupby.DataFrameGroupBy, column: str) -> pd.DataFrame:
    if column not in grouped.obj.columns:
        return pd.DataFrame({"repo_norm": grouped.size().index})

    counts = grouped[column].value_counts(dropna=True).unstack(fill_value=0)
    counts.columns = [
        f"{COUNT_PREFIXES[column]}_{str(col).strip().lower()}_scenario_count"
        for col in counts.columns
    ]
    return counts.reset_index()


def _metadata_by_repo(dataset: pd.DataFrame) -> pd.DataFrame:
    enriched = _scenario_columns(dataset)
    grouped = enriched.groupby("repo_norm", dropna=False)

    agg_spec: dict[str, tuple[str, Any]] = {
        "benchmark_scenario_count": ("repo_norm", "size"),
        "conflict_files_total": ("n_conflict_files", "sum"),
        "conflict_files_median": ("n_conflict_files", "median"),
        "merge_conflicts_total": ("n_total_conflicts", "sum"),
        "merge_conflicts_median": ("n_total_conflicts", "median"),
        "merge_parent_count_median": ("merge_parent_count", "median"),
    }

    for col in REPO_METADATA_COLUMNS:
        if col in enriched.columns:
            agg_spec[col] = (col, _first_non_null)

    if "difficulty" in enriched.columns:
        agg_spec["dominant_difficulty"] = ("difficulty", _mode_or_empty)
    if "project_size" in enriched.columns:
        agg_spec["dominant_project_size"] = ("project_size", _mode_or_empty)

    metadata = grouped.agg(**agg_spec).reset_index()

    for count_col in ["difficulty", "project_size"]:
        counts = _count_columns(grouped, count_col)
        metadata = metadata.merge(counts, on="repo_norm", how="left")

    count_cols = [
        col
        for col in metadata.columns
        if col.startswith("difficulty_") or col.startswith("project_size_")
    ]
    metadata[count_cols] = metadata[count_cols].fillna(0).astype(int)
    return metadata


def _add_derived_metrics(repos: pd.DataFrame) -> pd.DataFrame:
    out = repos.copy()
    for col in ["created_at", "last_commit"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    if "last_commit" in out.columns:
        snapshot_date = out["last_commit"].max()
    else:
        snapshot_date = pd.NaT

    if "created_at" in out.columns and "last_commit" in out.columns:
        active_span = out["last_commit"] - out["created_at"]
        out["active_span_days"] = active_span.dt.days
        out["active_span_years"] = out["active_span_days"] / 365.25
    else:
        out["active_span_days"] = pd.NA
        out["active_span_years"] = pd.NA

    if "created_at" in out.columns and pd.notna(snapshot_date):
        repo_age = snapshot_date - out["created_at"]
        out["repository_age_days"] = repo_age.dt.days
        out["repository_age_years"] = out["repository_age_days"] / 365.25
    else:
        out["repository_age_days"] = pd.NA
        out["repository_age_years"] = pd.NA

    if "commits" in out.columns:
        out["commits_per_year"] = _safe_divide(out["commits"], out["active_span_years"])
    if "code_lines" in out.columns and "contributors" in out.columns:
        out["code_lines_per_contributor"] = _safe_divide(
            out["code_lines"], out["contributors"]
        )
    if "stargazers" in out.columns:
        out["stars_per_year"] = _safe_divide(
            out["stargazers"], out["repository_age_years"]
        )

    for col in ["created_at", "last_commit"]:
        if col in out.columns:
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return out


def build_repository_characteristics(
    results_csv: Path = DEFAULT_RESULTS_CSV,
    dataset_csv: Path = DEFAULT_DATASET_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-repository characteristics and missing-metadata tables."""
    results, dataset = _load_inputs(results_csv, dataset_csv)
    coverage = _coverage_by_repo(results)
    evaluated_repos = set(coverage["repo_norm"])

    filtered_dataset = dataset[dataset["repo_norm"].isin(evaluated_repos)].copy()
    metadata = _metadata_by_repo(filtered_dataset)

    repos = coverage.merge(metadata, on="repo_norm", how="left")
    repos = _add_derived_metrics(repos)

    missing = repos[repos["benchmark_scenario_count"].isna()][
        ["repo", "repo_norm", "evaluated_row_count", "evaluated_instance_count"]
    ].copy()

    repos["metadata_matched"] = repos["benchmark_scenario_count"].notna()
    numeric_fill_cols = [
        "benchmark_scenario_count",
        "conflict_files_total",
        "conflict_files_median",
        "merge_conflicts_total",
        "merge_conflicts_median",
        "merge_parent_count_median",
    ]
    for col in numeric_fill_cols:
        if col in repos.columns:
            repos[col] = repos[col].fillna(0)

    repos = repos.sort_values("repo", key=lambda s: s.str.lower()).reset_index(drop=True)
    missing = missing.sort_values("repo", key=lambda s: s.str.lower()).reset_index(drop=True)
    return repos, missing


def _numeric_summary(repos: pd.DataFrame) -> pd.DataFrame:
    numeric = repos.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        values = numeric[col].dropna()
        rows.append(
            {
                "metric": col,
                "count": int(values.count()),
                "min": values.min() if not values.empty else pd.NA,
                "median": values.median() if not values.empty else pd.NA,
                "mean": values.mean() if not values.empty else pd.NA,
                "max": values.max() if not values.empty else pd.NA,
                "sum": values.sum() if not values.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _group_summary(repos: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in repos.columns:
        return pd.DataFrame()

    grouped = repos.copy()
    grouped[group_col] = grouped[group_col].fillna("unknown").astype(str).str.strip()
    grouped.loc[grouped[group_col] == "", group_col] = "unknown"
    summary = (
        grouped.groupby(group_col, as_index=False)
        .agg(
            repo_count=("repo", "nunique"),
            evaluated_row_count=("evaluated_row_count", "sum"),
            evaluated_instance_count=("evaluated_instance_count", "sum"),
            benchmark_scenario_count=("benchmark_scenario_count", "sum"),
            stargazers_median=("stargazers", "median"),
            stargazers_total=("stargazers", "sum"),
            forks_median=("forks", "median"),
            forks_total=("forks", "sum"),
            commits_median=("commits", "median"),
            commits_total=("commits", "sum"),
            contributors_median=("contributors", "median"),
            contributors_total=("contributors", "sum"),
            code_lines_median=("code_lines", "median"),
            code_lines_total=("code_lines", "sum"),
        )
        .sort_values(["repo_count", group_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def write_outputs(
    repos: pd.DataFrame,
    missing: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all repository-characteristics artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "repository_characteristics": output_dir / "repository_characteristics.csv",
        "repository_characteristics_summary": output_dir
        / "repository_characteristics_summary.csv",
        "repository_characteristics_by_language": output_dir
        / "repository_characteristics_by_language.csv",
        "repository_characteristics_by_size": output_dir
        / "repository_characteristics_by_size.csv",
        "missing_metadata_repositories": output_dir / "missing_metadata_repositories.csv",
    }

    repos.to_csv(outputs["repository_characteristics"], index=False)
    _numeric_summary(repos).to_csv(
        outputs["repository_characteristics_summary"], index=False
    )
    _group_summary(repos, "programming_language").to_csv(
        outputs["repository_characteristics_by_language"], index=False
    )
    _group_summary(repos, "dominant_project_size").to_csv(
        outputs["repository_characteristics_by_size"], index=False
    )
    missing.to_csv(outputs["missing_metadata_repositories"], index=False)
    return outputs


def generate_repository_characteristics(
    results_csv: Path = DEFAULT_RESULTS_CSV,
    dataset_csv: Path = DEFAULT_DATASET_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate repository-characteristics CSVs and print a compact summary."""
    repos, missing = build_repository_characteristics(results_csv, dataset_csv)
    outputs = write_outputs(repos, missing, output_dir)

    matched = int(repos["metadata_matched"].sum())
    summary = {
        "results_csv": str(results_csv),
        "dataset_csv": str(dataset_csv),
        "output_dir": str(output_dir),
        "evaluated_repositories": int(repos["repo"].nunique()),
        "metadata_matched_repositories": matched,
        "missing_metadata_repositories": int(len(missing)),
    }
    print(summary)
    for name, path in outputs.items():
        logger.info("Saved %s: %s", name, path)
    return outputs


def main(flags: RepoCharacteristicsFlags) -> None:
    generate_repository_characteristics(
        results_csv=flags.results_csv,
        dataset_csv=flags.dataset_csv,
        output_dir=flags.output_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(RepoCharacteristicsFlags))
