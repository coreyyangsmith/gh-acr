"""Analyze Summary Agent artifacts across raw model outputs.

This complements ``summary_agent_preference_analysis.py`` by reading every
paired summarizer A/B output under ``data/raw_model_outputs`` (new layout:
``<method>/<file_slug>/summarizer/{a,b}/output.txt``) instead of only the
curated labeling-results tree. It reports both all available summary
artifacts and the subset whose joined result metadata indicates at least one
exact-match failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .summary_agent_preference_analysis import (
    PARENT_LABELS,
    _as_bool,
    _format_count_pct,
    _model_family_from_text,
    add_selection_columns,
    build_type_counts,
    load_result_metadata,
    majority_vote_label,
    parse_summary_file,
)


DEFAULT_RAW_ARTIFACT_ROOT = Path("data/raw_model_outputs")
DEFAULT_RESULTS_CSV = Path("results/figures/force_mix_vs_final_combined.csv")
DEFAULT_EM_DATASET_ROOT = Path("results/em_datasets")
DEFAULT_OUTPUT_DIR = Path("results/rq3/raw_summary_agent_preference")

OUTPUT_FILE_LEVEL = "raw_summary_agent_preference_file_level.csv"
OUTPUT_INSTANCE_LEVEL = "raw_summary_agent_preference_instance_level.csv"
OUTPUT_TYPE_COUNTS = "raw_summary_agent_preference_type_counts.csv"
OUTPUT_TYPE_DISTRIBUTION = "raw_summary_agent_preference_type_distribution.csv"
OUTPUT_METADATA_COVERAGE = "raw_summary_agent_preference_metadata_coverage.csv"
OUTPUT_SCOPE_COMPARISON = "raw_summary_agent_preference_scope_comparison.csv"
OUTPUT_MODEL_BREAKDOWN = "raw_summary_agent_preference_model_breakdown.csv"
OUTPUT_PARSE_ERROR_SAMPLES = "raw_summary_agent_preference_parse_error_samples.csv"
OUTPUT_AUDIT = "raw_summary_agent_preference_audit.md"


@dataclass(frozen=True)
class RawPaths:
    """Input and output paths for raw Summary Agent artifact analysis."""

    artifact_root: Path = DEFAULT_RAW_ARTIFACT_ROOT
    results_csv: Path = DEFAULT_RESULTS_CSV
    em_dataset_root: Path = DEFAULT_EM_DATASET_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR


def discover_raw_summary_pairs(artifact_root: Path) -> list[tuple[Path, Path]]:
    """Find paired A/B summaries under raw model output directories."""
    if not artifact_root.exists():
        return []

    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    for a_path in sorted(artifact_root.rglob("output.txt")):
        if a_path.parent.name != "a" or a_path.parent.parent.name != "summarizer":
            continue
        b_path = a_path.parent.parent / "b" / "output.txt"
        if b_path.exists():
            key = (a_path, b_path)
            if key not in seen:
                pairs.append(key)
                seen.add(key)

    # Legacy flat layout
    for a_path in sorted(artifact_root.rglob("a_summary.txt")):
        b_path = a_path.with_name("b_summary.txt")
        if b_path.exists():
            key = (a_path, b_path)
            if key not in seen:
                pairs.append(key)
                seen.add(key)
    return pairs


def extract_raw_path_metadata(path: Path, artifact_root: Path) -> dict[str, str]:
    """Extract model, sample, method, and file slug from a raw artifact path."""
    try:
        relative = path.relative_to(artifact_root)
    except ValueError:
        relative = path

    parts = relative.parts
    # New: <model>/<sample>/<method>/<file_slug>/summarizer/a/output.txt  (7 parts)
    if len(parts) >= 7 and parts[-3] == "summarizer" and parts[-2] == "a":
        return {
            "raw_model_dir": parts[0],
            "sample_id": parts[1],
            "eval_method": parts[2],
            "artifact_file_slug": parts[3],
        }
    # Legacy: <model>/<sample>/<method>/<file_slug>/a_summary.txt  (5 parts)
    if len(parts) < 5:
        return {
            "raw_model_dir": "",
            "sample_id": "",
            "eval_method": "",
            "artifact_file_slug": "",
        }

    return {
        "raw_model_dir": parts[0],
        "sample_id": parts[1],
        "eval_method": parts[2],
        "artifact_file_slug": parts[3],
    }


def build_raw_file_level_table(artifact_root: Path) -> pd.DataFrame:
    """Build one row per raw file-level A/B summary pair."""
    rows: list[dict[str, object]] = []
    all_types: set[str] = set()

    for a_path, b_path in discover_raw_summary_pairs(artifact_root):
        path_metadata = extract_raw_path_metadata(a_path, artifact_root)
        a_counts, a_diag = parse_summary_file(a_path)
        b_counts, b_diag = parse_summary_file(b_path)
        all_types.update(a_counts)
        all_types.update(b_counts)

        raw_model_dir = path_metadata["raw_model_dir"]
        file_path = str(a_diag.get("file_path") or b_diag.get("file_path") or "")
        parse_ok = bool(a_diag["parse_ok"]) and bool(b_diag["parse_ok"])
        row: dict[str, object] = {
            **path_metadata,
            "model_family": _model_family_from_text(raw_model_dir),
            "file_path": file_path,
            "a_summary_path": str(a_path),
            "b_summary_path": str(b_path),
            "a_parse_ok": bool(a_diag["parse_ok"]),
            "b_parse_ok": bool(b_diag["parse_ok"]),
            "parse_ok": parse_ok,
            "a_parse_error": str(a_diag["parse_error"]),
            "b_parse_error": str(b_diag["parse_error"]),
            "a_total_changes": int(sum(a_counts.values())),
            "b_total_changes": int(sum(b_counts.values())),
            "a_likely_intent": str(a_diag.get("likely_intent", "")),
            "b_likely_intent": str(b_diag.get("likely_intent", "")),
        }
        for change_type in all_types:
            row[f"a_type_{change_type}"] = int(a_counts.get(change_type, 0))
            row[f"b_type_{change_type}"] = int(b_counts.get(change_type, 0))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for change_type in sorted(all_types):
        for parent in ("a", "b"):
            col = f"{parent}_type_{change_type}"
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def join_raw_metadata(file_df: pd.DataFrame, paths: RawPaths) -> pd.DataFrame:
    """Join raw summary rows to bypass7 selected-parent and EM metadata."""
    if file_df.empty:
        return file_df

    metadata = load_result_metadata(
        type(
            "Paths",
            (),
            {
                "results_csv": paths.results_csv,
                "em_dataset_root": paths.em_dataset_root,
            },
        )()
    )
    out = file_df.copy()
    if metadata.empty:
        out["selected_parent"] = ""
        out["metadata_source"] = ""
        out["all_files_failed_exact_match"] = pd.NA
        out["any_file_failed_exact_match"] = pd.NA
        return add_selection_columns(out)

    out = out.merge(metadata, on=["sample_id", "model_family"], how="left", suffixes=("", "_result"))
    return add_selection_columns(out)


def aggregate_raw_instance_level(file_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw file-level rows to one row per sample/model/method."""
    if file_df.empty:
        return pd.DataFrame()

    group_cols = ["sample_id", "raw_model_dir", "model_family", "eval_method"]
    type_cols = [c for c in file_df.columns if c.startswith("a_type_") or c.startswith("b_type_")]
    agg_map: dict[str, str] = {
        "a_total_changes": "sum",
        "b_total_changes": "sum",
        "parse_ok": "all",
        "a_parse_ok": "all",
        "b_parse_ok": "all",
        "file_path": "count",
        "selected_parent": "first",
        "metadata_source": "first",
        "result_file_rows": "first",
        "result_failure_file_rows": "first",
        "all_files_failed_exact_match": "first",
        "any_file_failed_exact_match": "first",
        "mean_exact_match": "first",
        "repo": "first",
        "difficulty": "first",
        "project_size": "first",
        "model_name": "first",
    }
    for col in type_cols:
        agg_map[col] = "sum"

    existing_agg_map = {col: func for col, func in agg_map.items() if col in file_df.columns}
    grouped = (
        file_df.groupby(group_cols, as_index=False)
        .agg(existing_agg_map)
        .rename(columns={"file_path": "summary_file_rows"})
    )
    grouped = add_selection_columns(grouped)

    label_counts = (
        file_df.groupby(group_cols)["summary_preference_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["favored_complexity", "favored_simplicity", "tie_ambiguous", "unparseable", "mix_or_missing_selection"]:
        if label not in label_counts.columns:
            label_counts[label] = 0
        label_counts = label_counts.rename(columns={label: f"file_label_{label}_count"})

    majority = (
        file_df.groupby(group_cols)["summary_preference_label"]
        .apply(majority_vote_label)
        .reset_index(name="majority_vote_preference_label")
    )
    grouped = grouped.drop(columns=["summary_preference_label"]).merge(label_counts, on=group_cols, how="left")
    grouped = grouped.merge(majority, on=group_cols, how="left")
    grouped["summary_preference_label"] = grouped["majority_vote_preference_label"]
    return grouped


def failure_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose joined metadata marks at least one EM failure."""
    if df.empty or "any_file_failed_exact_match" not in df.columns:
        return df.iloc[0:0].copy()
    failure_mask = _as_bool(df["any_file_failed_exact_match"])
    return df[failure_mask].copy()


def build_metadata_coverage(file_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize metadata availability at file and instance granularity."""
    rows: list[dict[str, object]] = []
    for grain, df in [("file_level", file_df), ("instance_level", instance_df)]:
        total = len(df)
        if total == 0:
            rows.append({"grain": grain, "rows": 0})
            continue
        has_selected = df["selected_parent"].fillna("").astype(str).str.upper().isin(PARENT_LABELS)
        has_failure = df["any_file_failed_exact_match"].notna() if "any_file_failed_exact_match" in df else pd.Series(False, index=df.index)
        failures = _as_bool(df["any_file_failed_exact_match"]) if "any_file_failed_exact_match" in df else pd.Series(False, index=df.index)
        rows.append(
            {
                "grain": grain,
                "rows": total,
                "rows_with_selected_parent": int(has_selected.sum()),
                "selected_parent_coverage_pct": 100 * float(has_selected.mean()),
                "rows_with_exact_match_metadata": int(has_failure.sum()),
                "exact_match_metadata_coverage_pct": 100 * float(has_failure.mean()),
                "failure_rows": int(failures.sum()),
                "non_failure_rows_with_metadata": int((has_failure & ~failures).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_scope(name: str, grain: str, df: pd.DataFrame) -> dict[str, object]:
    """Create a compact one-row distribution summary for a scope."""
    row: dict[str, object] = {"scope": name, "grain": grain, "rows": len(df)}
    if df.empty:
        return row

    labels = df["summary_preference_label"].value_counts(dropna=False)
    selected = df["selected_parent"].value_counts(dropna=False)
    parse_ok = df["parse_ok"].fillna(False).astype(bool)

    row.update(
        {
            "parse_ok_rows": int(parse_ok.sum()),
            "parse_ok_pct": 100 * float(parse_ok.mean()),
            "selected_parent_A": int(selected.get("A", 0)),
            "selected_parent_B": int(selected.get("B", 0)),
            "selected_parent_missing_or_other": int(len(df) - selected.get("A", 0) - selected.get("B", 0)),
            "favored_simplicity": int(labels.get("favored_simplicity", 0)),
            "favored_complexity": int(labels.get("favored_complexity", 0)),
            "tie_ambiguous": int(labels.get("tie_ambiguous", 0)),
            "unparseable": int(labels.get("unparseable", 0)),
            "mix_or_missing_selection": int(labels.get("mix_or_missing_selection", 0)),
            "mean_a_total_changes": float(df["a_total_changes"].mean()),
            "mean_b_total_changes": float(df["b_total_changes"].mean()),
            "mean_selected_minus_rejected_changes": float(df["selected_minus_rejected_changes"].mean()),
        }
    )
    return row


def build_scope_comparison(file_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    """Compare distributions across all rows and failures-only rows."""
    rows = [
        summarize_scope("all", "file_level", file_df),
        summarize_scope("failures_only", "file_level", failure_subset(file_df)),
        summarize_scope("all", "instance_level", instance_df),
        summarize_scope("failures_only", "instance_level", failure_subset(instance_df)),
    ]
    return pd.DataFrame(rows)


def build_type_distribution(file_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    """Count raw A/B change types for all and failures-only scopes."""
    rows: list[dict[str, object]] = []
    for grain, base_df in [("file_level", file_df), ("instance_level", instance_df)]:
        for scope, df in [("all", base_df), ("failures_only", failure_subset(base_df))]:
            change_types = sorted(
                col.removeprefix("a_type_")
                for col in df.columns
                if col.startswith("a_type_")
            )
            for change_type in change_types:
                a_count = int(df[f"a_type_{change_type}"].sum()) if not df.empty else 0
                b_col = f"b_type_{change_type}"
                b_count = int(df[b_col].sum()) if b_col in df.columns and not df.empty else 0
                rows.append(
                    {
                        "scope": scope,
                        "grain": grain,
                        "change_type": change_type,
                        "a_parent_count": a_count,
                        "b_parent_count": b_count,
                        "a_minus_b": a_count - b_count,
                        "total_count": a_count + b_count,
                    }
                )
    return pd.DataFrame(rows)


def build_model_breakdown(instance_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize preference labels and metadata by model for all/failure scopes."""
    rows: list[dict[str, object]] = []
    if instance_df.empty:
        return pd.DataFrame(rows)

    for scope, df in [("all", instance_df), ("failures_only", failure_subset(instance_df))]:
        for model, group in df.groupby("raw_model_dir", dropna=False):
            labels = group["summary_preference_label"].value_counts(dropna=False)
            selected = group["selected_parent"].value_counts(dropna=False)
            parse_ok = group["parse_ok"].fillna(False).astype(bool)
            rows.append(
                {
                    "scope": scope,
                    "raw_model_dir": model,
                    "rows": len(group),
                    "parse_ok_rows": int(parse_ok.sum()),
                    "parse_ok_pct": 100 * float(parse_ok.mean()) if len(group) else 0,
                    "selected_parent_A": int(selected.get("A", 0)),
                    "selected_parent_B": int(selected.get("B", 0)),
                    "favored_simplicity": int(labels.get("favored_simplicity", 0)),
                    "favored_complexity": int(labels.get("favored_complexity", 0)),
                    "tie_ambiguous": int(labels.get("tie_ambiguous", 0)),
                    "unparseable": int(labels.get("unparseable", 0)),
                    "mix_or_missing_selection": int(labels.get("mix_or_missing_selection", 0)),
                    "mean_selected_minus_rejected_changes": float(group["selected_minus_rejected_changes"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_parse_error_samples(file_df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Return representative parse-error rows for manual inspection."""
    if file_df.empty:
        return pd.DataFrame()
    errors = file_df[~file_df["parse_ok"].fillna(False).astype(bool)].copy()
    if errors.empty:
        return errors
    keep_cols = [
        "raw_model_dir",
        "sample_id",
        "eval_method",
        "artifact_file_slug",
        "file_path",
        "a_parse_ok",
        "b_parse_ok",
        "a_parse_error",
        "b_parse_error",
        "a_summary_path",
        "b_summary_path",
    ]
    return errors[[col for col in keep_cols if col in errors.columns]].head(limit)


def _format_label_distribution(df: pd.DataFrame) -> list[str]:
    labels = df["summary_preference_label"].value_counts(dropna=False) if not df.empty else pd.Series(dtype=int)
    total = len(df)
    if total == 0:
        return ["- No rows."]
    return [f"- `{label}`: {_format_count_pct(int(count), total)}" for label, count in labels.items()]


def write_audit(
    output_path: Path,
    file_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    type_counts: pd.DataFrame,
    metadata_coverage: pd.DataFrame,
    scope_comparison: pd.DataFrame,
    type_distribution: pd.DataFrame,
    model_breakdown: pd.DataFrame,
    parse_error_samples: pd.DataFrame,
) -> None:
    """Write a markdown report for the raw-output analysis."""
    all_instances = len(instance_df)
    all_files = len(file_df)
    failure_instances = len(failure_subset(instance_df))
    failure_files = len(failure_subset(file_df))
    models = sorted(file_df["raw_model_dir"].dropna().astype(str).unique()) if not file_df.empty else []
    methods = sorted(file_df["eval_method"].dropna().astype(str).unique()) if not file_df.empty else []
    parse_failures = int((~file_df["parse_ok"].fillna(False).astype(bool)).sum()) if not file_df.empty else 0

    instance_type_counts = type_counts[
        (type_counts["scope"] == "all")
        & (type_counts["grain"] == "instance_level")
        & type_counts["selected_parent_count"].notna()
    ].copy()

    most_under = instance_type_counts.sort_values("selected_minus_rejected").head(8)
    most_over = instance_type_counts.sort_values("selected_minus_rejected", ascending=False).head(8)
    failure_instance_type_counts = type_counts[
        (type_counts["scope"] == "failures_only")
        & (type_counts["grain"] == "instance_level")
        & type_counts["selected_parent_count"].notna()
    ].copy()
    failure_most_under = failure_instance_type_counts.sort_values("selected_minus_rejected").head(8)
    failure_most_over = failure_instance_type_counts.sort_values("selected_minus_rejected", ascending=False).head(8)

    lines = [
        "# Raw Summary Agent Preference Audit",
        "",
        "## Coverage",
        "",
        f"- Raw file-level summary pairs parsed: {all_files:,}.",
        f"- Raw instance-level rows after aggregation: {all_instances:,}.",
        f"- Failures-only file-level rows with joined EM failure metadata: {failure_files:,}.",
        f"- Failures-only instance-level rows with joined EM failure metadata: {failure_instances:,}.",
        f"- Parse failure file rows: {parse_failures:,}.",
        f"- Models: {', '.join(models) if models else 'none'}.",
        f"- Methods: {', '.join(methods) if methods else 'none'}.",
        "",
        "## Metadata Coverage",
        "",
        metadata_coverage.to_markdown(index=False) if not metadata_coverage.empty else "No metadata coverage rows.",
        "",
        "## Derived Preference Labels: All Raw Instances",
        "",
        *_format_label_distribution(instance_df),
        "",
        "## Derived Preference Labels: Failures Only",
        "",
        *_format_label_distribution(failure_subset(instance_df)),
        "",
        "## All vs Failures Comparison",
        "",
        scope_comparison.to_markdown(index=False) if not scope_comparison.empty else "No scope comparison rows.",
        "",
        "## Model-Level Breakdown",
        "",
        model_breakdown.to_markdown(index=False) if not model_breakdown.empty else "No model breakdown rows.",
        "",
        "## Selected vs Rejected Change Types",
        "",
        "Negative values mean the pipeline-selected parent had fewer summarized entries of that type than the rejected parent; positive values mean the selected parent had more.",
        "",
        "### Most Under-Represented In Selected Parent (All Instances)",
        "",
        most_under.to_markdown(index=False) if not most_under.empty else "No selected-parent type rows.",
        "",
        "### Most Over-Represented In Selected Parent (All Instances)",
        "",
        most_over.to_markdown(index=False) if not most_over.empty else "No selected-parent type rows.",
        "",
        "### Most Under-Represented In Selected Parent (Failures Only)",
        "",
        failure_most_under.to_markdown(index=False) if not failure_most_under.empty else "No selected-parent type rows.",
        "",
        "### Most Over-Represented In Selected Parent (Failures Only)",
        "",
        failure_most_over.to_markdown(index=False) if not failure_most_over.empty else "No selected-parent type rows.",
        "",
        "## Raw A/B Change-Type Distribution",
        "",
        type_distribution.sort_values(["scope", "grain", "total_count"], ascending=[True, True, False])
        .head(40)
        .to_markdown(index=False)
        if not type_distribution.empty
        else "No raw type distribution rows.",
        "",
        "## Interpretation",
        "",
        "This analysis expands the Summary Agent artifact audit from curated labeled cases to all paired summaries available under `data/raw_model_outputs`. The all-instance rows describe the raw handoff artifacts regardless of outcome. The failures-only rows are limited to artifacts that could be joined to exact-match metadata and had at least one failed file in the corresponding result rows.",
        "",
        "The selected-vs-rejected statistics remain proxy evidence: they compare the compact Summary Agent records for the parent selected by the bypass decision against the non-selected parent. They show whether asymmetry is already visible in the upstream summaries, but they do not prove that every missing summarized change was omitted by the Summarizer rather than absent from the underlying parent diff.",
        "",
        "## Parse Error Samples",
        "",
        parse_error_samples.to_markdown(index=False) if not parse_error_samples.empty else "No parse errors.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_selected_type_counts_by_scope(file_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    """Build selected-vs-rejected type counts for all and failures-only scopes."""
    frames: list[pd.DataFrame] = []
    for scope, scoped_file, scoped_instance in [
        ("all", file_df, instance_df),
        ("failures_only", failure_subset(file_df), failure_subset(instance_df)),
    ]:
        scoped = build_type_counts(scoped_file, scoped_instance)
        if scoped.empty:
            continue
        scoped.insert(0, "scope_filter", scope)
        scoped = scoped.rename(columns={"scope": "grain"})
        scoped = scoped.rename(columns={"scope_filter": "scope"})
        frames.append(scoped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def run_raw_analysis(paths: RawPaths) -> dict[str, Path]:
    """Run raw Summary Agent artifact analysis and write outputs."""
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    file_df = build_raw_file_level_table(paths.artifact_root)
    file_df = join_raw_metadata(file_df, paths)
    instance_df = aggregate_raw_instance_level(file_df)
    type_counts = build_selected_type_counts_by_scope(file_df, instance_df)
    metadata_coverage = build_metadata_coverage(file_df, instance_df)
    scope_comparison = build_scope_comparison(file_df, instance_df)
    type_distribution = build_type_distribution(file_df, instance_df)
    model_breakdown = build_model_breakdown(instance_df)
    parse_error_samples = build_parse_error_samples(file_df)

    outputs = {
        "file_level": paths.output_dir / OUTPUT_FILE_LEVEL,
        "instance_level": paths.output_dir / OUTPUT_INSTANCE_LEVEL,
        "type_counts": paths.output_dir / OUTPUT_TYPE_COUNTS,
        "type_distribution": paths.output_dir / OUTPUT_TYPE_DISTRIBUTION,
        "metadata_coverage": paths.output_dir / OUTPUT_METADATA_COVERAGE,
        "scope_comparison": paths.output_dir / OUTPUT_SCOPE_COMPARISON,
        "model_breakdown": paths.output_dir / OUTPUT_MODEL_BREAKDOWN,
        "parse_error_samples": paths.output_dir / OUTPUT_PARSE_ERROR_SAMPLES,
        "audit": paths.output_dir / OUTPUT_AUDIT,
    }
    file_df.to_csv(outputs["file_level"], index=False)
    instance_df.to_csv(outputs["instance_level"], index=False)
    type_counts.to_csv(outputs["type_counts"], index=False)
    type_distribution.to_csv(outputs["type_distribution"], index=False)
    metadata_coverage.to_csv(outputs["metadata_coverage"], index=False)
    scope_comparison.to_csv(outputs["scope_comparison"], index=False)
    model_breakdown.to_csv(outputs["model_breakdown"], index=False)
    parse_error_samples.to_csv(outputs["parse_error_samples"], index=False)
    write_audit(
        outputs["audit"],
        file_df,
        instance_df,
        type_counts,
        metadata_coverage,
        scope_comparison,
        type_distribution,
        model_breakdown,
        parse_error_samples,
    )
    return outputs


def parse_args() -> RawPaths:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_RAW_ARTIFACT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--em-dataset-root", type=Path, default=DEFAULT_EM_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return RawPaths(
        artifact_root=args.artifact_root,
        results_csv=args.results_csv,
        em_dataset_root=args.em_dataset_root,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the CLI entry point."""
    outputs = run_raw_analysis(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
