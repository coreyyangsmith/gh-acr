"""Derive complexity-preference labels from summary-agent A/B outputs.

This analysis parses the persisted bypass summary artifacts, counts the
``changes[*].type`` entries for Parent A and Parent B, and labels exact-match
failure cases by whether the conflict analyzer selected the parent with more
or fewer summarized change objects.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ARTIFACT_ROOT = Path("data/labeling_results")
DEFAULT_RESULTS_CSV = Path("results/figures/force_mix_vs_final_combined.csv")
DEFAULT_EM_DATASET_ROOT = Path("results/em_datasets")
DEFAULT_FAIL_ONLY_PAIRED_CSV = Path("results/rq3_fail_only/paired_data.csv")
DEFAULT_MANUAL_PREF_CSV = Path("results/rq3/complexity_preference_summary.csv")
DEFAULT_OUTPUT_DIR = Path("results/rq3")

OUTPUT_FILE_LEVEL = "summary_agent_preference_file_level.csv"
OUTPUT_INSTANCE_LEVEL = "summary_agent_preference_instance_level.csv"
OUTPUT_TYPE_COUNTS = "summary_agent_preference_type_counts.csv"
OUTPUT_AUDIT = "summary_agent_preference_audit.md"

PARENT_LABELS = ("A", "B")
KNOWN_TYPES = ("addition", "modification", "removal", "unknown")
TYPE_NORMALIZATION = {
    "add": "addition",
    "added": "addition",
    "addition": "addition",
    "delete": "removal",
    "deleted": "removal",
    "deletion": "removal",
    "remove": "removal",
    "removed": "removal",
    "removal": "removal",
    "modify": "modification",
    "modified": "modification",
    "modification": "modification",
    "update": "modification",
    "updated": "modification",
}


@dataclass(frozen=True)
class Paths:
    """Input and output paths for the summary-agent preference analysis."""

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    results_csv: Path = DEFAULT_RESULTS_CSV
    em_dataset_root: Path = DEFAULT_EM_DATASET_ROOT
    fail_only_paired_csv: Path = DEFAULT_FAIL_ONLY_PAIRED_CSV
    manual_preference_csv: Path = DEFAULT_MANUAL_PREF_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _as_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def _format_count_pct(count: int, denominator: int) -> str:
    pct = 100 * count / denominator if denominator else 0
    return f"{count:,} ({pct:.1f}%)"


def extract_base_id(case_dir_name: str) -> str:
    """Extract the merge instance ID from a case directory name."""
    match = re.match(r"^(?P<sample_id>.+)-(?P<file_index>\d+)$", case_dir_name)
    return match.group("sample_id") if match else case_dir_name


def source_file_from_run_dir(run_dir_name: str) -> str:
    """Map a failure-case artifact directory to the RQ3 source_file value."""
    if run_dir_name.endswith("-cases"):
        return f"{run_dir_name[:-len('-cases')]}-classifications"
    return run_dir_name


def _model_family_from_text(value: object) -> str:
    text = str(value).lower()
    if "gpt" in text and "nano" in text:
        return "gpt5nano"
    if "qwen" in text:
        return "qwen3"
    if "llama" in text:
        return "llama"
    return ""


def normalize_change_type(value: object) -> str:
    """Normalize a raw summary change type to a small auditable vocabulary."""
    normalized = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in TYPE_NORMALIZATION:
        return TYPE_NORMALIZATION[normalized]
    for token, canonical in TYPE_NORMALIZATION.items():
        if token in normalized.split():
            return canonical
    return normalized or "unknown"


def _strip_json_fence(text: str) -> str:
    """Remove common markdown wrappers from model JSON output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_likely_intent(text: str) -> str:
    match = re.search(r'"likely_intent"\s*:\s*"(?P<intent>(?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group("intent")}"')
    except Exception:
        return match.group("intent")


def _decode_summary_payload(raw: object) -> tuple[list[Any], str, str]:
    """Decode summary JSON, accepting common malformed-but-countable variants."""
    if not isinstance(raw, str):
        if isinstance(raw, dict):
            changes = raw.get("changes", [])
            return (changes if isinstance(changes, list) else [], str(raw.get("likely_intent", "")), "")
        if isinstance(raw, list):
            return raw, "", ""
        return [], "", "nested_summary_json: expected string, object, or list"

    text = _strip_json_fence(raw)
    if not text or text.lower() in {"(no changes)", "no changes"}:
        return [], "", ""

    try:
        decoded = json.loads(text)
    except Exception as exc:
        try:
            decoded, _ = json.JSONDecoder().raw_decode(text)
        except Exception:
            type_matches = re.findall(r'"type"\s*:\s*"(?P<type>[^"]+)"', text, flags=re.IGNORECASE)
            if type_matches:
                return (
                    [{"type": change_type} for change_type in type_matches],
                    _extract_likely_intent(text),
                    f"nested_summary_json_fallback_type_regex: {exc}",
                )
            return [], "", f"nested_summary_json: {exc}"
        error = f"nested_summary_json_prefix: {exc}"
    else:
        error = ""

    if isinstance(decoded, dict):
        changes = decoded.get("changes", [])
        likely_intent = str(decoded.get("likely_intent", "")).strip()
    elif isinstance(decoded, list):
        changes = decoded
        likely_intent = _extract_likely_intent(text)
    else:
        return [], "", "nested_summary_json: expected object or array"

    if not isinstance(changes, list):
        return [], likely_intent, "changes: expected list"
    return changes, likely_intent, error


def parse_summary_file(path: Path) -> tuple[Counter[str], dict[str, object]]:
    """Parse one parent summary artifact and return change-type counts plus diagnostics."""
    diagnostics: dict[str, object] = {
        "parse_ok": False,
        "parse_error": "",
        "parent": "",
        "file_path": "",
        "likely_intent": "",
        "change_count": 0,
    }
    counts: Counter[str] = Counter()

    try:
        outer = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        diagnostics["parse_error"] = f"outer_json: {exc}"
        counts["unknown"] += 1
        return counts, diagnostics

    diagnostics["parent"] = str(outer.get("parent", "")).strip().upper()
    diagnostics["file_path"] = str(outer.get("file_path", "")).strip()
    nested_raw = outer.get("summary", "")

    changes, likely_intent, parse_error = _decode_summary_payload(nested_raw)
    diagnostics["likely_intent"] = likely_intent
    diagnostics["parse_error"] = parse_error

    for change in changes:
        if isinstance(change, dict):
            counts[normalize_change_type(change.get("type", ""))] += 1
        else:
            counts["unknown"] += 1

    if parse_error and not counts:
        counts["unknown"] += 1
        diagnostics["change_count"] = int(sum(counts.values()))
        return counts, diagnostics

    diagnostics["parse_ok"] = not bool(parse_error) or bool(counts)
    diagnostics["change_count"] = int(sum(counts.values()))
    return counts, diagnostics


def discover_summary_pairs(artifact_root: Path) -> list[tuple[Path, Path]]:
    """Find file-level A/B summary artifact pairs.

    Supports the per-agent layout::

        <method>/<file_slug>/summarizer/a/output.txt
        <method>/<file_slug>/summarizer/b/output.txt

    Also accepts the legacy flat ``a_summary.txt`` / ``b_summary.txt`` pairs under
    a ``bypass`` method folder for older curated labeling trees.
    """
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

    for a_path in sorted(artifact_root.rglob("a_summary.txt")):
        if a_path.parent.parent.name != "bypass":
            continue
        b_path = a_path.with_name("b_summary.txt")
        if b_path.exists():
            key = (a_path, b_path)
            if key not in seen:
                pairs.append(key)
                seen.add(key)
    return pairs


def _summary_file_slug(a_path: Path) -> str:
    """Return the conflicted-file slug for a summary artifact path."""
    # New layout: .../<file_slug>/summarizer/a/output.txt
    if a_path.name == "output.txt" and a_path.parent.name == "a":
        return a_path.parent.parent.parent.name
    # Legacy: .../<file_slug>/a_summary.txt
    return a_path.parent.name


def build_file_level_table(artifact_root: Path) -> pd.DataFrame:
    """Build one row per file-level A/B summary pair."""
    rows: list[dict[str, object]] = []
    all_types: set[str] = set(KNOWN_TYPES)

    for a_path, b_path in discover_summary_pairs(artifact_root):
        file_slug = _summary_file_slug(a_path)
        if a_path.name == "output.txt":
            # .../<case>/<method>/<file_slug>/summarizer/a/output.txt
            method_dir = a_path.parent.parent.parent.parent
            case_dir = method_dir.parent
        else:
            # .../<case>/bypass/<file_slug>/a_summary.txt
            file_dir = a_path.parent
            case_dir = file_dir.parent.parent
        run_dir = case_dir.parent
        sample_id = extract_base_id(case_dir.name)
        source_file = source_file_from_run_dir(run_dir.name)

        a_counts, a_diag = parse_summary_file(a_path)
        b_counts, b_diag = parse_summary_file(b_path)
        all_types.update(a_counts)
        all_types.update(b_counts)

        file_path = str(a_diag.get("file_path") or b_diag.get("file_path") or "")
        parse_ok = bool(a_diag["parse_ok"]) and bool(b_diag["parse_ok"])
        row: dict[str, object] = {
            "sample_id": sample_id,
            "case_id": case_dir.name,
            "source_run": run_dir.name,
            "source_file": source_file,
            "model_family": _model_family_from_text(source_file),
            "artifact_file_slug": file_slug,
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
        for change_type in sorted(all_types):
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


def _collapse_result_rows(df: pd.DataFrame, metadata_source: str) -> pd.DataFrame:
    """Collapse per-file bypass7 result rows to one row per sample/model family."""
    if df.empty or "id" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["sample_id"] = _as_id(work["id"])
    if "eval_method" in work.columns:
        work = work[work["eval_method"].astype(str).str.lower() == "bypass7"]
    if work.empty:
        return pd.DataFrame()

    if "exact_match" not in work.columns:
        return pd.DataFrame()

    work["exact_match_bool"] = _as_bool(work["exact_match"])
    work["is_failure"] = ~work["exact_match_bool"]
    if "bypass_method" not in work.columns:
        work["bypass_method"] = ""
    if "model_name" not in work.columns:
        work["model_name"] = ""
    work["model_family"] = work["model_name"].apply(_model_family_from_text)

    metadata_cols = [c for c in ["repo", "difficulty", "project_size", "model_name"] if c in work.columns]
    rows: list[dict[str, object]] = []
    for (sample_id, model_family), group in work.groupby(["sample_id", "model_family"], dropna=False):
        method_counts = group["bypass_method"].astype(str).str.upper().value_counts()
        selected = str(method_counts.index[0]).upper() if not method_counts.empty else ""
        row: dict[str, object] = {
            "sample_id": sample_id,
            "model_family": model_family,
            "selected_parent": selected,
            "metadata_source": metadata_source,
            "result_file_rows": int(len(group)),
            "result_failure_file_rows": int(group["is_failure"].sum()),
            "all_files_failed_exact_match": bool(group["is_failure"].all()),
            "any_file_failed_exact_match": bool(group["is_failure"].any()),
            "mean_exact_match": float(group["exact_match_bool"].mean()),
        }
        for col in metadata_cols:
            values = sorted(set(group[col].dropna().astype(str)))
            row[col] = ";".join(values)
        rows.append(row)

    return pd.DataFrame(rows)


def load_combined_result_metadata(path: Path) -> pd.DataFrame:
    """Load bypass7 parent selections from the combined results CSV."""
    if not path.exists():
        return pd.DataFrame()
    return _collapse_result_rows(pd.read_csv(path), str(path))


def load_em_dataset_metadata(root: Path) -> pd.DataFrame:
    """Load bypass7 parent selections from per-model EM datasets as a fallback."""
    if not root.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/bypass7/dataset.csv")):
        df = pd.read_csv(path)
        if "eval_method" not in df.columns:
            df["eval_method"] = "bypass7"
        if "model_name" not in df.columns:
            df["model_name"] = path.parents[1].name
        frames.append(_collapse_result_rows(df, str(path)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_result_metadata(paths: Paths) -> pd.DataFrame:
    """Load and de-duplicate selected-parent metadata."""
    frames = [
        load_combined_result_metadata(paths.results_csv),
        load_em_dataset_metadata(paths.em_dataset_root),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    metadata = pd.concat(frames, ignore_index=True, sort=False)
    metadata = metadata[metadata["model_family"].astype(str) != ""].copy()
    if metadata.empty:
        return metadata

    priority = metadata["metadata_source"].astype(str).str.contains("force_mix_vs_final_combined").map(
        {True: 0, False: 1}
    )
    metadata["_priority"] = priority
    metadata = metadata.sort_values(["sample_id", "model_family", "_priority"])
    metadata = metadata.drop_duplicates(["sample_id", "model_family"], keep="first")
    return metadata.drop(columns=["_priority"])


def apply_preference_label(row: pd.Series) -> str:
    """Label whether the selected parent has more or fewer summary changes."""
    selected = str(row.get("selected_parent", "")).upper()
    parse_ok = bool(row.get("parse_ok", False))
    if not parse_ok:
        return "unparseable"
    if selected not in PARENT_LABELS:
        return "mix_or_missing_selection"

    a_total = int(row.get("a_total_changes", 0))
    b_total = int(row.get("b_total_changes", 0))
    selected_count = a_total if selected == "A" else b_total
    rejected_count = b_total if selected == "A" else a_total

    if selected_count > rejected_count:
        return "favored_complexity"
    if selected_count < rejected_count:
        return "favored_simplicity"
    return "tie_ambiguous"


def add_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add selected/rejected count columns and file-level preference labels."""
    if df.empty:
        return df

    out = df.copy()
    out["selected_parent"] = out["selected_parent"].fillna("").astype(str).str.upper()
    out["rejected_parent"] = out["selected_parent"].map({"A": "B", "B": "A"}).fillna("")
    out["selected_change_count"] = out.apply(
        lambda row: row["a_total_changes"]
        if row["selected_parent"] == "A"
        else (row["b_total_changes"] if row["selected_parent"] == "B" else math.nan),
        axis=1,
    )
    out["rejected_change_count"] = out.apply(
        lambda row: row["b_total_changes"]
        if row["selected_parent"] == "A"
        else (row["a_total_changes"] if row["selected_parent"] == "B" else math.nan),
        axis=1,
    )
    out["selected_minus_rejected_changes"] = out["selected_change_count"] - out["rejected_change_count"]
    out["summary_preference_label"] = out.apply(apply_preference_label, axis=1)
    return out


def join_metadata(file_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Join file-level summary rows to result metadata."""
    if file_df.empty:
        return file_df
    if metadata_df.empty:
        out = file_df.copy()
        out["selected_parent"] = ""
        out["metadata_source"] = ""
        out["all_files_failed_exact_match"] = False
        out["any_file_failed_exact_match"] = False
        return add_selection_columns(out)

    join_cols = ["sample_id", "model_family"]
    out = file_df.merge(metadata_df, on=join_cols, how="left", suffixes=("", "_result"))
    return add_selection_columns(out)


def majority_vote_label(labels: pd.Series) -> str:
    """Resolve an instance label by majority vote over file-level labels."""
    clean = labels.dropna().astype(str)
    if clean.empty:
        return "unparseable"

    counts = clean.value_counts()
    top_count = int(counts.iloc[0])
    winners = sorted(counts[counts == top_count].index)
    if len(winners) == 1:
        return str(winners[0])
    return "tie_ambiguous"


def aggregate_instance_level(file_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts and resolve instance labels by file-level majority vote."""
    if file_df.empty:
        return pd.DataFrame()

    group_cols = ["sample_id", "source_file", "model_family"]
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


def build_type_counts(file_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected and rejected change-type counts."""
    rows: list[dict[str, object]] = []
    if file_df.empty:
        return pd.DataFrame(rows)

    change_types = sorted(
        {
            col.removeprefix("a_type_")
            for col in file_df.columns
            if col.startswith("a_type_")
        }
    )
    for scope_name, df in [("file_level", file_df), ("instance_level", instance_df)]:
        selected_parent = df["selected_parent"].fillna("").astype(str).str.upper()
        valid = selected_parent.isin(PARENT_LABELS)
        for change_type in change_types:
            a_col = f"a_type_{change_type}"
            b_col = f"b_type_{change_type}"
            selected_count = int(
                df.loc[valid & (selected_parent == "A"), a_col].sum()
                + df.loc[valid & (selected_parent == "B"), b_col].sum()
            )
            rejected_count = int(
                df.loc[valid & (selected_parent == "A"), b_col].sum()
                + df.loc[valid & (selected_parent == "B"), a_col].sum()
            )
            rows.append(
                {
                    "scope": scope_name,
                    "change_type": change_type,
                    "selected_parent_count": selected_count,
                    "rejected_parent_count": rejected_count,
                    "selected_minus_rejected": selected_count - rejected_count,
                }
            )
    return pd.DataFrame(rows)


def load_manual_preferences(path: Path) -> pd.DataFrame:
    """Load existing manual preference labels for agreement checks."""
    if not path.exists():
        return pd.DataFrame()
    manual = pd.read_csv(path)
    if "sample_id" not in manual.columns:
        return pd.DataFrame()
    manual["sample_id"] = _as_id(manual["sample_id"])
    keep_cols = [c for c in ["sample_id", "source_file", "dominant_preference"] if c in manual.columns]
    return manual[keep_cols].copy()


def add_manual_agreement(instance_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """Attach manual labels and mark agreement with derived summary labels."""
    if instance_df.empty or manual_df.empty:
        return instance_df

    out = instance_df.merge(manual_df, on=["sample_id", "source_file"], how="left")
    summary_to_manual = {
        "favored_complexity": "favored complexity",
        "favored_simplicity": "favored simplicity",
        "tie_ambiguous": "tie (ambiguous)",
    }
    out["summary_preference_manual_equivalent"] = out["summary_preference_label"].map(summary_to_manual).fillna("")
    out["manual_agreement"] = (
        out["summary_preference_manual_equivalent"].astype(str)
        == out["dominant_preference"].fillna("").astype(str)
    ).astype("boolean")
    out.loc[out["dominant_preference"].isna(), "manual_agreement"] = pd.NA
    return out


def write_audit(
    output_path: Path,
    file_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    type_counts: pd.DataFrame,
    fail_only_paired: pd.DataFrame,
) -> None:
    """Write a concise markdown audit report."""
    total_file_rows = len(file_df)
    total_instances = len(instance_df)
    parse_failures = int((~file_df["parse_ok"]).sum()) if not file_df.empty else 0
    selected_parent_counts = instance_df["selected_parent"].value_counts(dropna=False) if not instance_df.empty else pd.Series(dtype=int)
    label_counts = instance_df["summary_preference_label"].value_counts(dropna=False) if not instance_df.empty else pd.Series(dtype=int)
    metadata_matched = int(instance_df["selected_parent"].isin(PARENT_LABELS).sum()) if not instance_df.empty else 0
    exact_failure_instances = (
        int(instance_df["any_file_failed_exact_match"].fillna(False).sum())
        if "any_file_failed_exact_match" in instance_df
        else 0
    )
    fail_only_ids = set(fail_only_paired["id"].astype(str)) if not fail_only_paired.empty and "id" in fail_only_paired else set()
    parsed_ids = set(instance_df["sample_id"].astype(str)) if not instance_df.empty else set()
    missing_artifacts = len(fail_only_ids.difference(parsed_ids))

    lines = [
        "# Summary-Agent Complexity Preference Audit",
        "",
        "## Coverage",
        "",
        f"- File-level summary pairs parsed: {total_file_rows:,}.",
        f"- Instance-level rows after aggregation: {total_instances:,}.",
        f"- Instances with selected parent metadata A/B: {metadata_matched:,}.",
        f"- Instances with at least one exact-match failure row: {exact_failure_instances:,}.",
        f"- Parse failure file rows: {parse_failures:,}.",
        f"- Fail-only paired IDs without parsed summary artifacts: {missing_artifacts:,}.",
        "",
        "## Analyzer Decisions",
        "",
    ]
    if selected_parent_counts.empty:
        lines.append("No selected-parent metadata was available.")
    else:
        for label, count in selected_parent_counts.items():
            lines.append(f"- `{label}`: {_format_count_pct(int(count), total_instances)}")

    lines.extend(["", "## Derived Preference Labels", ""])
    if label_counts.empty:
        lines.append("No derived labels were available.")
    else:
        for label, count in label_counts.items():
            lines.append(f"- `{label}`: {_format_count_pct(int(count), total_instances)}")

    if "manual_agreement" in instance_df.columns:
        comparable = instance_df["manual_agreement"].dropna()
        agreement = int(comparable.sum()) if not comparable.empty else 0
        lines.extend(
            [
                "",
                "## Manual Label Agreement",
                "",
                f"- Comparable instances: {len(comparable):,}.",
                f"- Agreement with existing manual dominant preference: {_format_count_pct(agreement, len(comparable))}.",
            ]
        )

    lines.extend(["", "## Selected vs Rejected Type Counts", ""])
    if type_counts.empty:
        lines.append("No type-count rows were available.")
    else:
        visible = type_counts[type_counts["scope"] == "instance_level"].copy()
        lines.append(visible.to_markdown(index=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_fail_only_paired(path: Path) -> pd.DataFrame:
    """Load fail-only paired data for coverage checks."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "id" in df.columns:
        df["id"] = _as_id(df["id"])
    return df


def run_analysis(paths: Paths) -> dict[str, Path]:
    """Run the full summary-agent preference analysis."""
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    file_df = build_file_level_table(paths.artifact_root)
    metadata_df = load_result_metadata(paths)
    file_df = join_metadata(file_df, metadata_df)
    if "any_file_failed_exact_match" in file_df.columns:
        file_df = file_df[file_df["any_file_failed_exact_match"].fillna(False)].copy()

    instance_df = aggregate_instance_level(file_df)
    manual_df = load_manual_preferences(paths.manual_preference_csv)
    instance_df = add_manual_agreement(instance_df, manual_df)
    type_counts = build_type_counts(file_df, instance_df)
    fail_only_paired = load_fail_only_paired(paths.fail_only_paired_csv)

    outputs = {
        "file_level": paths.output_dir / OUTPUT_FILE_LEVEL,
        "instance_level": paths.output_dir / OUTPUT_INSTANCE_LEVEL,
        "type_counts": paths.output_dir / OUTPUT_TYPE_COUNTS,
        "audit": paths.output_dir / OUTPUT_AUDIT,
    }
    file_df.to_csv(outputs["file_level"], index=False)
    instance_df.to_csv(outputs["instance_level"], index=False)
    type_counts.to_csv(outputs["type_counts"], index=False)
    write_audit(outputs["audit"], file_df, instance_df, type_counts, fail_only_paired)
    return outputs


def parse_args() -> Paths:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--em-dataset-root", type=Path, default=DEFAULT_EM_DATASET_ROOT)
    parser.add_argument("--fail-only-paired-csv", type=Path, default=DEFAULT_FAIL_ONLY_PAIRED_CSV)
    parser.add_argument("--manual-preference-csv", type=Path, default=DEFAULT_MANUAL_PREF_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return Paths(
        artifact_root=args.artifact_root,
        results_csv=args.results_csv,
        em_dataset_root=args.em_dataset_root,
        fail_only_paired_csv=args.fail_only_paired_csv,
        manual_preference_csv=args.manual_preference_csv,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the CLI entry point."""
    outputs = run_analysis(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
