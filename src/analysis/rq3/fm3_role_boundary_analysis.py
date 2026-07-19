"""Empirical analysis for FM3 role-boundary ambiguity.

This module operationalizes FM3 with existing RQ3 labels, joins those labels to
MIX outcomes, and exports tables that can support paper claims with explicit
caveats about proxy validity.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - only used when scipy is unavailable.
    stats = None


METRICS = ["exact_match", "similarity", "bleu3", "rouge_l"]

LABEL_COLUMNS = [
    "favored_simplicity",
    "favored_complexity",
    "lost_information_compression",
    "misprioritization",
    "feature_oriented",
    "fix_oriented",
    "structural_change_bias",
    "unclear",
    "accurate",
    "vague_commit_message",
    "simple_commit_message",
    "detailed_commit_message",
    "refactor_oriented",
    "modification_bias",
    "test_oriented",
]

ORIENTATION_COLUMNS = [
    "feature_oriented",
    "fix_oriented",
    "refactor_oriented",
    "test_oriented",
]

TRACE_KEYWORDS = {
    "partial_merge": r"\bpartial merge|partially merged|partial\b",
    "plan_compliance": r"\bplan compliance|plan adherence|follow(?:ed)? the plan|does not comply\b",
    "missing_parent_edit": r"\bpreserv(?:e|ing)|missing|omitted|lost\b.*\b(parent|side|change|edit)\b",
    "minimality": r"\bminimal|minimality|concise|smaller change\b",
    "completeness": r"\bcomplete|incomplete|all changes|both sides\b",
}


@dataclass(frozen=True)
class Paths:
    """Input and output paths for FM3 analysis."""

    paired_csv: Path = Path("results/rq3/paired_data.csv")
    aggregate_csv: Path = Path("results/rq3/aggregate_combined.csv")
    supplemental_paired_csv: Path = Path("results/rq3_fail_only/paired_data.csv")
    supplemental_aggregate_csv: Path = Path("results/rq3_fail_only/aggregate_combined.csv")
    complexity_preference_csv: Path = Path("results/rq3/complexity_preference_summary.csv")
    mix_results_csv: Path = Path("results/figures/force_mix_vs_final_combined.csv")
    output_dir: Path = Path("results/rq3")
    trace_root: Path = Path("results")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def _as_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def _as_float_bool(series: pd.Series) -> pd.Series:
    return _as_bool(series).astype(float)


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.mean())


def _bootstrap_ci(values: Iterable[float], *, random_state: int = 42) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(random_state)
    boot = np.empty(2000)
    for i in range(len(boot)):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        boot[i] = float(np.mean(sample))
    return (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))


def _quantile_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def _bootstrap_mean_diffs(
    positive_values: Iterable[float],
    negative_values: Iterable[float],
    *,
    n_boot: int = 2000,
    random_state: int = 42,
) -> list[float]:
    pos = np.asarray(list(positive_values), dtype=float)
    neg = np.asarray(list(negative_values), dtype=float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return []

    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_boot):
        pos_sample = pos[rng.integers(0, len(pos), size=len(pos))]
        neg_sample = neg[rng.integers(0, len(neg), size=len(neg))]
        diffs.append(float(np.mean(pos_sample) - np.mean(neg_sample)))
    return diffs


def _corrected_odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """Haldane-Anscombe corrected odds ratio for sparse MIX tables."""
    return float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))


def build_proxy_mapping() -> pd.DataFrame:
    """Describe how FM3 subclaims map onto observable RQ3 variables."""
    rows = [
        {
            "fm3_subclaim": "Underspecified output expectations",
            "proxy_family": "ambiguity_incompleteness",
            "columns": "unclear; misprioritization; lost_information_compression",
            "rationale": (
                "These labels capture outputs that annotators found hard to characterize, "
                "optimized toward the wrong priority, or omitted relevant information."
            ),
            "claim_strength": "indirect RQ3 proxy",
        },
        {
            "fm3_subclaim": "Planner minimality conflicts with reviewer completeness",
            "proxy_family": "simplicity_complexity_tension",
            "columns": (
                "favored_simplicity; favored_complexity; simplicity_files; "
                "complexity_files; dominant_preference"
            ),
            "rationale": (
                "Co-occurring simplicity and complexity preferences indicate tension "
                "between minimal-change and comprehensive-merge objectives."
            ),
            "claim_strength": "indirect RQ3 proxy",
        },
        {
            "fm3_subclaim": "Agents optimize different notions of a good merge",
            "proxy_family": "role_tension",
            "columns": (
                "structural_change_bias; modification_bias; feature_oriented; "
                "fix_oriented; refactor_oriented; test_oriented"
            ),
            "rationale": (
                "Bias and orientation labels capture competing success criteria, "
                "such as structural preservation, feature inclusion, fixes, and tests."
            ),
            "claim_strength": "broad proxy; use as sensitivity analysis",
        },
        {
            "fm3_subclaim": "Failure to converge in MIX traces",
            "proxy_family": "trace_non_convergence",
            "columns": "planner/output.txt; resolver/attempt_*/output.txt; reviewer/attempt_*/output.txt (legacy: resolution*.txt; review*.txt; review_feedback_history.txt; review_results.txt)",
            "rationale": (
                "Saved trace artifacts can directly show repeated review/revision cycles, "
                "rejection rationales, and drift across resolver attempts."
            ),
            "claim_strength": "direct trace evidence when artifacts exist",
        },
    ]
    return pd.DataFrame(rows)


def _collapse_aggregate(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_df.copy()
    df["id"] = _as_id(df["sample_id"])

    agg_map: dict[str, str] = {
        "file_count": "max",
        "total_labels": "sum",
        "unique_label_count": "max",
    }
    for col in LABEL_COLUMNS:
        if col in df.columns:
            agg_map[col] = "max"
        count_col = f"{col}_count"
        if count_col in df.columns:
            agg_map[count_col] = "sum"

    collapsed = df.groupby("id", as_index=False).agg(agg_map)
    sources = (
        df.groupby("id")["source_file"]
        .apply(lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
        .reset_index()
    )
    return collapsed.merge(sources, on="id", how="left")


def _collapse_complexity_preference(pref_df: pd.DataFrame) -> pd.DataFrame:
    df = pref_df.copy()
    df["id"] = _as_id(df["sample_id"])
    numeric_cols = [
        "simplicity_files",
        "complexity_files",
        "complexity_preference_files",
        "both_preference_labels_files",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    grouped = df.groupby("id", as_index=False)[numeric_cols].sum()
    dominant = (
        df.groupby("id")["dominant_preference"]
        .apply(lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
        .reset_index()
    )
    return grouped.merge(dominant, on="id", how="left")


def _load_label_tables(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load primary RQ3 labels and fill missing IDs from fail-only labels."""
    paired = _read_csv(paths.paired_csv).copy()
    paired["id"] = _as_id(paired["id"])
    paired["label_dataset"] = "rq3"

    aggregate = _collapse_aggregate(_read_csv(paths.aggregate_csv))
    aggregate["label_dataset_aggregate"] = "rq3"

    if paths.supplemental_paired_csv.exists():
        supplemental_paired = pd.read_csv(paths.supplemental_paired_csv)
        supplemental_paired["id"] = _as_id(supplemental_paired["id"])
        supplemental_paired = supplemental_paired[~supplemental_paired["id"].isin(set(paired["id"]))]
        if not supplemental_paired.empty:
            supplemental_paired["label_dataset"] = "rq3_fail_only"
            paired = pd.concat([paired, supplemental_paired], ignore_index=True, sort=False)

    if paths.supplemental_aggregate_csv.exists():
        supplemental_aggregate = _collapse_aggregate(pd.read_csv(paths.supplemental_aggregate_csv))
        supplemental_aggregate = supplemental_aggregate[
            ~supplemental_aggregate["id"].isin(set(aggregate["id"]))
        ]
        if not supplemental_aggregate.empty:
            supplemental_aggregate["label_dataset_aggregate"] = "rq3_fail_only"
            aggregate = pd.concat([aggregate, supplemental_aggregate], ignore_index=True, sort=False)

    return paired, aggregate


def _summarize_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = results_df.copy()
    df["id"] = _as_id(df["id"])
    df["exact_match"] = _as_float_bool(df["exact_match"])
    for metric in ["similarity", "bleu3", "rouge_l"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    is_bypass_mix = (df["eval_method"].astype(str) == "bypass7") & (
        df["bypass_method"].astype(str).str.upper() == "MIX"
    )
    is_force_mix = df["eval_method"].astype(str) == "force_mix"

    rows = []
    for sample_id, group in df.groupby("id"):
        bypass_mix = group[is_bypass_mix.loc[group.index]]
        force_mix = group[is_force_mix.loc[group.index]]
        row: dict[str, object] = {
            "id": sample_id,
            "is_bypass7_mix": not bypass_mix.empty,
            "bypass7_mix_file_rows": int(len(bypass_mix)),
            "bypass7_mix_files": int(bypass_mix["file_name"].nunique()) if "file_name" in bypass_mix else 0,
            "bypass7_mix_models": ";".join(sorted(set(bypass_mix["model_name"].dropna().astype(str)))),
            "is_force_mix_evaluated": not force_mix.empty,
            "force_mix_file_rows": int(len(force_mix)),
            "force_mix_models": ";".join(sorted(set(force_mix["model_name"].dropna().astype(str)))),
        }

        for method in ["agent", "bypass7", "force_mix"]:
            method_group = group[group["eval_method"].astype(str) == method]
            row[f"{method}_file_rows"] = int(len(method_group))
            for metric in METRICS:
                if metric in method_group.columns:
                    row[f"{method}_{metric}_mean"] = _safe_mean(method_group[metric])
        rows.append(row)

    case_cols = [
        "id",
        "repo",
        "file_name",
        "model_name",
        "difficulty",
        "project_size",
        "exact_match",
        "similarity",
        "bleu3",
        "rouge_l",
        "tokens_total",
        "processing_time_s",
    ]
    mix_cases = df.loc[is_bypass_mix, [c for c in case_cols if c in df.columns]].copy()
    mix_cases["mix_unit"] = "bypass7_file_trace"
    return pd.DataFrame(rows), mix_cases


def build_mix_label_table(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired, aggregate = _load_label_tables(paths)
    pref = _collapse_complexity_preference(_read_csv(paths.complexity_preference_csv))
    result_summary, mix_cases = _summarize_results(_read_csv(paths.mix_results_csv))

    paired = paired.copy()
    for label in LABEL_COLUMNS:
        if label not in paired.columns:
            paired[label] = 0
        paired[label] = _as_bool(paired[label]).astype(int)

    table = paired.merge(aggregate, on="id", how="left", suffixes=("", "_aggregate"))
    table = table.merge(pref, on="id", how="left")
    table = table.merge(result_summary, on="id", how="left")

    table["is_bypass7_mix"] = table["is_bypass7_mix"].fillna(False).astype(bool)
    table["is_force_mix_evaluated"] = table["is_force_mix_evaluated"].fillna(False).astype(bool)
    for col in [
        "bypass7_mix_file_rows",
        "bypass7_mix_files",
        "force_mix_file_rows",
        "simplicity_files",
        "complexity_files",
        "both_preference_labels_files",
    ]:
        if col not in table.columns:
            table[col] = 0
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)

    orientation_count = table[ORIENTATION_COLUMNS].sum(axis=1)
    table["fm3_ambiguity_incompleteness"] = (
        (table["unclear"] == 1)
        | (table["misprioritization"] == 1)
        | (table["lost_information_compression"] == 1)
    )
    table["fm3_simplicity_complexity_tension"] = (
        ((table["favored_simplicity"] == 1) & (table["favored_complexity"] == 1))
        | ((table["simplicity_files"] > 0) & (table["complexity_files"] > 0))
        | (table["both_preference_labels_files"] > 0)
        | table["dominant_preference"].fillna("").str.contains("tie \\(ambiguous\\)", regex=True)
    )
    table["fm3_role_tension"] = (
        (table["structural_change_bias"] == 1)
        | (table["modification_bias"] == 1)
        | (orientation_count >= 2)
    )
    table["fm3_narrow_proxy"] = table["fm3_ambiguity_incompleteness"]
    table["fm3_medium_proxy"] = table["fm3_narrow_proxy"] | table["fm3_simplicity_complexity_tension"]
    table["fm3_broad_proxy"] = table["fm3_medium_proxy"] | table["fm3_role_tension"]
    table["fm3_proxy_count"] = table[
        ["fm3_ambiguity_incompleteness", "fm3_simplicity_complexity_tension", "fm3_role_tension"]
    ].sum(axis=1)

    proxy_cols = [
        "fm3_ambiguity_incompleteness",
        "fm3_simplicity_complexity_tension",
        "fm3_role_tension",
        "fm3_narrow_proxy",
        "fm3_medium_proxy",
        "fm3_broad_proxy",
    ]
    for col in proxy_cols:
        table[col] = table[col].astype(bool)

    mix_cases = mix_cases.merge(
        table[
            [
                "id",
                "source_file",
                "label_dataset",
                "file_count",
                *LABEL_COLUMNS,
                "dominant_preference",
                "fm3_narrow_proxy",
                "fm3_medium_proxy",
                "fm3_broad_proxy",
                "fm3_proxy_count",
            ]
        ],
        on="id",
        how="left",
    )
    return table, mix_cases


def compute_prevalence(table: pd.DataFrame) -> pd.DataFrame:
    proxy_cols = [
        "fm3_ambiguity_incompleteness",
        "fm3_simplicity_complexity_tension",
        "fm3_role_tension",
        "fm3_narrow_proxy",
        "fm3_medium_proxy",
        "fm3_broad_proxy",
    ]
    rows = []
    for proxy in proxy_cols:
        mix = table["is_bypass7_mix"]
        pos = table[proxy]
        a = int((mix & pos).sum())
        b = int((mix & ~pos).sum())
        c = int((~mix & pos).sum())
        d = int((~mix & ~pos).sum())

        mix_values = pos[mix].astype(float)
        nonmix_values = pos[~mix].astype(float)
        diff = float(mix_values.mean() - nonmix_values.mean()) if len(mix_values) and len(nonmix_values) else math.nan
        ci_low, ci_high = _quantile_ci(_bootstrap_mean_diffs(mix_values, nonmix_values))

        if stats is not None:
            _, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
        else:
            fisher_p = math.nan

        rows.append(
            {
                "proxy": proxy,
                "n_mix": int(mix.sum()),
                "n_nonmix": int((~mix).sum()),
                "mix_positive": a,
                "nonmix_positive": c,
                "mix_prevalence": float(mix_values.mean()) if len(mix_values) else math.nan,
                "nonmix_prevalence": float(nonmix_values.mean()) if len(nonmix_values) else math.nan,
                "prevalence_diff": diff,
                "prevalence_diff_ci_low": ci_low,
                "prevalence_diff_ci_high": ci_high,
                "corrected_odds_ratio": _corrected_odds_ratio(a, b, c, d),
                "fisher_exact_p": float(fisher_p),
            }
        )
    return pd.DataFrame(rows)


def compute_outcomes(table: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_labeled": pd.Series(True, index=table.index),
        "bypass7_mix_instances": table["is_bypass7_mix"],
        "force_mix_evaluated_instances": table["is_force_mix_evaluated"],
    }
    proxies = ["fm3_narrow_proxy", "fm3_medium_proxy", "fm3_broad_proxy"]
    metric_cols = [
        "bypass_exact_match",
        "bypass_similarity",
        "bypass_bleu3",
        "bypass_rouge_l",
        "force_mix_exact_match_mean",
        "force_mix_similarity_mean",
        "force_mix_bleu3_mean",
        "force_mix_rouge_l_mean",
    ]

    rows = []
    for scope_name, scope_mask in scopes.items():
        scoped = table[scope_mask.fillna(False)].copy()
        if scoped.empty:
            continue
        for proxy in proxies:
            for metric in metric_cols:
                if metric not in scoped.columns:
                    continue
                values = pd.to_numeric(scoped[metric], errors="coerce")
                with_values = values[scoped[proxy]]
                without_values = values[~scoped[proxy]]
                diff_values = _bootstrap_mean_diffs(with_values.dropna(), without_values.dropna())
                ci_low, ci_high = _quantile_ci(diff_values)
                if stats is not None and len(with_values.dropna()) > 1 and len(without_values.dropna()) > 1:
                    p_value = float(stats.mannwhitneyu(with_values.dropna(), without_values.dropna(), alternative="two-sided").pvalue)
                else:
                    p_value = math.nan
                rows.append(
                    {
                        "scope": scope_name,
                        "proxy": proxy,
                        "metric": metric,
                        "n_proxy_positive": int(with_values.notna().sum()),
                        "n_proxy_negative": int(without_values.notna().sum()),
                        "mean_proxy_positive": _safe_mean(with_values),
                        "mean_proxy_negative": _safe_mean(without_values),
                        "mean_difference_positive_minus_negative": (
                            _safe_mean(with_values) - _safe_mean(without_values)
                            if len(with_values.dropna()) and len(without_values.dropna())
                            else math.nan
                        ),
                        "diff_ci_low": ci_low,
                        "diff_ci_high": ci_high,
                        "mannwhitney_p": p_value,
                    }
                )
    return pd.DataFrame(rows)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _iter_trace_files(trace_root: Path) -> Iterable[Path]:
    """Yield plausible persisted trace artifacts without scanning dependency trees."""
    # New per-agent layout
    wanted_agent_dirs = {"planner", "resolver", "reviewer"}
    # Legacy flat filenames
    wanted_names = {"review_feedback_history.txt", "review_results.txt", "agent_plan.txt", "plan.txt"}
    wanted_prefixes = ("resolution", "review")
    skip_dirs = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

    if not trace_root.exists():
        return

    for root, dirs, files in os.walk(trace_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        root_path = Path(root)
        agent_name = root_path.name
        parent_name = root_path.parent.name if root_path.parent else ""

        for name in files:
            path = root_path / name
            # New layout: planner/output.txt, resolver/attempt_N/output.txt, reviewer/...
            if name == "output.txt" and (
                agent_name in wanted_agent_dirs
                or parent_name in wanted_agent_dirs
            ):
                yield path
                continue
            if name in wanted_names or (
                name.endswith(".txt") and name.startswith(wanted_prefixes)
            ):
                yield path


def audit_traces(table: pd.DataFrame, mix_cases: pd.DataFrame, trace_root: Path) -> pd.DataFrame:
    mix_ids = set(table.loc[table["is_bypass7_mix"], "id"].astype(str))
    trace_files = list(_iter_trace_files(trace_root))

    files_by_id: dict[str, list[Path]] = {sample_id: [] for sample_id in mix_ids}
    for path in trace_files:
        path_text = str(path)
        for sample_id in mix_ids:
            if sample_id in path_text:
                files_by_id[sample_id].append(path)

    rows = []
    base_cases = mix_cases[["id", "file_name"]].drop_duplicates() if not mix_cases.empty else pd.DataFrame()
    if base_cases.empty:
        base_cases = table.loc[table["is_bypass7_mix"], ["id"]].assign(file_name="")

    for _, case in base_cases.iterrows():
        sample_id = str(case["id"])
        file_name = str(case.get("file_name", ""))
        files = files_by_id.get(sample_id, [])
        resolution_files = [
            p
            for p in files
            if re.search(r"resolution\d+\.txt$", p.name)
            or (
                p.name == "output.txt"
                and ("resolver" in p.parts)
            )
        ]
        review_files = [
            p
            for p in files
            if re.search(r"review\d+\.txt$", p.name)
            or (
                p.name == "output.txt"
                and ("reviewer" in p.parts)
            )
        ]
        feedback_files = [p for p in files if p.name == "review_feedback_history.txt"]
        plan_files = [
            p
            for p in files
            if p.name in {"plan.txt", "agent_plan.txt"}
            or (p.name == "output.txt" and "planner" in p.parts)
        ]
        review_text = "\n\n".join(_read_text(p) for p in review_files + feedback_files)

        keyword_hits = {
            name: bool(re.search(pattern, review_text, flags=re.IGNORECASE | re.MULTILINE))
            for name, pattern in TRACE_KEYWORDS.items()
        }

        rows.append(
            {
                "id": sample_id,
                "file_name": file_name,
                "artifacts_found": bool(files),
                "n_plan_files": len(plan_files),
                "n_resolution_files": len(resolution_files),
                "n_review_files": len(review_files),
                "n_feedback_history_files": len(feedback_files),
                "hit_review_loop_cap_proxy": len(resolution_files) >= 3 or len(review_files) >= 3,
                "trace_paths": ";".join(str(p) for p in sorted(files)),
                "trace_caveat": "" if files else "No persisted trace artifacts found in checkout; regenerate or locate archived run outputs.",
                **keyword_hits,
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{100 * value:.1f}%"


def write_summary(
    output_path: Path,
    table: pd.DataFrame,
    mix_cases: pd.DataFrame,
    prevalence: pd.DataFrame,
    outcomes: pd.DataFrame,
    trace_evidence: pd.DataFrame,
) -> None:
    total_mix_instances = int(mix_cases["id"].nunique()) if not mix_cases.empty else 0
    total_mix_file_rows = int(len(mix_cases))
    labeled_mix = table[table["is_bypass7_mix"]]
    mix_instances = int(len(labeled_mix))
    mix_file_rows = int(labeled_mix["bypass7_mix_file_rows"].sum())
    force_mix_instances = int(table["is_force_mix_evaluated"].sum())
    trace_artifacts = int(trace_evidence["artifacts_found"].sum()) if not trace_evidence.empty else 0

    broad_row = prevalence[prevalence["proxy"] == "fm3_broad_proxy"].iloc[0]
    narrow_row = prevalence[prevalence["proxy"] == "fm3_narrow_proxy"].iloc[0]
    medium_row = prevalence[prevalence["proxy"] == "fm3_medium_proxy"].iloc[0]

    mix_case_cols = [
        "id",
        "file_name",
        "exact_match",
        "similarity",
        "difficulty",
        "project_size",
        "label_dataset",
        "fm3_narrow_proxy",
        "fm3_medium_proxy",
        "fm3_broad_proxy",
        "dominant_preference",
    ]
    visible_cases = mix_cases[[c for c in mix_case_cols if c in mix_cases.columns]].copy()
    if "label_dataset" in visible_cases.columns:
        visible_cases["label_dataset"] = visible_cases["label_dataset"].fillna("unlabeled")
    for col in ["fm3_narrow_proxy", "fm3_medium_proxy", "fm3_broad_proxy", "dominant_preference"]:
        if col in visible_cases.columns:
            visible_cases[col] = visible_cases[col].fillna("NA")

    lines = [
        "# FM3 Role-Boundary Ambiguity Analysis",
        "",
        "## Construct Mapping",
        "",
        "The repository does not contain a direct `FM3` or `role-boundary ambiguity` label. "
        "This analysis treats RQ3 labels as pre-specified proxies: a narrow ambiguity/incompleteness "
        "proxy, a medium proxy that adds simplicity-vs-complexity tension, and a broad proxy that "
        "also includes structural/modification/orientation tension.",
        "",
        "Prompt evidence supports the construct definition: the force-MIX planner asks for the "
        "simplest strategy and minimal change sets, the resolver is instructed to keep edits minimal "
        "and implement exactly the plan, while the reviewer checks partial merges and plan compliance.",
        "",
        "## MIX Coverage",
        "",
        f"- Labeled RQ3 instances analyzed: {len(table)}.",
        f"- Bypass7 MIX rows in result data: {total_mix_instances} unique merge instances and {total_mix_file_rows} file-level MIX traces.",
        f"- RQ3-labeled Bypass7 MIX overlap: {mix_instances} unique merge instances and {mix_file_rows} file-level MIX traces.",
        f"- Force-MIX evaluated instances available for secondary comparison: {force_mix_instances}.",
        "",
        "## Proxy Prevalence In MIX",
        "",
        (
            f"- Narrow proxy prevalence: MIX {_fmt_pct(float(narrow_row['mix_prevalence']))} vs "
            f"non-MIX {_fmt_pct(float(narrow_row['nonmix_prevalence']))}; "
            f"Fisher p={float(narrow_row['fisher_exact_p']):.4g}."
        ),
        (
            f"- Medium proxy prevalence: MIX {_fmt_pct(float(medium_row['mix_prevalence']))} vs "
            f"non-MIX {_fmt_pct(float(medium_row['nonmix_prevalence']))}; "
            f"Fisher p={float(medium_row['fisher_exact_p']):.4g}."
        ),
        (
            f"- Broad proxy prevalence: MIX {_fmt_pct(float(broad_row['mix_prevalence']))} vs "
            f"non-MIX {_fmt_pct(float(broad_row['nonmix_prevalence']))}; "
            f"Fisher p={float(broad_row['fisher_exact_p']):.4g}."
        ),
        "",
        "## Trace Evidence",
        "",
    ]

    if trace_artifacts:
        capped = int(trace_evidence["hit_review_loop_cap_proxy"].sum())
        lines.extend(
            [
                f"Persisted artifacts were found for {trace_artifacts} MIX file rows; {capped} show a trace-count proxy for reaching the review-loop cap.",
                "Use `fm3_trace_evidence.csv` for keyword hits in reviewer feedback and resolution/review counts.",
            ]
        )
    else:
        lines.extend(
            [
                "No persisted planner/resolver/reviewer ``output.txt`` (or legacy "
                "``resolution*.txt`` / ``review*.txt`` / ``agent_plan.txt``) artifacts "
                "for the MIX cases were found in this checkout.",
                "The iteration/oscillation part of FM3 therefore remains a trace-level claim until archived outputs are located or the runs are regenerated with existing artifact persistence.",
            ]
        )

    lines.extend(
        [
            "",
            "## Paper-Ready Interpretation",
            "",
            "The RQ3 labels can empirically substantiate FM3 as a proxy-based, correlational claim: "
            "MIX cases can be audited for enrichment in ambiguity/incompleteness labels and "
            "simplicity-complexity tension, then linked to final correctness metrics. The strongest "
            "wording is that the labels are consistent with role-boundary ambiguity, while direct "
            "evidence for oscillation across all three iterations requires the saved role traces.",
            "",
            "## Bypass7 MIX Case Audit",
            "",
        ]
    )

    if visible_cases.empty:
        lines.append("No Bypass7 MIX cases were found after joining results to RQ3 labels.")
    else:
        lines.append(visible_cases.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `fm3_proxy_mapping.csv`",
            "- `fm3_mix_label_table.csv`",
            "- `fm3_proxy_prevalence.csv`",
            "- `fm3_proxy_outcomes.csv`",
            "- `fm3_mix_case_audit.csv`",
            "- `fm3_trace_evidence.csv`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(paths: Paths) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    proxy_mapping = build_proxy_mapping()
    table, mix_cases = build_mix_label_table(paths)
    prevalence = compute_prevalence(table)
    outcomes = compute_outcomes(table)
    trace_evidence = audit_traces(table, mix_cases, paths.trace_root)

    outputs = {
        "proxy_mapping": paths.output_dir / "fm3_proxy_mapping.csv",
        "mix_label_table": paths.output_dir / "fm3_mix_label_table.csv",
        "proxy_prevalence": paths.output_dir / "fm3_proxy_prevalence.csv",
        "proxy_outcomes": paths.output_dir / "fm3_proxy_outcomes.csv",
        "mix_case_audit": paths.output_dir / "fm3_mix_case_audit.csv",
        "trace_evidence": paths.output_dir / "fm3_trace_evidence.csv",
        "summary": paths.output_dir / "fm3_role_boundary_summary.md",
    }

    proxy_mapping.to_csv(outputs["proxy_mapping"], index=False)
    table.to_csv(outputs["mix_label_table"], index=False)
    prevalence.to_csv(outputs["proxy_prevalence"], index=False)
    outcomes.to_csv(outputs["proxy_outcomes"], index=False)
    mix_cases.to_csv(outputs["mix_case_audit"], index=False)
    trace_evidence.to_csv(outputs["trace_evidence"], index=False)
    write_summary(outputs["summary"], table, mix_cases, prevalence, outcomes, trace_evidence)
    return outputs


def parse_args() -> Paths:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", type=Path, default=Paths.paired_csv)
    parser.add_argument("--aggregate-csv", type=Path, default=Paths.aggregate_csv)
    parser.add_argument("--supplemental-paired-csv", type=Path, default=Paths.supplemental_paired_csv)
    parser.add_argument("--supplemental-aggregate-csv", type=Path, default=Paths.supplemental_aggregate_csv)
    parser.add_argument("--complexity-preference-csv", type=Path, default=Paths.complexity_preference_csv)
    parser.add_argument("--mix-results-csv", type=Path, default=Paths.mix_results_csv)
    parser.add_argument("--output-dir", type=Path, default=Paths.output_dir)
    parser.add_argument("--trace-root", type=Path, default=Paths.trace_root)
    args = parser.parse_args()
    return Paths(
        paired_csv=args.paired_csv,
        aggregate_csv=args.aggregate_csv,
        supplemental_paired_csv=args.supplemental_paired_csv,
        supplemental_aggregate_csv=args.supplemental_aggregate_csv,
        complexity_preference_csv=args.complexity_preference_csv,
        mix_results_csv=args.mix_results_csv,
        output_dir=args.output_dir,
        trace_root=args.trace_root,
    )


def main() -> None:
    outputs = run_analysis(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
