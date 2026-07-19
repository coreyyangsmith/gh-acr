"""Audit agent raw_model_outputs coverage against the full benchmark.

Compares each model under data/raw_model_outputs to
data/git_good_bench_merge_commits_all.csv for the ``agent`` eval method only
(bypass7 ignored). Writes one gap CSV per model plus a summary.

Gap categories
--------------
not_processed   – scenario and/or agent dir missing entirely
incomplete      – agent dir exists but one or more conflict file outputs missing
empty_output    – at least one agent output file is 0 bytes
invalid_output  – whole-file looks like an API/LLM error payload (not source)
missing_results – artifacts look fine but scenario absent from results CSV
results_only    – present in results CSV but raw agent artifacts missing/bad
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw_model_outputs")
BENCH = Path("data/git_good_bench_merge_commits_all.csv")
OUT_DIR = Path("data/agent_coverage_gaps")

MODELS = {
    "openai_gpt-5-nano": {
        "results": Path("data/2025_09_29_gptnano_results_combined.csv"),
        "label": "gpt-5-nano",
    },
    "llama-3.1-8b": {
        "results": Path("data/2026_01_18_llama_results_combined_final.csv"),
        "label": "llama-3.1-8b",
    },
    "groq_qwen_qwen3-32b": {
        "results": Path("data/2025_09_29_qwen32_results_combined.csv"),
        "label": "qwen3-32b",
    },
}

# Whole-file / header patterns only (avoid matching strings inside real source).
WHOLE_FILE_ERROR_RES = [
    re.compile(r"^\s*traceback \(most recent call last\)", re.I | re.M),
    re.compile(r"^\s*\{[^{}]{0,400}\"(error|message|type)\"\s*:", re.I),
    re.compile(r"^\s*(openai|anthropic|httpx|requests)\.\w*error", re.I),
    re.compile(r"^\s*error code:\s*\d+", re.I),
    re.compile(r"^\s*rate limit", re.I),
    re.compile(r"context[_ ]length[_ ]exceeded", re.I),
    re.compile(r"maximum context length", re.I),
]


def parse_scenario(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return {}


def file_path_to_slug(file_path: str) -> str:
    return str(file_path).replace("/", "_").replace("\\", "_")


def find_agent_output(agent_dir: Path, file_path: str) -> Path | None:
    slug = file_path_to_slug(file_path)
    candidates = [
        agent_dir / f"{slug}.txt",
        agent_dir / slug / "final" / "resolved.txt",
        agent_dir / slug / "resolved.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    if agent_dir.is_dir():
        for p in agent_dir.glob(f"{slug}*"):
            if p.is_file() and p.suffix == ".txt":
                return p
            if p.is_dir():
                for nested in [
                    p / "final" / "resolved.txt",
                    p / "resolved.txt",
                    p / f"{slug}.txt",
                ]:
                    if nested.is_file():
                        return nested
    return None


def classify_output(path: Path | None) -> tuple[str, int]:
    """Return (status, size_bytes)."""
    if path is None:
        return "missing_file", 0
    try:
        size = path.stat().st_size
    except OSError:
        return "unreadable", 0
    if size == 0:
        return "empty_file", 0
    try:
        # Only inspect the start of the file for whole-file error payloads.
        text = path.read_bytes()[:4096].decode("utf-8", errors="replace")
    except OSError:
        return "unreadable", size

    stripped = text.strip()
    if len(stripped) < 5:
        return "too_short", size

    # Whole-file error: short file dominated by error boilerplate, or header match.
    head = stripped[:800]
    if any(rx.search(head) for rx in WHOLE_FILE_ERROR_RES):
        # Large source files that merely mention errors in a docstring near top
        # are uncommon with these header-anchored patterns; still guard by size.
        if size < 2500 or stripped.startswith("{") or stripped.lower().startswith("traceback"):
            return "error_payload", size

    return "ok", size


def load_bench() -> pd.DataFrame:
    df = pd.read_csv(BENCH)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "scenario_id"})
    else:
        df["scenario_id"] = df["id"]
    df["scenario_id"] = df["scenario_id"].astype(str)
    df["scenario_json"] = df["scenario"].map(parse_scenario)
    df["conflict_files"] = df["scenario_json"].map(
        lambda s: list(s.get("files_in_merge_conflict") or []) if isinstance(s, dict) else []
    )
    df["n_conflict_files"] = df["conflict_files"].map(len)
    return df


def load_agent_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "eval_method" in df.columns:
        df = df[df["eval_method"] == "agent"].copy()
    df["id"] = df["id"].astype(str)
    return df


def primary_gap_category(issues: list[str]) -> str:
    if "missing_scenario_dir" in issues or "missing_agent_dir" in issues:
        return "not_processed"
    if "missing_agent_output_file" in issues:
        return "incomplete"
    if "empty_file" in issues or "too_short" in issues:
        return "empty_output"
    if "error_payload" in issues or "unreadable" in issues:
        return "invalid_output"
    if "missing_from_results_csv" in issues:
        return "missing_results"
    return "other"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = load_bench()
    print(f"Benchmark scenarios: {len(bench)}")

    summary_rows = []
    verify_rows = []

    for model_dir, meta in MODELS.items():
        model_root = ROOT / model_dir
        results = load_agent_results(meta["results"])
        result_ids = set(results["id"]) if not results.empty else set()

        if not results.empty and "exact_match" in results.columns:
            agg_kwargs = {
                "n_result_rows": ("id", "size"),
                "n_exact_true": (
                    "exact_match",
                    lambda s: int((s.astype(str).str.lower() == "true").sum()),
                ),
                "n_exact_false": (
                    "exact_match",
                    lambda s: int((s.astype(str).str.lower() == "false").sum()),
                ),
            }
            if "similarity" in results.columns:
                agg_kwargs["mean_similarity"] = ("similarity", "mean")
            grouped = results.groupby("id", as_index=False).agg(**agg_kwargs)
            result_meta = grouped.set_index("id").to_dict(orient="index")
        else:
            result_meta = {}

        gap_rows = []
        status_counter: Counter[str] = Counter()
        issue_counter: Counter[str] = Counter()
        category_counter: Counter[str] = Counter()

        n_files_checked = 0
        n_files_ok = 0
        n_files_empty = 0
        n_files_missing = 0
        n_files_invalid = 0
        total_ok_bytes = 0

        for _, row in bench.iterrows():
            sid = row["scenario_id"]
            conflict_files = row["conflict_files"]
            scenario_path = model_root / sid
            agent_dir = scenario_path / "agent"

            issues: list[str] = []
            missing_files: list[str] = []
            bad_files: list[str] = []
            ok_files: list[str] = []
            ok_sizes: list[int] = []

            if not scenario_path.is_dir():
                issues.append("missing_scenario_dir")
                n_files_missing += len(conflict_files) or 1
            elif not agent_dir.is_dir():
                issues.append("missing_agent_dir")
                n_files_missing += len(conflict_files) or 1
            else:
                for fp in conflict_files:
                    n_files_checked += 1
                    out = find_agent_output(agent_dir, fp)
                    st, size = classify_output(out)
                    if st == "ok":
                        ok_files.append(fp)
                        ok_sizes.append(size)
                        n_files_ok += 1
                        total_ok_bytes += size
                    elif st == "missing_file":
                        missing_files.append(fp)
                        issues.append("missing_agent_output_file")
                        n_files_missing += 1
                    elif st == "empty_file" or st == "too_short":
                        bad_files.append(f"{fp}:{st}")
                        issues.append(st if st == "empty_file" else "too_short")
                        n_files_empty += 1
                    else:
                        bad_files.append(f"{fp}:{st}")
                        issues.append(st)
                        n_files_invalid += 1

                if not conflict_files:
                    issues.append("no_conflict_files_in_benchmark")

            in_results = sid in result_ids
            if not in_results:
                issues.append("missing_from_results_csv")

            rm = result_meta.get(sid, {})

            seen: set[str] = set()
            uniq_issues: list[str] = []
            for i in issues:
                if i not in seen:
                    seen.add(i)
                    uniq_issues.append(i)

            # Artifacts complete & valid if every conflict file has ok output
            artifacts_complete = (
                scenario_path.is_dir()
                and agent_dir.is_dir()
                and bool(conflict_files)
                and len(ok_files) == len(conflict_files)
                and not bad_files
                and not missing_files
            )

            if not uniq_issues:
                status_counter["complete_ok"] += 1
                category_counter["complete_ok"] += 1
            else:
                # Drop "missing_from_results_csv" alone when artifacts are broken —
                # primary category should reflect artifact problems first.
                cat = primary_gap_category(uniq_issues)
                # If artifacts are fine and only results missing:
                if artifacts_complete and uniq_issues == ["missing_from_results_csv"]:
                    cat = "missing_results"
                category_counter[cat] += 1
                for i in uniq_issues:
                    issue_counter[i] += 1
                gap_rows.append(
                    {
                        "gap_category": cat,
                        "scenario_id": sid,
                        "repo": row["name"],
                        "difficulty": row.get("difficulty", ""),
                        "project_size": row.get("project_size", ""),
                        "n_conflict_files": row["n_conflict_files"],
                        "conflict_files": "|".join(conflict_files),
                        "has_scenario_dir": scenario_path.is_dir(),
                        "has_agent_dir": agent_dir.is_dir(),
                        "artifacts_complete": artifacts_complete,
                        "n_ok_outputs": len(ok_files),
                        "n_missing_outputs": len(missing_files),
                        "n_bad_outputs": len(bad_files),
                        "missing_output_files": "|".join(missing_files),
                        "bad_output_files": "|".join(bad_files),
                        "ok_output_files": "|".join(ok_files),
                        "ok_output_bytes_sum": int(sum(ok_sizes)),
                        "in_results_csv": in_results,
                        "n_result_rows": rm.get("n_result_rows", 0),
                        "n_exact_true": rm.get("n_exact_true", 0),
                        "n_exact_false": rm.get("n_exact_false", 0),
                        "issues": "|".join(uniq_issues),
                        "model_dir": model_dir,
                        "model_label": meta["label"],
                    }
                )

        out_csv = OUT_DIR / f"{meta['label']}_agent_missing_or_failed.csv"
        gap_df = pd.DataFrame(gap_rows)
        if not gap_df.empty:
            cat_order = {
                "not_processed": 0,
                "incomplete": 1,
                "empty_output": 2,
                "invalid_output": 3,
                "missing_results": 4,
                "other": 5,
            }
            gap_df["_ord"] = gap_df["gap_category"].map(lambda c: cat_order.get(c, 9))
            gap_df = gap_df.sort_values(
                ["_ord", "has_scenario_dir", "has_agent_dir", "n_ok_outputs", "scenario_id"],
                ascending=[True, True, True, True, True],
            ).drop(columns=["_ord"])
        gap_df.to_csv(out_csv, index=False)

        # Also write a strictly "needs reprocessing" file (exclude missing_results-only
        # if you only care about re-runs — keep both in main file; this is the actionable set)
        reprocess = (
            gap_df[gap_df["gap_category"].isin(["not_processed", "incomplete", "empty_output", "invalid_output"])]
            if not gap_df.empty
            else gap_df
        )
        reprocess_csv = OUT_DIR / f"{meta['label']}_agent_needs_reprocess.csv"
        reprocess.to_csv(reprocess_csv, index=False)

        verify = {
            "model_label": meta["label"],
            "model_dir": model_dir,
            "benchmark_total": len(bench),
            "scenario_dirs_present": int(
                sum(1 for sid in bench["scenario_id"] if (model_root / sid).is_dir())
            ),
            "agent_dirs_present": int(
                sum(1 for sid in bench["scenario_id"] if (model_root / sid / "agent").is_dir())
            ),
            "complete_ok_scenarios": int(status_counter["complete_ok"]),
            "gap_scenarios": len(gap_df),
            "needs_reprocess": len(reprocess),
            "files_checked": n_files_checked,
            "files_ok": n_files_ok,
            "files_missing": n_files_missing,
            "files_empty": n_files_empty,
            "files_invalid": n_files_invalid,
            "ok_output_bytes_total": total_ok_bytes,
            "ok_output_bytes_avg": round(total_ok_bytes / n_files_ok, 1) if n_files_ok else 0,
            "results_csv_agent_unique_ids": len(result_ids),
            "category_breakdown": dict(category_counter),
            "issue_breakdown": dict(issue_counter),
            "gap_csv": str(out_csv),
            "reprocess_csv": str(reprocess_csv),
        }
        verify_rows.append(verify)
        summary_rows.append(verify)

        print(f"\n=== {meta['label']} ===")
        printable = {k: v for k, v in verify.items() if k not in ("category_breakdown", "issue_breakdown")}
        print(json.dumps(printable, indent=2))
        print("category_breakdown:", dict(category_counter))
        print("issue_breakdown:", dict(issue_counter))

    sum_path = OUT_DIR / "agent_coverage_summary.csv"
    pd.DataFrame(
        [
            {
                **{k: v for k, v in s.items() if k not in ("category_breakdown", "issue_breakdown")},
                "category_breakdown": json.dumps(s["category_breakdown"]),
                "issue_breakdown": json.dumps(s["issue_breakdown"]),
            }
            for s in summary_rows
        ]
    ).to_csv(sum_path, index=False)

    verify_md = OUT_DIR / "agent_verification_report.md"
    lines = [
        "# Agent raw_model_outputs verification",
        "",
        f"Benchmark: `{BENCH}` ({len(bench)} scenarios).",
        "Scope: `eval_method=agent` only (bypass7 ignored).",
        "",
        "## Per-model summary",
        "",
    ]
    for s in summary_rows:
        lines.extend(
            [
                f"### {s['model_label']} (`{s['model_dir']}`)",
                "",
                f"- Scenario dirs: **{s['scenario_dirs_present']}** / {s['benchmark_total']}",
                f"- Agent dirs: **{s['agent_dirs_present']}** / {s['benchmark_total']}",
                f"- Fully OK (artifacts + results): **{s['complete_ok_scenarios']}**",
                f"- Gap rows: **{s['gap_scenarios']}** (needs reprocess: **{s['needs_reprocess']}**)",
                f"- File-level: ok={s['files_ok']}, missing={s['files_missing']}, "
                f"empty={s['files_empty']}, invalid={s['files_invalid']} "
                f"(checked={s['files_checked']})",
                f"- Avg OK output size: {s['ok_output_bytes_avg']} bytes",
                f"- Results CSV unique agent IDs: {s['results_csv_agent_unique_ids']}",
                f"- Categories: `{json.dumps(s['category_breakdown'])}`",
                f"- Outputs: `{s['gap_csv']}`, `{s['reprocess_csv']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Legitimacy checks applied",
            "",
            "1. Every benchmark scenario ID should have `raw_model_outputs/<model>/<id>/agent/`.",
            "2. Each `files_in_merge_conflict` entry should map to a non-empty agent output "
            "(`<slug>.txt` or `<slug>/final/resolved.txt`).",
            "3. Reject 0-byte / tiny outputs and whole-file API/traceback payloads.",
            "4. Do **not** flag ordinary source that merely contains words like `failed to` "
            "or `Exception:` (common false positives).",
            "5. Cross-check presence in the corresponding combined results CSV (`eval_method=agent`).",
            "",
        ]
    )
    verify_md.write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote summary:", sum_path)
    print("Wrote verification report:", verify_md)
    print("Gap CSVs in:", OUT_DIR)


if __name__ == "__main__":
    main()
