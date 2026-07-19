"""Remap legacy raw_model_outputs / nan artifacts into the current on-disk layout.

Legacy layout (numeric DF index as scenario key)::

    data/raw_model_outputs/<model>/<numeric_id>/
      agent/<file_slug>.txt
      bypass7/<file_slug>/{a_summary,b_summary,plan,resolution1,...}.txt
      <file_slug>/{original,a,b,ground_truth}.txt

New layout (benchmark ``id`` slug, possibly containing ``/``)::

    data/new_<model>/<owner>/<repo-merge-XXXXX>/
      agent/<file_slug>/{agent/output.txt, final/resolved.txt}
      bypass7/<file_slug>/{summarizer/{a,b}/output.txt, resolver/attempt_1/..., final/...}
      <file_slug>/{original,a,b,ground_truth}.txt

Numeric IDs are resolved via the leading index column of
``data/git_good_bench_merge_commits_all.csv``.

Example
-------
uv run python scripts/remap_raw_outputs_to_new_layout.py
uv run python scripts/remap_raw_outputs_to_new_layout.py --models llama --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

BENCH = Path("data/git_good_bench_merge_commits_all.csv")
RAW_ROOT = Path("data/raw_model_outputs")
NAN_ROOT = Path("data/nan")

MODEL_SOURCES = {
    "llama": RAW_ROOT / "llama-3.1-8b",
    "qwen": RAW_ROOT / "groq_qwen_qwen3-32b",
    "nano": RAW_ROOT / "openai_gpt-5-nano",
}

METHOD_DIRS = {"agent", "bypass7", "bypass", "better_judge", "base_a", "base_b", "prep"}
SHARED_INPUT_NAMES = {
    "original.txt",
    "a.txt",
    "b.txt",
    "a.diff",
    "b.diff",
    "ground_truth.txt",
    "ground_truth.diff",
    "a_commit_message.txt",
    "b_commit_message.txt",
}


def load_numeric_to_slug(bench_csv: Path) -> dict[str, str]:
    df = pd.read_csv(bench_csv, index_col=0)
    mapping = {str(idx): str(slug) for idx, slug in zip(df.index, df["id"])}
    return mapping


def link_or_copy(src: Path, dst: Path, *, dry_run: bool = False) -> str:
    """Prefer hardlink; fall back to copy. Returns action taken."""
    if dry_run:
        return "dry-run"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_text(dst: Path, text: str, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def remap_agent(src_agent: Path, dst_agent: Path, *, dry_run: bool = False) -> int:
    """Map legacy flat agent/<slug>.txt → agent/<slug>/{final/resolved,agent/output}.txt."""
    n = 0
    if not src_agent.is_dir():
        return 0
    for item in src_agent.iterdir():
        if item.is_file() and item.suffix == ".txt":
            slug = item.stem
            for rel in (Path(slug) / "final" / "resolved.txt", Path(slug) / "agent" / "output.txt"):
                link_or_copy(item, dst_agent / rel, dry_run=dry_run)
            n += 1
        elif item.is_dir():
            # Already nested or partial new layout — copy tree under same slug.
            for f in item.rglob("*"):
                if f.is_file():
                    link_or_copy(f, dst_agent / item.name / f.relative_to(item), dry_run=dry_run)
                    n += 1
    return n


def _pick_resolved(file_dir: Path) -> Path | None:
    slug = file_dir.name
    candidates = [
        file_dir / f"bypass_{slug}.txt",
        file_dir / "resolution1.txt",
    ]
    # Also accept any bypass_*.txt that isn't a diff / final_diff
    for p in sorted(file_dir.glob("bypass_*.txt")):
        name = p.name
        if name.endswith("_final_diff.txt") or name.endswith(".diff"):
            continue
        if p not in candidates:
            candidates.append(p)
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c
    for c in candidates:
        if c.is_file():
            return c
    return None


def _pick_final_diff(file_dir: Path) -> Path | None:
    slug = file_dir.name
    for c in (
        file_dir / "final_diff.txt",
        file_dir / f"bypass_{slug}_final_diff.txt",
        *sorted(file_dir.glob("bypass_*_final_diff.txt")),
    ):
        if c.is_file():
            return c
    return None


def remap_bypass7(src_bypass: Path, dst_bypass: Path, *, dry_run: bool = False) -> int:
    """Map legacy flat bypass7 file dirs into nested agent-call / final layout."""
    n = 0
    if not src_bypass.is_dir():
        return 0

    per_file_plans: dict[str, str] = {}

    for file_dir in src_bypass.iterdir():
        if not file_dir.is_dir():
            # Unexpected top-level files — keep under _misc
            if file_dir.is_file():
                link_or_copy(file_dir, dst_bypass / "_misc" / file_dir.name, dry_run=dry_run)
                n += 1
            continue

        slug = file_dir.name
        dst_file = dst_bypass / slug

        a_sum = file_dir / "a_summary.txt"
        b_sum = file_dir / "b_summary.txt"
        if a_sum.is_file():
            link_or_copy(a_sum, dst_file / "summarizer" / "a" / "output.txt", dry_run=dry_run)
            n += 1
        if b_sum.is_file():
            link_or_copy(b_sum, dst_file / "summarizer" / "b" / "output.txt", dry_run=dry_run)
            n += 1

        plan = file_dir / "plan.txt"
        if plan.is_file():
            text = plan.read_text(encoding="utf-8", errors="replace")
            per_file_plans[slug] = text
            link_or_copy(plan, dst_file / "planner" / "output.txt", dry_run=dry_run)
            n += 1

        res1 = file_dir / "resolution1.txt"
        if res1.is_file():
            link_or_copy(res1, dst_file / "resolver" / "attempt_1" / "output.txt", dry_run=dry_run)
            n += 1

        resolved = _pick_resolved(file_dir)
        if resolved is not None:
            link_or_copy(resolved, dst_file / "final" / "resolved.txt", dry_run=dry_run)
            # If it came from a bypass_* file, also mirror under bypass/
            if resolved.name.startswith("bypass_"):
                link_or_copy(resolved, dst_file / "bypass" / "output.txt", dry_run=dry_run)
            n += 1

        fdiff = _pick_final_diff(file_dir)
        if fdiff is not None:
            link_or_copy(fdiff, dst_file / "final" / "final_diff.txt", dry_run=dry_run)
            n += 1

        # Preserve any remaining legacy files under legacy/
        known = {
            "a_summary.txt",
            "b_summary.txt",
            "plan.txt",
            "resolution1.txt",
            "final_diff.txt",
        }
        for f in file_dir.iterdir():
            if not f.is_file():
                continue
            if f.name in known:
                continue
            if f.name.startswith("bypass_") and (
                f.name.endswith(".txt") or f.name.endswith(".diff")
            ):
                # already mapped primary outputs; keep a copy under legacy/
                link_or_copy(f, dst_file / "legacy" / f.name, dry_run=dry_run)
                n += 1
                continue
            link_or_copy(f, dst_file / "legacy" / f.name, dry_run=dry_run)
            n += 1

    if per_file_plans:
        # Scenario-level planner JSON (keys are file slugs; best-effort).
        payload = json.dumps(per_file_plans, ensure_ascii=False, indent=2)
        write_text(dst_bypass / "planner" / "output.txt", payload, dry_run=dry_run)

    return n


def remap_shared_inputs(src_scenario: Path, dst_scenario: Path, *, dry_run: bool = False) -> int:
    n = 0
    for child in src_scenario.iterdir():
        if not child.is_dir():
            continue
        if child.name in METHOD_DIRS:
            continue
        # File-slug dirs with shared conflict inputs
        has_shared = any((child / name).is_file() for name in SHARED_INPUT_NAMES)
        if not has_shared:
            continue
        for f in child.iterdir():
            if f.is_file():
                link_or_copy(f, dst_scenario / child.name / f.name, dry_run=dry_run)
                n += 1
    return n


def remap_method_passthrough(
    src_method: Path, dst_method: Path, *, dry_run: bool = False
) -> int:
    """Copy base_a / base_b trees as-is."""
    n = 0
    if not src_method.is_dir():
        return 0
    for f in src_method.rglob("*"):
        if f.is_file():
            link_or_copy(f, dst_method / f.relative_to(src_method), dry_run=dry_run)
            n += 1
    return n


def remap_scenario(
    src: Path,
    dst: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    counts = {"agent": 0, "bypass7": 0, "shared": 0, "base": 0, "other": 0}
    if src_agent := src / "agent":
        counts["agent"] = remap_agent(src_agent, dst / "agent", dry_run=dry_run)
    if (src / "bypass7").is_dir():
        counts["bypass7"] = remap_bypass7(src / "bypass7", dst / "bypass7", dry_run=dry_run)
    elif (src / "bypass").is_dir():
        counts["bypass7"] = remap_bypass7(src / "bypass", dst / "bypass7", dry_run=dry_run)
    for base in ("base_a", "base_b", "prep"):
        if (src / base).is_dir():
            counts["base"] += remap_method_passthrough(src / base, dst / base, dry_run=dry_run)
    counts["shared"] = remap_shared_inputs(src, dst, dry_run=dry_run)
    return counts


def iter_nan_scenarios(nan_root: Path, numeric_to_slug: dict[str, str]):
    """Yield (src_dir, slug_id, kind) for nan entries."""
    for child in sorted(nan_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.isdigit() or child.name in numeric_to_slug:
            slug = numeric_to_slug.get(child.name)
            if slug is None:
                yield child, None, "unmapped_numeric"
            else:
                yield child, slug, "numeric"
            continue
        # owner/… nesting produced by Path(slug_id)
        for repo_merge in sorted(child.iterdir()):
            if not repo_merge.is_dir():
                continue
            slug = f"{child.name}/{repo_merge.name}"
            yield repo_merge, slug, "slug_path"


def remap_model_tree(
    src_root: Path,
    dst_root: Path,
    numeric_to_slug: dict[str, str],
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    totals = {
        "scenarios": 0,
        "mapped": 0,
        "unmapped": 0,
        "agent_files": 0,
        "bypass_files": 0,
        "shared_files": 0,
        "base_files": 0,
    }
    scenarios = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if limit is not None:
        scenarios = scenarios[:limit]

    for src in scenarios:
        totals["scenarios"] += 1
        slug = numeric_to_slug.get(src.name)
        if slug is None:
            totals["unmapped"] += 1
            rows.append(
                {
                    "src": str(src),
                    "numeric_id": src.name,
                    "slug_id": "",
                    "status": "unmapped",
                }
            )
            continue
        dst = dst_root.joinpath(*Path(slug).parts)
        counts = remap_scenario(src, dst, dry_run=dry_run)
        totals["mapped"] += 1
        totals["agent_files"] += counts["agent"]
        totals["bypass_files"] += counts["bypass7"]
        totals["shared_files"] += counts["shared"]
        totals["base_files"] += counts["base"]
        rows.append(
            {
                "src": str(src),
                "numeric_id": src.name,
                "slug_id": slug,
                "dst": str(dst),
                "status": "ok",
                **{f"n_{k}": v for k, v in counts.items()},
            }
        )
    return rows, totals


def remap_nan(
    nan_root: Path,
    dst_root: Path,
    numeric_to_slug: dict[str, str],
    *,
    dry_run: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    totals = {
        "scenarios": 0,
        "mapped": 0,
        "unmapped": 0,
        "agent_files": 0,
        "bypass_files": 0,
        "shared_files": 0,
        "base_files": 0,
    }
    for src, slug, kind in iter_nan_scenarios(nan_root, numeric_to_slug):
        totals["scenarios"] += 1
        if slug is None:
            totals["unmapped"] += 1
            rows.append(
                {
                    "src": str(src),
                    "numeric_id": src.name,
                    "slug_id": "",
                    "status": f"unmapped_{kind}",
                }
            )
            continue
        dst = dst_root.joinpath(*Path(slug).parts)
        counts = remap_scenario(src, dst, dry_run=dry_run)
        totals["mapped"] += 1
        totals["agent_files"] += counts["agent"]
        totals["bypass_files"] += counts["bypass7"]
        totals["shared_files"] += counts["shared"]
        totals["base_files"] += counts["base"]
        rows.append(
            {
                "src": str(src),
                "numeric_id": src.name if kind == "numeric" else "",
                "slug_id": slug,
                "dst": str(dst),
                "status": f"ok_{kind}",
                **{f"n_{k}": v for k, v in counts.items()},
            }
        )
    return rows, totals


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            # Keep numeric IDs as strings so CSV doesn't float-format them.
            out = dict(row)
            if "numeric_id" in out and out["numeric_id"] is not None:
                out["numeric_id"] = str(out["numeric_id"])
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama", "qwen", "nano", "nan"],
        help="Which targets to remap (llama/qwen/nano/nan).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit scenarios per model (debug).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Parent directory for new_* folders.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    numeric_to_slug = load_numeric_to_slug(BENCH)
    print(f"Loaded {len(numeric_to_slug)} numeric->slug mappings from {BENCH}")

    selected = [m.lower() for m in args.models]
    all_rows: list[dict] = []

    for key, src in MODEL_SOURCES.items():
        if key not in selected:
            continue
        dst = args.output_root / f"new_{key}"
        print(f"\n=== {key}: {src} -> {dst} ===")
        if not src.is_dir():
            print(f"  SKIP missing source {src}")
            continue
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
        rows, totals = remap_model_tree(
            src, dst, numeric_to_slug, dry_run=args.dry_run, limit=args.limit
        )
        manifest = dst / "_remap_manifest.csv"
        if not args.dry_run:
            write_manifest(manifest, rows)
        print(f"  totals: {totals}")
        print(f"  manifest: {manifest}")
        all_rows.extend({"model": key, **r} for r in rows)

    if "nan" in selected:
        dst = args.output_root / "new_nan"
        print(f"\n=== nan: {NAN_ROOT} -> {dst} ===")
        if NAN_ROOT.is_dir():
            if not args.dry_run:
                dst.mkdir(parents=True, exist_ok=True)
            rows, totals = remap_nan(
                NAN_ROOT, dst, numeric_to_slug, dry_run=args.dry_run
            )
            manifest = dst / "_remap_manifest.csv"
            if not args.dry_run:
                write_manifest(manifest, rows)
            print(f"  totals: {totals}")
            print(f"  manifest: {manifest}")
            all_rows.extend({"model": "nan", **r} for r in rows)
        else:
            print(f"  SKIP missing source {NAN_ROOT}")

    summary_path = args.output_root / "new_remap_summary.csv"
    if not args.dry_run:
        write_manifest(summary_path, all_rows)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
