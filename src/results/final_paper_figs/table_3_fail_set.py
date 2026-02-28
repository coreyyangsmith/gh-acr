"""TABLE III: Fail-set overview by model.

Shows labeled MCR counts, difficulty breakdown (with row-wise percentages),
and total MCR for each model's fail set.

Labeled MCR comes from per-model paired_data.csv in rq3_fail_only/.
Total MCR counts file-level bypass7 entries where exact_match is False.

Usage::

    python -m src.results.final_paper_figs.table_3_fail_set
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from src.results.final_paper_figs.shared import (
        DIFF_ORDER,
        MODEL_FULL_TO_TABLE3,
        OUTPUT_DIR,
        RESULTS_CSV,
        coerce_em,
        logger,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.final_paper_figs.shared import (
        DIFF_ORDER,
        MODEL_FULL_TO_TABLE3,
        OUTPUT_DIR,
        RESULTS_CSV,
        coerce_em,
        logger,
    )

# Per-model fail-only paired_data files (relative to repo root)
_MODEL_FAIL_PAIRED = {
    "GPT-5 Nano": (
        Path("results/rq3_fail_only/2025-11-09-gpt5nano-failure/paired_data.csv"),
        "openai/gpt-5-nano",
    ),
    "Qwen3": (
        Path("results/rq3_fail_only/2025-11-23-qwen3-failure/paired_data.csv"),
        "groq:qwen/qwen3-32b",
    ),
    "Llama": (
        Path("results/rq3_fail_only/2026-02-03-llama-fail/paired_data.csv"),
        "local:meta-llama/Llama-3.1-8B-Instruct",
    ),
}


def generate_table_3(
    results_csv: Path | None = None,
    output_dir: Path | None = None,
    **_kwargs,
) -> pd.DataFrame:
    """Generate TABLE III: fail-set overview by model."""
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("TABLE III: Fail-set overview")

    # ── Load results for Total MCR ────────────────────────────────────
    results_csv = results_csv or RESULTS_CSV
    results = pd.read_csv(results_csv)
    results["id"] = results["id"].astype(str)
    results["exact_match"] = coerce_em(results["exact_match"])

    bypass = results[results["eval_method"] == "bypass7"].copy()

    model_order = ["GPT-5 Nano", "Qwen3", "Llama"]
    rows = []

    for model_display in model_order:
        paired_path, model_full = _MODEL_FAIL_PAIRED[model_display]

        # ── Labeled MCR from per-model paired_data ────────────────────
        paired = pd.read_csv(paired_path)
        paired["difficulty"] = paired["difficulty"].str.lower().str.strip()
        labeled_count = len(paired)

        # Difficulty breakdown
        diff_counts = {}
        for d in DIFF_ORDER:
            diff_counts[d] = int((paired["difficulty"] == d).sum())

        # ── Total MCR: file-level bypass7 rows with EM=False ──────────
        model_bypass = bypass[bypass["model_name"] == model_full]
        total_mcr = int((model_bypass["exact_match"] == 0).sum())

        row = {"Model": model_display, "Labeled MCR": labeled_count}
        for d in DIFF_ORDER:
            cnt = diff_counts[d]
            pct = cnt / labeled_count * 100 if labeled_count > 0 else 0
            row[d.capitalize()] = f"{cnt} ({pct:.1f}%)"
        row["Total MCR"] = f"{total_mcr:,}"
        rows.append(row)

    # ── Overall row ───────────────────────────────────────────────────
    total_labeled = sum(r["Labeled MCR"] for r in rows)
    overall_diff = {d: 0 for d in DIFF_ORDER}
    for model_display in model_order:
        paired = pd.read_csv(_MODEL_FAIL_PAIRED[model_display][0])
        paired["difficulty"] = paired["difficulty"].str.lower().str.strip()
        for d in DIFF_ORDER:
            overall_diff[d] += int((paired["difficulty"] == d).sum())

    total_mcr_all = int((bypass["exact_match"] == 0).sum())

    overall = {"Model": "Overall", "Labeled MCR": total_labeled}
    for d in DIFF_ORDER:
        cnt = overall_diff[d]
        pct = cnt / total_labeled * 100 if total_labeled > 0 else 0
        overall[d.capitalize()] = f"{cnt} ({pct:.1f}%)"
    overall["Total MCR"] = f"{total_mcr_all:,}"
    rows.append(overall)

    table = pd.DataFrame(rows)

    # Save
    csv_path = out / "Table_III_fail_set_overview.csv"
    table.to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path.name}")

    # Print
    print("\n" + "=" * 90)
    print("TABLE III: Fail-set overview by model")
    print("=" * 90)
    print(table.to_string(index=False))
    print("=" * 90 + "\n")

    return table


if __name__ == "__main__":
    generate_table_3()
