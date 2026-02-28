"""TABLE IV: Performance metrics by model and setup (Agent vs MACRO).

Shows EM, Similarity, BLEU-3, and ROUGE-L for single-agent (Agent) and
multi-agent (MACRO/Bypass) setups, plus delta columns.

Usage::

    python -m src.results.final_paper_figs.table_4_performance
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from src.results.final_paper_figs.shared import (
        MODEL_ORDER,
        MODEL_TABLE_IV_NAMES,
        OUTPUT_DIR,
        PERF,
        common_ids,
        instance_agg,
        load_results,
        logger,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.final_paper_figs.shared import (
        MODEL_ORDER,
        MODEL_TABLE_IV_NAMES,
        OUTPUT_DIR,
        PERF,
        common_ids,
        instance_agg,
        load_results,
        logger,
    )


def generate_table_4(results_csv: Path | None = None,
                      output_dir: Path | None = None) -> pd.DataFrame:
    """Generate TABLE IV: performance by model and setup."""
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("TABLE IV: Performance metrics")

    df = load_results(results_csv)
    common = common_ids(df)
    inst = instance_agg(df, common)

    n_instances = len(common)
    logger.info(f"  Common instances: {n_instances}")

    rows = []
    for model in MODEL_ORDER:
        model_display = MODEL_TABLE_IV_NAMES.get(model, model)

        agent_sub = inst[(inst["model"] == model) & (inst["method"] == "Agent")]
        bypass_sub = inst[(inst["model"] == model) & (inst["method"] == "Bypass")]

        if agent_sub.empty or bypass_sub.empty:
            continue

        # Single-agent row
        agent_row = {"Model": model_display, "Setup": "Single-agent"}
        for m in PERF:
            if m in agent_sub.columns:
                agent_row[m] = round(agent_sub[m].mean(), 2)
        for m in PERF:
            agent_row[f"delta_{m}"] = "--"
        rows.append(agent_row)

        # Multi-agent row
        bypass_row = {"Model": model_display, "Setup": "Multi-agent"}
        for m in PERF:
            if m in bypass_sub.columns:
                bypass_row[m] = round(bypass_sub[m].mean(), 2)
        for m in PERF:
            if m in agent_sub.columns and m in bypass_sub.columns:
                delta = round(bypass_sub[m].mean() - agent_sub[m].mean(), 2)
                sign = "+" if delta > 0 else ""
                bypass_row[f"delta_{m}"] = f"{sign}{delta:.2f}"
        rows.append(bypass_row)

    table = pd.DataFrame(rows)

    # Rename columns for display
    col_rename = {
        "exact_match": "EM",
        "similarity": "Sim.",
        "bleu3": "BLEU-3",
        "rouge_l": "ROUGE-L",
        "delta_exact_match": "Delta_EM",
        "delta_similarity": "Delta_Sim.",
        "delta_bleu3": "Delta_BLEU-3",
        "delta_rouge_l": "Delta_ROUGE-L",
    }
    table = table.rename(columns=col_rename)

    # Save
    csv_path = out / "Table_IV_performance_metrics.csv"
    table.to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path.name}")

    # Print
    print("\n" + "=" * 110)
    print(f"TABLE IV: Performance by model and setup (N={n_instances} instances per row)")
    print("=" * 110)
    print(table.to_string(index=False))
    print("=" * 110 + "\n")

    return table


if __name__ == "__main__":
    generate_table_4()
