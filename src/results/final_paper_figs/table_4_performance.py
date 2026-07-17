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
        MODEL_SHORT,
        MODEL_TABLE_IV_NAMES,
        OUTPUT_DIR,
        PERF,
        common_ids,
        instance_agg,
        load_results,
        logger,
        sig_stars,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.final_paper_figs.shared import (
        MODEL_ORDER,
        MODEL_SHORT,
        MODEL_TABLE_IV_NAMES,
        OUTPUT_DIR,
        PERF,
        common_ids,
        instance_agg,
        load_results,
        logger,
        sig_stars,
    )

try:
    from src.results.rq1.config import DEFAULT_CONFIG as RQ1_DEFAULT_CONFIG
    from src.results.rq1.data import compute_paired_delta_statistics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.results.rq1.config import DEFAULT_CONFIG as RQ1_DEFAULT_CONFIG
    from src.results.rq1.data import compute_paired_delta_statistics


def _full_model_name_from_short(short: str) -> str | None:
    """Map table short model label to results CSV ``model_name``."""
    for full, s in MODEL_SHORT.items():
        if s == short:
            return full
    return None


def _format_delta_with_ci(mean_delta: float, lo: float, hi: float, p_value: float) -> str:
    """Format multi-agent minus single-agent delta with 95% paired bootstrap CI and stars."""
    stars = sig_stars(float(p_value))
    sign = "+" if mean_delta > 0 else ""
    if not (lo == lo and hi == hi):  # NaN CI
        return f"{sign}{mean_delta:.3f}{stars}"
    return f"{sign}{mean_delta:.3f} [{lo:.3f}, {hi:.3f}]{stars}"


def generate_table_4(results_csv: Path | None = None,
                      output_dir: Path | None = None) -> pd.DataFrame:
    """Generate TABLE IV: performance by model and setup."""
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("TABLE IV: Performance metrics")

    df = load_results(results_csv)
    common = common_ids(df)
    inst = instance_agg(df, common)

    df_common = df[df["id"].astype(str).isin(common)].copy()
    paired_stats = compute_paired_delta_statistics(
        df_common, RQ1_DEFAULT_CONFIG, granularity="instance"
    )
    paired_map = {(ps.model_name, ps.metric): ps for ps in paired_stats}

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
        full_name = _full_model_name_from_short(model)
        for m in PERF:
            if m in bypass_sub.columns:
                bypass_row[m] = round(bypass_sub[m].mean(), 2)
        for m in PERF:
            if m in agent_sub.columns and m in bypass_sub.columns:
                delta = round(bypass_sub[m].mean() - agent_sub[m].mean(), 2)
                sign = "+" if delta > 0 else ""
                ps = paired_map.get((full_name, m)) if full_name else None
                if ps is not None and ps.n_pairs > 0:
                    bypass_row[f"delta_{m}"] = _format_delta_with_ci(
                        ps.mean_delta, ps.ci_low, ps.ci_high, ps.p_value
                    )
                else:
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
    print("Delta columns: multi-agent minus single-agent; 95% paired bootstrap CI; "
          "significance: * p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon for soft metrics; "
          "exact binomial on discordant pairs for EM).")
    print("=" * 110)
    print(table.to_string(index=False))
    print("=" * 110 + "\n")

    return table


if __name__ == "__main__":
    generate_table_4()
