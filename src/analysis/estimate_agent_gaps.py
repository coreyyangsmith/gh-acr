"""Estimate cost to fill missing agent runs across the 3 models.

Uses ``data/agent_coverage_gaps/*_agent_missing_or_failed.csv`` as the full
gap lists (not_processed, empty_output, missing_results) and each model's
2% pilot ``agent`` averages (by difficulty) for token/time/cost.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.estimate_eval_budget import (
    _lookup_avg,
    average_tables,
    format_hours,
    scenario_level,
)
from src.config.model_costs import estimate_usd_cost, get_model_config

META = {
    "llama-3.1-8b": {
        "pilot": Path("data/2026_07_18_openrouter_llama31_8b_2pct_bj_ablations.csv"),
        "model": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "gaps": Path("data/agent_coverage_gaps/llama-3.1-8b_agent_missing_or_failed.csv"),
        "subset": Path("data/agent_coverage_gaps/subsets/llama-3.1-8b_agent_gaps.csv"),
    },
    "qwen3-32b": {
        "pilot": Path("data/2026_07_18_openrouter_qwen3_32b_2pct_bj_ablations.csv"),
        "model": "openrouter/qwen/qwen3-32b",
        "gaps": Path("data/agent_coverage_gaps/qwen3-32b_agent_missing_or_failed.csv"),
        "subset": Path("data/agent_coverage_gaps/subsets/qwen3-32b_agent_gaps.csv"),
    },
    "gpt-5-nano": {
        "pilot": Path("data/2026_07_18_openrouter_gpt5nano_2pct_bj_ablations.csv"),
        "model": "openrouter/openai/gpt-5-nano",
        "gaps": Path("data/agent_coverage_gaps/gpt-5-nano_agent_missing_or_failed.csv"),
        "subset": Path("data/agent_coverage_gaps/subsets/gpt-5-nano_agent_gaps.csv"),
    },
}


def _difficulty_counts(gaps: pd.DataFrame, full: pd.DataFrame) -> pd.Series:
    if "difficulty" in gaps.columns:
        return gaps["difficulty"].value_counts()
    id_col = "id" if "id" in gaps.columns else gaps.columns[0]
    ids = set(gaps[id_col].astype(str))
    return full.loc[full["id"].astype(str).isin(ids), "difficulty"].value_counts()


def main() -> None:
    full = pd.read_csv("data/git_good_bench_merge_commits_all.csv")
    full["id"] = full["id"].astype(str)

    print("Gap source: data/agent_coverage_gaps/*_agent_missing_or_failed.csv")
    print(
        "Rates: each model's OpenRouter MODEL_COSTS; "
        "token/time avgs from that model's 2% pilot agent rows (by difficulty)."
    )

    grand_cost = 0.0
    grand_cpu = 0.0
    rows_out: list[dict] = []

    for label, meta in META.items():
        gaps = pd.read_csv(meta["gaps"])
        subset_n = (
            len(pd.read_csv(meta["subset"])) if meta["subset"].exists() else None
        )
        miss_counts = _difficulty_counts(gaps, full)

        pilot = pd.read_csv(meta["pilot"])
        scen = scenario_level(pilot)
        agent = scen[scen["eval_method"] == "agent"].copy()
        by_md, _, by_m = average_tables(agent)

        model = meta["model"]
        cfg = get_model_config(model)
        tokens_in = tokens_out = cpu = recorded = 0.0

        print("\n" + "=" * 72)
        print(f"MODEL: {model}")
        print(f"Gaps file: {meta['gaps'].name} ({len(gaps)} scenarios)")
        if subset_n is not None:
            print(f"Runnable subset: {meta['subset'].name} ({subset_n} rows)")
        print(f"Missing by difficulty: {miss_counts.to_dict()}")
        if "category" in gaps.columns:
            print(f"Categories: {gaps['category'].value_counts().to_dict()}")
        print(
            f"Agent pilot avgs: n={len(agent)} "
            f"in={agent['tokens_in'].mean():,.0f} "
            f"out={agent['tokens_out'].mean():,.0f} "
            f"${agent['total_cost'].mean():.6f} "
            f"{agent['processing_time_s'].mean():.1f}s"
        )
        print(
            f"Rates: ${cfg.get('input_cost_per_1k', 0):.6f}/1k in, "
            f"${cfg.get('output_cost_per_1k', 0):.6f}/1k out"
        )

        print("Per-difficulty gap estimate:")
        for diff, n in miss_counts.items():
            n_i = int(n)
            ain = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_tokens_in")
            aout = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_tokens_out")
            atime = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_time_s")
            acost = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_cost_usd")
            tokens_in += ain * n_i
            tokens_out += aout * n_i
            cpu += atime * n_i
            recorded += acost * n_i
            _cin, _cout, cpart = estimate_usd_cost(model, ain * n_i, aout * n_i)
            print(
                f"  {diff}: n={n_i} x "
                f"(in={ain:,.0f}, out={aout:,.0f}, ${acost:.6f}, {atime:.1f}s) "
                f"-> ${cpart:.4f}"
            )
            rows_out.append(
                {
                    "model_label": label,
                    "model": model,
                    "difficulty": diff,
                    "n_missing": n_i,
                    "avg_tokens_in": ain,
                    "avg_tokens_out": aout,
                    "avg_cost_usd": acost,
                    "avg_time_s": atime,
                    "est_cost_usd": cpart,
                    "est_cpu_seconds": atime * n_i,
                }
            )

        cin, cout, ctot = estimate_usd_cost(model, tokens_in, tokens_out)
        print(
            f"GAP TOTAL: tokens_in={tokens_in:,.0f} tokens_out={tokens_out:,.0f} "
            f"| priced=${ctot:.4f} (in ${cin:.4f} + out ${cout:.4f}) "
            f"| recorded-extrap=${recorded:.4f}"
        )
        print(f"GAP CPU-time: {format_hours(cpu)} ({cpu:,.0f}s)")
        grand_cost += ctot
        grand_cpu += cpu

    print("\n" + "=" * 72)
    print(f"ALL 3 MODELS agent all-gaps cost: ${grand_cost:.4f}")
    print(f"ALL 3 MODELS agent all-gaps CPU-time: {format_hours(grand_cpu)}")

    out = Path("data/budget_estimates/agent_gap_costs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
