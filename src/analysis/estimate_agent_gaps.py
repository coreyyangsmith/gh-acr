"""Estimate cost to fill missing agent runs across the 3 model pilots."""

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

FILES = {
    "llama-3.1-8b": {
        "path": Path("data/2026_07_18_openrouter_llama31_8b_2pct_bj_ablations.csv"),
        "model": "openrouter/meta-llama/llama-3.1-8b-instruct",
    },
    "qwen3-32b": {
        "path": Path("data/2026_07_18_openrouter_qwen3_32b_2pct_bj_ablations.csv"),
        "model": "openrouter/qwen/qwen3-32b",
    },
    "gpt-5-nano": {
        "path": Path("data/2026_07_18_openrouter_gpt5nano_2pct_bj_ablations.csv"),
        "model": "openrouter/openai/gpt-5-nano",
    },
}


def main() -> None:
    full = pd.read_csv("data/git_good_bench_merge_commits_all.csv")
    sample = full.sample(frac=0.02, random_state=42).reset_index(drop=True)
    sample["id"] = sample["id"].astype(str)
    expected_ids = set(sample["id"])
    print(f"Expected 2% sample (seed=42): {len(expected_ids)} scenarios")
    print(f"  by difficulty: {sample['difficulty'].value_counts().to_dict()}")

    # Optional precomputed gap subsets
    gap_dir = Path("data/agent_coverage_gaps/subsets")
    if gap_dir.exists():
        print(f"\nFound gap subsets in {gap_dir}:")
        for p in sorted(gap_dir.glob("*.csv")):
            g = pd.read_csv(p)
            print(f"  {p.name}: {len(g)} rows")

    grand_cost = 0.0
    grand_cpu = 0.0
    print("\n" + "=" * 72)

    for label, meta in FILES.items():
        path = meta["path"]
        model = meta["model"]
        df = pd.read_csv(path)
        df["id"] = df["id"].astype(str)
        scen = scenario_level(df)

        any_ids = set(df["id"].unique())
        agent_scen = scen[scen["eval_method"] == "agent"].copy()
        agent_ids = set(agent_scen["id"].astype(str))

        missing_ids = expected_ids - agent_ids
        missing = sample[sample["id"].isin(missing_ids)].copy()
        miss_counts = missing["difficulty"].value_counts()

        # Agent averages from completed agent runs for this model
        by_md, _, by_m = average_tables(agent_scen) if len(agent_scen) else (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

        # Fallback: if no agent rows yet, use overall LLM averages — shouldn't happen
        tokens_in = tokens_out = cpu = recorded = 0.0
        detail = []
        for diff, n in miss_counts.items():
            n_i = int(n)
            if len(agent_scen):
                ain = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_tokens_in")
                aout = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_tokens_out")
                atime = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_time_s")
                acost = _lookup_avg(by_md, by_m, "agent", str(diff), "avg_cost_usd")
            else:
                ain = aout = atime = acost = 0.0
            tokens_in += ain * n_i
            tokens_out += aout * n_i
            cpu += atime * n_i
            recorded += acost * n_i
            cin, cout, ctot = estimate_usd_cost(model, ain * n_i, aout * n_i)
            detail.append(
                {
                    "difficulty": diff,
                    "n_missing": n_i,
                    "avg_in": ain,
                    "avg_out": aout,
                    "avg_time_s": atime,
                    "avg_cost": acost,
                    "est_cost": ctot,
                }
            )

        cin, cout, ctot = estimate_usd_cost(model, tokens_in, tokens_out)
        cfg = get_model_config(model)
        grand_cost += ctot
        grand_cpu += cpu

        print(f"\nMODEL: {model}")
        print(f"Source: {path.name}")
        print(
            f"Agent complete: {len(agent_ids)}/{len(expected_ids)} | "
            f"missing: {len(missing_ids)}"
        )
        print(f"Missing by difficulty: {miss_counts.to_dict()}")
        print(
            f"Agent pilot avgs (completed): n={len(agent_scen)} "
            f"avg_in={agent_scen['tokens_in'].mean():,.0f} "
            f"avg_out={agent_scen['tokens_out'].mean():,.0f} "
            f"avg_cost=${agent_scen['total_cost'].mean():.6f} "
            f"avg_time={agent_scen['processing_time_s'].mean():.1f}s"
            if len(agent_scen)
            else "No completed agent rows"
        )
        print(
            f"Rates: ${cfg.get('input_cost_per_1k', 0):.6f}/1k in, "
            f"${cfg.get('output_cost_per_1k', 0):.6f}/1k out"
        )
        if detail:
            print("Per-difficulty gap estimate:")
            for d in detail:
                print(
                    f"  {d['difficulty']}: n={d['n_missing']} "
                    f"x (in={d['avg_in']:,.0f}, out={d['avg_out']:,.0f}, "
                    f"${d['avg_cost']:.6f}, {d['avg_time_s']:.1f}s) "
                    f"-> ${d['est_cost']:.4f}"
                )
        print(
            f"GAP TOTAL: tokens_in={tokens_in:,.0f} tokens_out={tokens_out:,.0f} "
            f"| priced=${ctot:.4f} (in ${cin:.4f} + out ${cout:.4f}) "
            f"| recorded-extrap=${recorded:.4f}"
        )
        print(f"GAP CPU-time: {format_hours(cpu)} ({cpu:,.0f}s)")

    print("\n" + "=" * 72)
    print(f"ALL 3 MODELS agent-gap cost: ${grand_cost:.4f}")
    print(f"ALL 3 MODELS agent-gap CPU-time: {format_hours(grand_cpu)}")


if __name__ == "__main__":
    main()
