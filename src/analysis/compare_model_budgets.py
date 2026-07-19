"""Fresh per-model difficulty stats and full-run cost estimates from pilot CSVs.

Each results CSV is analyzed independently (own token/cost averages by
easy/medium/hard), then extrapolated to a target dataset.

Example
-------
uv run python -m src.analysis.compare_model_budgets \\
  --results-csvs \\
    data/2026_07_18_openrouter_llama31_8b_2pct_bj_ablations.csv \\
    data/2026_07_18_openrouter_qwen3_32b_2pct_bj_ablations.csv \\
    data/2026_07_18_openrouter_gpt5nano_2pct_bj_ablations.csv \\
  --target-csv data/git_good_bench_merge_commits_all.csv \\
  --output-dir data/budget_estimates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import tyro

from src.analysis.estimate_eval_budget import (
    DEFAULT_METHODS,
    average_tables,
    difficulty_counts,
    extrapolate_tokens_time,
    scenario_level,
)
from src.config.model_costs import estimate_usd_cost, get_model_config

DEFAULT_RESULTS = (
    Path("data/2026_07_18_openrouter_llama31_8b_2pct_bj_ablations.csv"),
    Path("data/2026_07_18_openrouter_qwen3_32b_2pct_bj_ablations.csv"),
    Path("data/2026_07_18_openrouter_gpt5nano_2pct_bj_ablations.csv"),
)


@dataclass
class Flags:
    results_csvs: tuple[Path, ...] = DEFAULT_RESULTS
    target_csv: Path = Path("data/git_good_bench_merge_commits_all.csv")
    methods: tuple[str, ...] = DEFAULT_METHODS
    output_dir: Path | None = field(default_factory=lambda: Path("data/budget_estimates"))


def _infer_model_id(df: pd.DataFrame, path: Path) -> str:
    names = [x for x in df.get("model_name", pd.Series(dtype=str)).dropna().unique().tolist() if x and str(x) != "NA"]
    if names:
        # Prefer the most common non-NA model name
        vc = df.loc[df["model_name"].isin(names), "model_name"].value_counts()
        return str(vc.index[0])
    # Fallback from filename hints
    name = path.name.lower()
    if "llama" in name:
        return "openrouter/meta-llama/llama-3.1-8b-instruct"
    if "qwen" in name:
        return "openrouter/qwen/qwen3-32b"
    if "gpt5" in name or "gpt-5" in name:
        return "openrouter/openai/gpt-5-nano"
    return "unknown"


def _price_per_difficulty_case(
    *,
    methods: tuple[str, ...],
    by_md: pd.DataFrame,
    by_m: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    """One scenario of a given difficulty, summed across all methods."""
    from src.analysis.estimate_eval_budget import _lookup_avg

    rows = []
    for difficulty in sorted(set(by_md["difficulty"].tolist()) | {"easy", "medium", "hard"}):
        tin = tout = recorded = 0.0
        for method in methods:
            tin += _lookup_avg(by_md, by_m, method, difficulty, "avg_tokens_in")
            tout += _lookup_avg(by_md, by_m, method, difficulty, "avg_tokens_out")
            recorded += _lookup_avg(by_md, by_m, method, difficulty, "avg_cost_usd")
        cin, cout, ctot = estimate_usd_cost(model_id, tin, tout)
        rows.append(
            {
                "difficulty": difficulty,
                "tokens_in_per_case": tin,
                "tokens_out_per_case": tout,
                "recorded_cost_per_case_usd": recorded,
                "priced_cost_in_usd": cin,
                "priced_cost_out_usd": cout,
                "priced_cost_per_case_usd": ctot,
            }
        )
    return pd.DataFrame(rows)


def analyze_one(path: Path, methods: tuple[str, ...], counts: pd.Series) -> dict:
    raw = pd.read_csv(path)
    model_id = _infer_model_id(raw, path)
    scen = scenario_level(raw)
    present = [m for m in methods if m in set(scen["eval_method"])]
    scen_f = scen[scen["eval_method"].isin(present)].copy()
    by_md, by_d, by_m = average_tables(scen_f)
    llm = scen_f[scen_f["tokens_in"] + scen_f["tokens_out"] > 0]
    _, by_d_llm, _ = average_tables(llm) if not llm.empty else (by_md, by_d, by_m)

    price_cases = _price_per_difficulty_case(
        methods=methods, by_md=by_md, by_m=by_m, model_id=model_id
    )

    extrap = extrapolate_tokens_time(
        methods=methods, counts=counts, by_md=by_md, by_m=by_m
    )
    total_in = float(extrap["tokens_in"].sum())
    total_out = float(extrap["tokens_out"].sum())
    # recorded-cost extrapolation (uses each model's own recorded avg costs)
    recorded_total = 0.0
    recorded_by_diff: dict[str, float] = {}
    for diff, n in counts.items():
        n_i = int(n)
        case = price_cases.loc[price_cases["difficulty"] == diff]
        unit = float(case["recorded_cost_per_case_usd"].iloc[0]) if not case.empty else 0.0
        recorded_by_diff[str(diff)] = unit * n_i
        recorded_total += unit * n_i

    cin, cout, ctot = estimate_usd_cost(model_id, total_in, total_out)
    priced_by_diff = {}
    for diff, n in counts.items():
        case = price_cases.loc[price_cases["difficulty"] == diff]
        if case.empty:
            priced_by_diff[str(diff)] = 0.0
        else:
            priced_by_diff[str(diff)] = float(case["priced_cost_per_case_usd"].iloc[0]) * int(n)

    cfg = get_model_config(model_id)
    return {
        "path": path,
        "model_id": model_id,
        "cfg": cfg,
        "n_rows": len(raw),
        "n_scenario_method": len(scen_f),
        "n_unique_scenarios": int(scen_f["id"].nunique()),
        "methods_present": present,
        "by_md": by_md,
        "by_d": by_d,
        "by_d_llm": by_d_llm,
        "by_m": by_m,
        "price_cases": price_cases,
        "extrap": extrap,
        "total_in": total_in,
        "total_out": total_out,
        "recorded_total": recorded_total,
        "recorded_by_diff": recorded_by_diff,
        "priced_total": ctot,
        "priced_in": cin,
        "priced_out": cout,
        "priced_by_diff": priced_by_diff,
        "pilot_recorded_cost": float(scen_f["total_cost"].sum()),
        "pilot_tokens_in": float(scen_f["tokens_in"].sum()),
        "pilot_tokens_out": float(scen_f["tokens_out"].sum()),
    }


def _fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df.copy()
    for c in cols:
        if c not in show.columns:
            continue
        if "cost" in c:
            show[c] = show[c].map(lambda x: f"{x:.6f}")
        elif "token" in c or "time" in c or c.startswith("avg_"):
            show[c] = show[c].map(lambda x: f"{x:,.2f}")
    return show.to_string(index=False)


def main(flags: Flags) -> None:
    counts = difficulty_counts(flags.target_csv)
    print(f"Target: {flags.target_csv}")
    print(f"Scenarios: {int(counts.sum())} | {counts.to_dict()}")
    print(f"Methods ({len(flags.methods)}): {list(flags.methods)}")

    summaries = []
    all_price_rows = []
    all_extrap_rows = []

    for path in flags.results_csvs:
        if not path.exists():
            print(f"\nWARNING: missing {path}")
            continue
        r = analyze_one(path, flags.methods, counts)
        summaries.append(r)

        print(f"\n{'=' * 72}")
        print(f"MODEL: {r['model_id']}")
        print(f"Source: {path.name}")
        print(
            f"Pilot: {r['n_rows']} file-rows, {r['n_scenario_method']} scenario x method, "
            f"{r['n_unique_scenarios']} unique scenarios"
        )
        print(
            f"Pilot observed: tokens_in={r['pilot_tokens_in']:,.0f} "
            f"tokens_out={r['pilot_tokens_out']:,.0f} "
            f"recorded_cost=${r['pilot_recorded_cost']:.4f}"
        )
        rates = r["cfg"] or {}
        print(
            f"Pricing: ${rates.get('input_cost_per_1k', 0):.6f}/1k in, "
            f"${rates.get('output_cost_per_1k', 0):.6f}/1k out"
        )

        print("\n-- Per-scenario averages by difficulty (all methods) --")
        print(
            _fmt_table(
                r["by_d"],
                ["avg_tokens_in", "avg_tokens_out", "avg_tokens_total", "avg_cost_usd", "avg_time_s"],
            )
        )
        print("\n-- Per-scenario averages by difficulty (LLM methods only) --")
        print(
            _fmt_table(
                r["by_d_llm"],
                ["avg_tokens_in", "avg_tokens_out", "avg_tokens_total", "avg_cost_usd", "avg_time_s"],
            )
        )
        print("\n-- Price per difficulty CASE (1 scenario x all 8 methods) --")
        pc = r["price_cases"].copy()
        pc = pc[pc["difficulty"].isin(["easy", "medium", "hard"])].sort_values("difficulty")
        print(
            _fmt_table(
                pc,
                [
                    "tokens_in_per_case",
                    "tokens_out_per_case",
                    "recorded_cost_per_case_usd",
                    "priced_cost_per_case_usd",
                ],
            )
        )

        print(f"\n-- Expected FULL run ({int(counts.sum())} scenarios x {len(flags.methods)} methods) --")
        print(f"Tokens: in={r['total_in']:,.0f}  out={r['total_out']:,.0f}")
        print(
            f"Expected cost (tokens x MODEL_COSTS): ${r['priced_total']:.4f} "
            f"(in ${r['priced_in']:.4f} + out ${r['priced_out']:.4f})"
        )
        print(f"Expected cost (recorded avg_cost extrapolation): ${r['recorded_total']:.4f}")
        print("By difficulty (priced):")
        for d in ["easy", "medium", "hard"]:
            n = int(counts.get(d, 0))
            unit = float(pc.loc[pc["difficulty"] == d, "priced_cost_per_case_usd"].iloc[0]) if d in set(pc["difficulty"]) else 0.0
            print(f"  {d}: n={n} x ${unit:.6f}/case = ${r['priced_by_diff'].get(d, 0.0):.4f}")

        for _, row in pc.iterrows():
            all_price_rows.append(
                {
                    "model": r["model_id"],
                    "source": path.name,
                    **row.to_dict(),
                }
            )
        all_extrap_rows.append(
            {
                "model": r["model_id"],
                "source": path.name,
                "n_target_scenarios": int(counts.sum()),
                "tokens_in": r["total_in"],
                "tokens_out": r["total_out"],
                "priced_cost_usd": r["priced_total"],
                "recorded_extrap_cost_usd": r["recorded_total"],
                "priced_easy_usd": r["priced_by_diff"].get("easy", 0.0),
                "priced_medium_usd": r["priced_by_diff"].get("medium", 0.0),
                "priced_hard_usd": r["priced_by_diff"].get("hard", 0.0),
                "pilot_recorded_cost_usd": r["pilot_recorded_cost"],
            }
        )

    if summaries:
        print(f"\n{'=' * 72}")
        print("SUMMARY: expected FULL-run cost by model")
        total_all = 0.0
        for r in summaries:
            print(
                f"  {r['model_id']}: ${r['priced_total']:.4f} "
                f"(recorded-extrap ${r['recorded_total']:.4f})"
            )
            total_all += r["priced_total"]
        print(f"  ALL 3 MODELS combined: ${total_all:.4f}")

    if flags.output_dir is not None:
        flags.output_dir.mkdir(parents=True, exist_ok=True)
        if all_price_rows:
            pd.DataFrame(all_price_rows).to_csv(
                flags.output_dir / "per_model_price_per_difficulty.csv", index=False
            )
        if all_extrap_rows:
            pd.DataFrame(all_extrap_rows).to_csv(
                flags.output_dir / "per_model_full_run_costs.csv", index=False
            )
        for r in summaries:
            stem = r["path"].stem
            r["by_d_llm"].to_csv(flags.output_dir / f"{stem}_avg_by_difficulty_llm.csv", index=False)
            r["by_md"].to_csv(flags.output_dir / f"{stem}_avg_by_method_difficulty.csv", index=False)
        print(f"\nWrote CSVs to {flags.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Flags))
