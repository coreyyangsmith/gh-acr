"""Estimate full-eval token, cost, and wall-clock budgets from a pilot run.

Uses observed per-scenario token/time averages from a results CSV (e.g. a 2%
pilot), stratified by difficulty and eval method, then extrapolates to a
target dataset (full bench and/or the 10% subset) for one or more models.

Example
-------
uv run python -m src.analysis.estimate_eval_budget \\
  --results-csv data/2026_07_18_openrouter_llama31_8b_2pct_bj_ablations.csv \\
  --methods agent base_a base_b better_judge bj_no_summary bj_no_judge bj_no_plan bj_no_review \\
  --models openrouter/meta-llama/llama-3.1-8b-instruct openrouter/qwen/qwen3-32b openrouter/openai/gpt-5-nano \\
  --scenario-concurrency 8 --method-concurrency 6 --batch-size 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import tyro

from src.config.model_costs import estimate_usd_cost, get_model_config
from src.config.settings import BATCH_SIZE, DATA_PATH

DEFAULT_METHODS = (
    "agent",
    "base_a",
    "base_b",
    "better_judge",
    "bj_no_summary",
    "bj_no_judge",
    "bj_no_plan",
    "bj_no_review",
)

DEFAULT_MODELS = (
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter/qwen/qwen3-32b",
    "openrouter/openai/gpt-5-nano",
)

DEFAULT_RESULTS = Path("data/2026_07_18_openrouter_llama31_8b_2pct_bj_ablations.csv")
DEFAULT_SUBSET = Path("data/git_good_bench_merge_commits_all_subset_10_seed42.csv")


@dataclass
class Flags:
    """CLI flags for budget estimation."""

    results_csv: Path = DEFAULT_RESULTS
    target_csvs: tuple[Path, ...] = field(
        default_factory=lambda: (Path(DATA_PATH), DEFAULT_SUBSET)
    )
    methods: tuple[str, ...] = DEFAULT_METHODS
    models: tuple[str, ...] = DEFAULT_MODELS
    scenario_concurrency: int = 8
    method_concurrency: int = 6
    batch_size: int = BATCH_SIZE
    output_dir: Path | None = None


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out


def scenario_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-file rows to one row per (id, eval_method).

    Tokens/costs are summed across conflict files. Processing time is stored
    once per scenario×method in the results CSV, so we take the first value.
    """
    work = _to_numeric(
        df,
        [
            "tokens_in",
            "tokens_out",
            "tokens_total",
            "cost_in",
            "cost_out",
            "total_cost",
            "processing_time_s",
        ],
    )
    if "difficulty" not in work.columns:
        work["difficulty"] = "unknown"
    agg_kwargs: dict = {
        "tokens_in": ("tokens_in", "sum"),
        "tokens_out": ("tokens_out", "sum"),
        "tokens_total": ("tokens_total", "sum"),
        "cost_in": ("cost_in", "sum"),
        "cost_out": ("cost_out", "sum"),
        "total_cost": ("total_cost", "sum"),
        "processing_time_s": ("processing_time_s", "first"),
        "n_files": ("file_name", "count") if "file_name" in work.columns else ("id", "size"),
    }
    grouped = (
        work.groupby(["id", "eval_method", "difficulty"], dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
    )
    return grouped


def average_tables(scen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (by method×difficulty, by difficulty, by method) average tables."""

    def _agg(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (
            frame.groupby(keys, dropna=False)
            .agg(
                n_scenarios=("id", "count"),
                avg_tokens_in=("tokens_in", "mean"),
                avg_tokens_out=("tokens_out", "mean"),
                avg_tokens_total=("tokens_total", "mean"),
                avg_cost_usd=("total_cost", "mean"),
                avg_time_s=("processing_time_s", "mean"),
            )
            .reset_index()
        )

    by_md = _agg(scen, ["eval_method", "difficulty"])
    by_d = _agg(scen, ["difficulty"])
    by_m = _agg(scen, ["eval_method"])
    by_d = by_d.copy()
    by_d.insert(0, "scope", "by_difficulty")
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "difficulty": "all",
                "n_scenarios": int(len(scen)),
                "avg_tokens_in": float(scen["tokens_in"].mean()) if len(scen) else 0.0,
                "avg_tokens_out": float(scen["tokens_out"].mean()) if len(scen) else 0.0,
                "avg_tokens_total": float(scen["tokens_total"].mean()) if len(scen) else 0.0,
                "avg_cost_usd": float(scen["total_cost"].mean()) if len(scen) else 0.0,
                "avg_time_s": float(scen["processing_time_s"].mean()) if len(scen) else 0.0,
            }
        ]
    )
    return by_md, pd.concat([by_d, overall], ignore_index=True), by_m


def difficulty_counts(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "difficulty" not in df.columns:
        raise ValueError(f"{path} has no difficulty column")
    return df["difficulty"].value_counts()


def _lookup_avg(
    by_md: pd.DataFrame,
    by_m: pd.DataFrame,
    method: str,
    difficulty: str,
    col: str,
) -> float:
    hit = by_md[(by_md["eval_method"] == method) & (by_md["difficulty"] == difficulty)]
    if not hit.empty:
        return float(hit.iloc[0][col])
    hit_m = by_m[by_m["eval_method"] == method]
    if not hit_m.empty:
        return float(hit_m.iloc[0][col])
    return 0.0


def extrapolate_tokens_time(
    *,
    methods: Sequence[str],
    counts: pd.Series,
    by_md: pd.DataFrame,
    by_m: pd.DataFrame,
) -> pd.DataFrame:
    """Difficulty-stratified extrapolation of tokens and CPU-seconds per method."""
    rows: list[dict] = []
    for method in methods:
        tin = tout = tsec = 0.0
        detail: dict[str, dict[str, float]] = {}
        for difficulty, n in counts.items():
            n_i = int(n)
            avg_in = _lookup_avg(by_md, by_m, method, str(difficulty), "avg_tokens_in")
            avg_out = _lookup_avg(by_md, by_m, method, str(difficulty), "avg_tokens_out")
            avg_t = _lookup_avg(by_md, by_m, method, str(difficulty), "avg_time_s")
            detail[str(difficulty)] = {
                "n": float(n_i),
                "avg_tokens_in": avg_in,
                "avg_tokens_out": avg_out,
                "avg_time_s": avg_t,
                "tokens_in": avg_in * n_i,
                "tokens_out": avg_out * n_i,
                "cpu_seconds": avg_t * n_i,
            }
            tin += avg_in * n_i
            tout += avg_out * n_i
            tsec += avg_t * n_i
        rows.append(
            {
                "eval_method": method,
                "n_scenarios": int(counts.sum()),
                "tokens_in": tin,
                "tokens_out": tout,
                "cpu_seconds": tsec,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def estimate_wall_clock_s(
    cpu_seconds: float,
    *,
    n_methods: int,
    scenario_concurrency: int,
    method_concurrency: int,
    batch_size: int,
) -> tuple[float, int]:
    """Approximate wall time from summed CPU-seconds and run_all parallelism.

    Within each batch, up to ``min(batch_size, scenario_concurrency)`` scenarios
    run, and each scenario runs up to ``min(n_methods, method_concurrency)``
    methods in parallel. Peak slots ≈ product of those two caps.
    """
    scen_slots = max(1, min(int(batch_size), int(scenario_concurrency)))
    method_slots = max(1, min(int(n_methods), int(method_concurrency)))
    parallel = scen_slots * method_slots
    return float(cpu_seconds) / parallel, parallel


def format_hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h < 1:
        return f"{seconds / 60.0:.1f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24.0:.1f} d ({h:.0f} h)"


def _print_df(
    title: str,
    df: pd.DataFrame,
    float_cols: Sequence[str] | None = None,
    cost_cols: Sequence[str] = ("avg_cost_usd",),
) -> None:
    print(f"\n=== {title} ===")
    show = df.copy()
    if float_cols:
        for c in float_cols:
            if c not in show.columns:
                continue
            if c in cost_cols:
                show[c] = show[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
            else:
                show[c] = show[c].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    print(show.to_string(index=False))


def main(flags: Flags) -> None:
    results_path = flags.results_csv
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    raw = pd.read_csv(results_path)
    scen = scenario_level(raw)
    # Restrict averages to requested methods when present
    present = [m for m in flags.methods if m in set(scen["eval_method"])]
    missing = [m for m in flags.methods if m not in set(scen["eval_method"])]
    if missing:
        print(f"WARNING: methods absent from results (will use 0 averages): {missing}")
    scen_f = scen[scen["eval_method"].isin(present)].copy()

    by_md, by_d_overall, by_m = average_tables(scen_f)

    print(f"Results source: {results_path}")
    print(
        f"Observed scenario x method rows: {len(scen_f)} "
        f"({scen_f['id'].nunique()} unique scenarios, methods={present})"
    )
    print(
        "Note: pilot may be incomplete; averages use completed scenario x method units only."
    )

    _print_df(
        "Per-scenario averages by method x difficulty",
        by_md.sort_values(["eval_method", "difficulty"]),
        ["avg_tokens_in", "avg_tokens_out", "avg_tokens_total", "avg_cost_usd", "avg_time_s"],
    )
    _print_df(
        "Per-scenario averages by difficulty (and overall)",
        by_d_overall,
        ["avg_tokens_in", "avg_tokens_out", "avg_tokens_total", "avg_cost_usd", "avg_time_s"],
    )
    _print_df(
        "Per-scenario averages by method (overall)",
        by_m.sort_values("eval_method"),
        ["avg_tokens_in", "avg_tokens_out", "avg_tokens_total", "avg_cost_usd", "avg_time_s"],
    )

    # LLM-only difficulty averages (exclude zero-token baselines)
    llm = scen_f[scen_f["tokens_in"] + scen_f["tokens_out"] > 0]
    if not llm.empty:
        _, llm_by_d, _ = average_tables(llm)
        _print_df(
            "Per-scenario averages by difficulty (LLM methods only)",
            llm_by_d,
            ["avg_tokens_in", "avg_tokens_out", "avg_tokens_total", "avg_cost_usd", "avg_time_s"],
        )

    out_rows: list[dict] = []
    for target in flags.target_csvs:
        if not target.exists():
            print(f"\nWARNING: skip missing target dataset {target}")
            continue
        counts = difficulty_counts(target)
        print(f"\n=== Target dataset: {target} ===")
        print(f"Scenarios: {int(counts.sum())} | by difficulty: {counts.to_dict()}")

        extrap = extrapolate_tokens_time(
            methods=flags.methods,
            counts=counts,
            by_md=by_md,
            by_m=by_m,
        )
        total_in = float(extrap["tokens_in"].sum())
        total_out = float(extrap["tokens_out"].sum())
        total_cpu = float(extrap["cpu_seconds"].sum())
        wall_s, parallel = estimate_wall_clock_s(
            total_cpu,
            n_methods=len(flags.methods),
            scenario_concurrency=flags.scenario_concurrency,
            method_concurrency=flags.method_concurrency,
            batch_size=flags.batch_size,
        )

        method_show = extrap.drop(columns=["detail"]).copy()
        method_show["tokens_in"] = method_show["tokens_in"].map(lambda x: f"{x:,.0f}")
        method_show["tokens_out"] = method_show["tokens_out"].map(lambda x: f"{x:,.0f}")
        method_show["cpu_seconds"] = method_show["cpu_seconds"].map(lambda x: f"{x:,.0f}")
        print("\nExtrapolated tokens / CPU-time by method:")
        print(method_show.to_string(index=False))
        print(
            f"\nTotals: tokens_in={total_in:,.0f}  tokens_out={total_out:,.0f}  "
            f"cpu_seconds={total_cpu:,.0f} ({format_hours(total_cpu)})"
        )
        print(
            f"Wall-clock estimate @ batch_size={flags.batch_size}, "
            f"scenario_concurrency={flags.scenario_concurrency}, "
            f"method_concurrency={flags.method_concurrency} "
            f"(~{parallel} parallel slots): {format_hours(wall_s)} "
            f"({wall_s:,.0f} s)"
        )
        print(
            "Time note: uses observed processing_time_s from the pilot model; "
            "other models may differ."
        )

        print("\nCost estimates by model (token counts x MODEL_COSTS rates):")
        for model in flags.models:
            cfg = get_model_config(model)
            if not cfg:
                print(f"  {model}: UNKNOWN pricing in MODEL_COSTS — skipped")
                continue
            cost_in, cost_out, total = estimate_usd_cost(model, total_in, total_out)
            print(
                f"  {model}\n"
                f"    rates: ${cfg['input_cost_per_1k']:.6f}/1k in, "
                f"${cfg['output_cost_per_1k']:.6f}/1k out\n"
                f"    cost_in=${cost_in:.4f}  cost_out=${cost_out:.4f}  "
                f"total=${total:.4f}"
            )
            # per-method cost breakdown
            for _, row in extrap.iterrows():
                cin, cout, ctot = estimate_usd_cost(
                    model, row["tokens_in"], row["tokens_out"]
                )
                out_rows.append(
                    {
                        "target_csv": str(target),
                        "n_scenarios": int(counts.sum()),
                        "model": model,
                        "eval_method": row["eval_method"],
                        "tokens_in": row["tokens_in"],
                        "tokens_out": row["tokens_out"],
                        "cost_in_usd": cin,
                        "cost_out_usd": cout,
                        "total_cost_usd": ctot,
                        "cpu_seconds": row["cpu_seconds"],
                        "wall_clock_s_all_methods": wall_s,
                        "parallel_slots": parallel,
                    }
                )
            # model total row
            out_rows.append(
                {
                    "target_csv": str(target),
                    "n_scenarios": int(counts.sum()),
                    "model": model,
                    "eval_method": "__ALL_METHODS__",
                    "tokens_in": total_in,
                    "tokens_out": total_out,
                    "cost_in_usd": cost_in,
                    "cost_out_usd": cost_out,
                    "total_cost_usd": total,
                    "cpu_seconds": total_cpu,
                    "wall_clock_s_all_methods": wall_s,
                    "parallel_slots": parallel,
                }
            )

    if flags.output_dir is not None:
        flags.output_dir.mkdir(parents=True, exist_ok=True)
        by_md.to_csv(flags.output_dir / "pilot_avg_method_difficulty.csv", index=False)
        by_d_overall.to_csv(flags.output_dir / "pilot_avg_difficulty.csv", index=False)
        by_m.to_csv(flags.output_dir / "pilot_avg_method.csv", index=False)
        if out_rows:
            pd.DataFrame(out_rows).to_csv(
                flags.output_dir / "extrapolated_budgets.csv", index=False
            )
        print(f"\nWrote CSV summaries to {flags.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Flags))
