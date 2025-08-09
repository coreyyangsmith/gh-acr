from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
import shutil
from typing import Literal

import pandas as pd
import tyro
from tqdm.asyncio import tqdm

from src.dataset.loader import load_benchmark
from src.agents.graph_router import build_graph
from src.cli.runner import run_and_save_report


EvalMethod = Literal["base_a", "base_b", "agent", "multi"]
ProcessMode = Literal["api", "clone"]


def main(
    max_scenarios: int | None = None,
    mode: ProcessMode = "clone",
    methods: list[EvalMethod] | None = None,
    model_name: str | None = None,
    n_easy: int | None = None,
    n_medium: int | None = None,
    n_hard: int | None = None,
):
    """Run the full benchmark across all evaluation methods.

    Parameters
    ----------
    max_scenarios
        Optional cap on total scenarios (ignored if difficulty sampling is used).
    mode
        Processing mode: "api" or "clone" (defaults to "clone").
    methods
        Subset of evaluation methods to run. Defaults to all: ["base_a", "base_b", "agent", "multi"].
    model_name
        Optional model override for LLM-based methods.
    n_easy / n_medium / n_hard
        If provided, sample exactly this many scenarios per difficulty.
    """

    asyncio.run(
        _run_all(
            max_scenarios=max_scenarios,
            mode=mode,
            methods=methods,
            model_name=model_name,
            n_easy=n_easy,
            n_medium=n_medium,
            n_hard=n_hard,
        )
    )


async def _run_all(
    *,
    max_scenarios: int | None,
    mode: ProcessMode,
    methods: list[EvalMethod] | None,
    model_name: str | None,
    n_easy: int | None,
    n_medium: int | None,
    n_hard: int | None,
):
    methods_to_run: list[EvalMethod] = methods or ["base_a", "base_b", "agent", "multi"]

    # Load and optionally sample benchmark scenarios
    benchmark_df = load_benchmark()
    if any(v is not None for v in (n_easy, n_medium, n_hard)):
        subsets = []
        if n_easy is not None:
            easy_df = benchmark_df[benchmark_df.get("difficulty", "").eq("easy")]
            if not easy_df.empty:
                subsets.append(easy_df.sample(n=min(n_easy, len(easy_df)), random_state=42))
        if n_medium is not None:
            med_df = benchmark_df[benchmark_df.get("difficulty", "").eq("medium")]
            if not med_df.empty:
                subsets.append(med_df.sample(n=min(n_medium, len(med_df)), random_state=42))
        if n_hard is not None:
            hard_df = benchmark_df[benchmark_df.get("difficulty", "").eq("hard")]
            if not hard_df.empty:
                subsets.append(hard_df.sample(n=min(n_hard, len(hard_df)), random_state=42))
        if subsets:
            benchmark_df = pd.concat(subsets, ignore_index=True)
    elif max_scenarios is not None:
        benchmark_df = benchmark_df.head(max_scenarios)

    output_root = Path.cwd() / "data" / "output"

    # Aggregate results into a single CSV per run
    date_str = datetime.date.today().strftime("%Y_%m_%d")
    results_path = Path.cwd() / "data" / f"{date_str}_results_all.csv"
    if results_path.exists():
        results_path.unlink()

    # Process in batches of 10 scenarios
    BATCH_SIZE = 10
    total = len(benchmark_df)
    for start in range(0, total, BATCH_SIZE):
        batch_df = benchmark_df.iloc[start : start + BATCH_SIZE]

        for method in methods_to_run:
            print(f"\n=== Running method: {method} (mode={mode}) batch {start+1}-{min(start+BATCH_SIZE, total)} ===")
            app = build_graph(process_mode=mode, eval_method=method)

            async def process_row(row):
                try:
                    return await run_and_save_report(app, row["id"], output_root, eval_method=method, model_name=model_name)
                except Exception as exc:  # pragma: no cover – runtime resilience
                    print(f"[run_all] Error processing scenario {row['id']} ({method}): {exc}")
                    return []

            tasks = [process_row(row) for _, row in batch_df.iterrows()]
            per_file_results_lists = await tqdm.gather(*tasks)

            # Flatten and persist
            flat_records = [rec for sub in per_file_results_lists for rec in (sub or [])]
            if flat_records:
                df = pd.DataFrame(flat_records)
                header = not results_path.exists() or results_path.stat().st_size == 0
                df.to_csv(results_path, mode="a", header=header, index=False)
                print(f"Method {method}: appended {len(flat_records)} file-level rows to {results_path}")
            else:
                print(f"Method {method}: no results to append.")

        # Cleanup batch repos (clone mode only)
        if mode == "clone":
            repos_root = Path.cwd() / "repos"
            for _, row in batch_df.iterrows():
                name = str(row.get("name", "")).replace("/", "___")
                if not name:
                    continue
                repo_dir = repos_root / name
                if repo_dir.exists():
                    try:
                        shutil.rmtree(repo_dir, ignore_errors=True)
                    except Exception:
                        pass

    print(f"\nAll evaluations complete. Consolidated results saved to {results_path}")


if __name__ == "__main__":
    tyro.cli(main)

