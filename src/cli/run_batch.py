from __future__ import annotations

import asyncio
from pathlib import Path
import datetime
from typing import Literal

import pandas as pd
import tyro
from tqdm.asyncio import tqdm

# Graph builder chosen at runtime
from src.cli.runner import run_and_save_report
from src.dataset.loader import load_benchmark
from src.agents.graph_router import build_graph


def main(
    max_scenarios: int | None = None,
    mode: Literal["api", "clone"] = "api",
    eval_method: Literal["agent", "base"] = "agent",
    n_easy: int | None = None,
    n_medium: int | None = None,
    n_hard: int | None = None,
):
    """Run the merge-resolution pipeline for the entire dataset.

    Parameters
    ----------
    max_scenarios
        Limit the number of scenarios processed (useful for quick tests).
    mode
        "api" or "clone" – selects repository access strategy.
    eval_method
        "agent" or "base" – selects merge-resolution strategy.
    """
    asyncio.run(
        _run(
            max_scenarios=max_scenarios,
            mode=mode,
            eval_method=eval_method,
            n_easy=n_easy,
            n_medium=n_medium,
            n_hard=n_hard,
        ),
    )


async def _run(max_scenarios: int | None, mode: str, eval_method: str, *, n_easy: int | None, n_medium: int | None, n_hard: int | None):
    """Internal async runner."""

    app = build_graph(process_mode=mode, eval_method=eval_method)
    output_root = Path.cwd() / "data" / "output"

    # Build results filename based on current date and evaluation method, e.g. "2025_07_26_agent_results.csv"
    date_str = datetime.date.today().strftime("%Y_%m_%d")
    results_filename = f"{date_str}_{eval_method}_results.csv"
    results_path = Path.cwd() / "data" / results_filename

    # Clear previous results if the file exists
    if results_path.exists():
        results_path.unlink()

    benchmark_df = load_benchmark()

    # -------------------------------------------------------------------
    # Optional difficulty-based sampling
    # -------------------------------------------------------------------
    if any(v is not None for v in (n_easy, n_medium, n_hard)):
        subsets = []
        if n_easy is not None:
            easy_df = benchmark_df[benchmark_df.get("difficulty", "").eq("easy")]
            subsets.append(easy_df.sample(n=min(n_easy, len(easy_df)), random_state=42))
        if n_medium is not None:
            med_df = benchmark_df[benchmark_df.get("difficulty", "").eq("medium")]
            subsets.append(med_df.sample(n=min(n_medium, len(med_df)), random_state=42))
        if n_hard is not None:
            hard_df = benchmark_df[benchmark_df.get("difficulty", "").eq("hard")]
            subsets.append(hard_df.sample(n=min(n_hard, len(hard_df)), random_state=42))

        if subsets:
            benchmark_df = pd.concat(subsets, ignore_index=True)
    elif max_scenarios is not None:
        benchmark_df = benchmark_df.head(max_scenarios)

    async def process_and_append(row):
        # `run_and_save_report` now returns a **list** of per-file dictionaries.
        try:
            results = await run_and_save_report(app, row["id"], output_root, eval_method=eval_method)
        except Exception as exc:
            print(f"[run_batch] Error processing scenario {row['id']}: {exc}")
            return []

        # Persist each file-level record to the shared CSV.
        df = pd.DataFrame(results)
        header = not results_path.exists() or results_path.stat().st_size == 0
        df.to_csv(results_path, mode="a", header=header, index=False)
        return results

    tasks = [
        process_and_append(row)
        for _, row in benchmark_df.iterrows()
    ]
    
    await tqdm.gather(*tasks)

    print(f"\nBatch evaluation complete. Metrics for {len(tasks)} scenarios appended to {results_path}")


if __name__ == "__main__":
    tyro.cli(main)
