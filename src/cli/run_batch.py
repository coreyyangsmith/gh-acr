from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import datetime
from typing import Literal

# Ensure one-time global startup (env, logging, tracing) before anything else
import src.startup  # noqa: F401

import pandas as pd
import tyro
from tqdm.asyncio import tqdm

# Graph builder chosen at runtime
from src.cli.runner import run_and_save_report
from src.dataset.loader import load_benchmark
from src.agents.graph_router import build_graph
from src.config.settings import BATCH_SIZE
from src.utils.logger import setup_logger


def main(
    max_scenarios: int | None = None,
    mode: Literal["api", "clone"] = "api",
    eval_method: Literal["base_a", "base_b", "agent", "multi", "bypass_multi"] = "agent",
    model_name: str | None = None,
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
        One of: "base_a", "base_b", "agent", "multi" – selects merge-resolution strategy.
    """
    asyncio.run(
        _run(
            max_scenarios=max_scenarios,
            mode=mode,
            eval_method=eval_method,
            model_name=model_name,
            n_easy=n_easy,
            n_medium=n_medium,
            n_hard=n_hard,
        ),
    )


async def _run(max_scenarios: int | None, mode: str, eval_method: str, *, model_name: str | None, n_easy: int | None, n_medium: int | None, n_hard: int | None):
    """Internal async runner."""

    logger = setup_logger()
    app = build_graph(process_mode=mode, eval_method=eval_method)
    output_root = Path.cwd() / "data" / "output"

    # Build results filename based on current date and evaluation method, e.g. "2025_07_26_agent_results.csv"
    date_str = datetime.date.today().strftime("%Y_%m_%d")
    results_filename = f"{date_str}_{eval_method}_results.csv"
    results_path = Path.cwd() / "data" / results_filename

    # Clear previous results if the file exists
    if results_path.exists():
        logger.info("Removing existing results file: %s", results_path)
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
            results = await run_and_save_report(app, row["id"], output_root, eval_method=eval_method, model_name=model_name)
        except Exception as exc:
            logger.exception("[run_batch] Error processing scenario %s", row.get("id"))
            return []

        # Persist each file-level record to the shared CSV.
        df = pd.DataFrame(results)
        header = not results_path.exists() or results_path.stat().st_size == 0
        df.to_csv(results_path, mode="a", header=header, index=False)
        return results

    # -------------------------------------------------------------------
    # Process in batches; stream-append each scenario's results as they complete
    # -------------------------------------------------------------------
    total = len(benchmark_df)
    processed = 0

    for start in range(0, total, BATCH_SIZE):
        batch_df = benchmark_df.iloc[start : start + BATCH_SIZE]
        try:
            tasks = [process_and_append(row) for _, row in batch_df.iterrows()]

            completed = 0
            for fut in tqdm.as_completed(tasks):
                res = await fut
                completed += 1 if res else 0
                processed += 1 if res else 0
            logger.info("Processed %s/%s scenarios; results appended to %s", processed, total, results_path)
        finally:
            # Cleanup cloned repos for this batch (clone mode only)
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
                            logger.info("Cleaned cloned repo: %s", repo_dir)
                        except Exception:
                            logger.exception("Failed to remove repo directory: %s", repo_dir)

    logger.info("Batch evaluation complete. Metrics for %s scenarios appended to %s", total, results_path)


if __name__ == "__main__":
    tyro.cli(main)
