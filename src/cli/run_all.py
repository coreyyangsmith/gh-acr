from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
import shutil
from typing import Literal

# Ensure one-time global startup (env, logging, tracing) before anything else
import src.startup  # noqa: F401

import pandas as pd
import tyro
from tqdm.asyncio import tqdm

from src.dataset.loader import load_benchmark
from src.config.settings import BATCH_SIZE
from src.config.eval_methods import EvalMethod, ALL_EVAL_METHODS
from src.utils.logger import setup_logger
from src.agents.graph_router import build_graph
from src.cli.runner import run_and_save_report, RESULTS_SCHEMA_COLUMNS


ProcessMode = Literal["clone"]


def main(
    max_scenarios: int | None = None,
    mode: ProcessMode = "clone",
    methods: list[EvalMethod] | None = None,
    model_name: str | None = None,
    results_filename: str | None = None,
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
        Processing mode: "clone" (defaults to "clone").
    methods
        Subset of evaluation methods to run. Defaults to all: ["base_a", "base_b", "agent", "multi", "bypass_multi"].
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
            results_filename=results_filename,
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
    results_filename: str | None,
    n_easy: int | None,
    n_medium: int | None,
    n_hard: int | None,
):
    # Configure root logger so all modules propagate here
    logger = setup_logger()
    methods_to_run: list[EvalMethod] = methods or ALL_EVAL_METHODS

    # Load and optionally sample benchmark scenarios
    logger.info("Loading benchmark dataset…")
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

    # Nest outputs under data/<model>/<id>
    output_root = Path.cwd() / "data"

    # Aggregate results into a single CSV per run (allow override)
    if results_filename:
        rp = Path(results_filename)
        results_path = rp if rp.is_absolute() else (Path.cwd() / "data" / rp.name)
    else:
        date_str = datetime.date.today().strftime("%Y_%m_%d")
        results_path = Path.cwd() / "data" / f"{date_str}_results_all.csv"
    if results_path.exists():
        logger.info("Removing existing results file: %s", results_path)
        results_path.unlink()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Process in batches, streaming results to CSV as they complete
    total = len(benchmark_df)
    for start in range(0, total, BATCH_SIZE):
        batch_df = benchmark_df.iloc[start : start + BATCH_SIZE]
        try:
            for method in methods_to_run:
                logger.info("=== Running method: %s (mode=%s) batch %s-%s ===", method, mode, start + 1, min(start + BATCH_SIZE, total))
                app = build_graph(process_mode=mode, eval_method=method)

                async def process_row(row):
                    try:
                        scenario_key = row.get("id")
                        if scenario_key is None:
                            # Be robust to unnamed first column exported as index
                            scenario_key = str(row.name)
                        # Only write prep for the first method to avoid duplicates
                        write_prep = method == methods_to_run[0]
                        return await run_and_save_report(app, scenario_key, output_root, eval_method=method, model_name=model_name, process_mode=mode, write_prep=write_prep)
                    except Exception as exc:  # pragma: no cover – runtime resilience
                        logger.exception("[run_all] Error processing scenario %s (%s)", row.get("id", row.name), method)
                        return []

                # Process scenarios sequentially (no concurrency)
                completed = 0
                tasks_count = len(batch_df)
                for _, row in batch_df.iterrows():
                    per_file_results = await process_row(row)
                    if per_file_results:
                        df = pd.DataFrame(per_file_results)
                        # Enforce unified column order/schema
                        df = df.reindex(columns=RESULTS_SCHEMA_COLUMNS)
                        header = not results_path.exists() or results_path.stat().st_size == 0
                        df.to_csv(results_path, mode="a", header=header, index=False)
                        completed += 1
                        logger.info("Appended %s rows for scenario (%s/%s) → %s", len(per_file_results), completed, tasks_count, results_path)
                if completed == 0:
                    logger.warning("Method %s: no results to append in this batch.", method)
        finally:
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
                            logger.info("Cleaned cloned repo: %s", repo_dir)
                        except Exception:
                            logger.exception("Failed to remove repo directory: %s", repo_dir)

    logger.info("All evaluations complete. Consolidated results saved to %s", results_path)


if __name__ == "__main__":
    tyro.cli(main)

