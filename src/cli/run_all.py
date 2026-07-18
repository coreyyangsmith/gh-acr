from __future__ import annotations

import asyncio
import datetime
import time
import traceback
from pathlib import Path
from typing import Literal

# Ensure one-time global startup (env, logging, tracing) before anything else
import src.startup  # noqa: F401

import pandas as pd
import tyro

from src.dataset.loader import load_benchmark
from src.config.settings import BATCH_SIZE
from src.config.eval_methods import EvalMethod, ALL_EVAL_METHODS
from src.utils.logger import setup_logger
from src.utils.run_ledger import RunLedger, capture_logs
from src.agents.graph_router import build_graph
from src.agents.observability import get_llm_calls
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
    start_index: int | None = None,
    end_index: int | None = None,
    sample_percent: int | None = None,
    sample_seed: int = 42,
):
    """Run the full benchmark across all evaluation methods.

    Parameters
    ----------
    max_scenarios
        Optional cap on total scenarios (ignored if difficulty sampling is used
        or if sample_percent is set).
    mode
        Processing mode: "clone" (defaults to "clone").
    methods
        Subset of evaluation methods to run. Defaults to all: ["base_a", "base_b", "agent", "bypass7", "force_mix"].
    model_name
        Optional model override for LLM-based methods.
    n_easy / n_medium / n_hard
        If provided, sample exactly this many scenarios per difficulty.
        Ignored when sample_percent is set.
    start_index
        Starting row index (0-based, inclusive). Use for batch processing.
    end_index
        Ending row index (0-based, exclusive). Use for batch processing.
        If None, processes to end of dataset.
    sample_percent
        Optional integer percent of the (possibly sliced) dataset to randomly
        sample (e.g. 2 = 2%). Takes precedence over difficulty sampling and
        max_scenarios. Applied after start_index/end_index.
    sample_seed
        Random seed for sample_percent (default: 42).
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
            start_index=start_index,
            end_index=end_index,
            sample_percent=sample_percent,
            sample_seed=sample_seed,
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
    start_index: int | None,
    end_index: int | None,
    sample_percent: int | None,
    sample_seed: int,
):
    # Configure root logger so all modules propagate here
    logger = setup_logger()
    methods_to_run: list[EvalMethod] = methods or ALL_EVAL_METHODS

    # Log run configuration for debugging
    logger.info("=" * 70)
    logger.info("PIPELINE RUN CONFIGURATION")
    logger.info("=" * 70)
    logger.info("  max_scenarios: %s", max_scenarios)
    logger.info("  mode: %s", mode)
    logger.info("  methods: %s", methods_to_run)
    logger.info("  model_name: %s", model_name)
    logger.info("  results_filename: %s", results_filename)
    logger.info("  n_easy: %s, n_medium: %s, n_hard: %s", n_easy, n_medium, n_hard)
    logger.info("  start_index: %s, end_index: %s", start_index, end_index)
    logger.info("  sample_percent: %s, sample_seed: %s", sample_percent, sample_seed)
    logger.info("=" * 70)

    # Load and optionally sample benchmark scenarios
    logger.info("Loading benchmark dataset…")
    benchmark_df = load_benchmark()
    
    # Apply start/end index slicing first (for batch processing)
    if start_index is not None or end_index is not None:
        start_idx = start_index if start_index is not None else 0
        end_idx = end_index if end_index is not None else len(benchmark_df)
        logger.info("Batch processing: slicing dataset from index %d to %d", start_idx, end_idx)
        benchmark_df = benchmark_df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Percent sampling takes precedence over difficulty / max_scenarios filters
    if sample_percent is not None:
        if any(v is not None for v in (n_easy, n_medium, n_hard, max_scenarios)):
            logger.warning(
                "sample_percent=%s is set; ignoring n_easy/n_medium/n_hard and max_scenarios",
                sample_percent,
            )
        if sample_percent < 0 or sample_percent > 100:
            raise ValueError(f"sample_percent must be in [0, 100], got {sample_percent}")
        frac = sample_percent / 100.0
        if frac <= 0.0 or benchmark_df.empty:
            benchmark_df = benchmark_df.iloc[0:0]
        else:
            benchmark_df = benchmark_df.sample(frac=frac, random_state=sample_seed).reset_index(drop=True)
        logger.info(
            "Sampled %d%% -> %d scenarios (seed=%d)",
            sample_percent,
            len(benchmark_df),
            sample_seed,
        )
    elif any(v is not None for v in (n_easy, n_medium, n_hard)):
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

    logger.info("Loaded %d scenarios to process", len(benchmark_df))
    if not benchmark_df.empty:
        logger.info("First scenario ID: %s", benchmark_df.iloc[0].get("id", benchmark_df.index[0]))
        logger.info("Columns: %s", list(benchmark_df.columns))

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

    # Success/failure ledger next to the results CSV (crash-resilient JSONL)
    ledger_path = results_path.with_name(f"{results_path.stem}_run_log.jsonl")
    if ledger_path.exists():
        logger.info("Removing existing run ledger: %s", ledger_path)
        ledger_path.unlink()
    ledger = RunLedger(ledger_path)
    logger.info("Run ledger: %s", ledger_path)

    # Process in batches, streaming results to CSV as they complete
    total = len(benchmark_df)
    for start in range(0, total, BATCH_SIZE):
        batch_df = benchmark_df.iloc[start : start + BATCH_SIZE]
        try:
            for method in methods_to_run:
                logger.info("=== Running method: %s (mode=%s) batch %s-%s ===", method, mode, start + 1, min(start + BATCH_SIZE, total))
                app = build_graph(process_mode=mode, eval_method=method)

                async def process_row(row):
                    scenario_key = row.get("id")
                    if scenario_key is None:
                        # Be robust to unnamed first column exported as index
                        scenario_key = str(row.name)
                    repo_slug = str(row.get("name", "") or "")
                    write_prep = method == methods_to_run[0]
                    row_start = time.perf_counter()
                    with capture_logs() as captured:
                        try:
                            per_file_results = await run_and_save_report(
                                app,
                                scenario_key,
                                output_root,
                                eval_method=method,
                                model_name=model_name,
                                process_mode=mode,
                                write_prep=write_prep,
                            )
                            elapsed = time.perf_counter() - row_start
                            # Derive lightweight status from returned rows (exclude prep)
                            data_rows = [
                                r for r in (per_file_results or [])
                                if r.get("eval_method") != "prep"
                            ]
                            exact_vals = [
                                bool(r.get("exact_match"))
                                for r in data_rows
                                if r.get("exact_match") != ""
                            ]
                            exact_overall = all(exact_vals) if exact_vals else None
                            df_index = data_rows[0].get("id") if data_rows else None
                            processing_time = (
                                data_rows[0].get("processing_time_s", round(elapsed, 3))
                                if data_rows
                                else round(elapsed, 3)
                            )
                            llm_calls = get_llm_calls()
                            prompt_total = sum(int(c.get("prompt_tokens") or 0) for c in llm_calls)
                            completion_total = sum(
                                int(c.get("completion_tokens") or 0) for c in llm_calls
                            )
                            logger.info(
                                "[run_all] scenario=%s method=%s llm_calls=%d "
                                "prompt_tokens=%d completion_tokens=%d",
                                scenario_key,
                                method,
                                len(llm_calls),
                                prompt_total,
                                completion_total,
                            )
                            ledger.record_success(
                                scenario_id=scenario_key,
                                df_index=df_index,
                                repo=repo_slug or (data_rows[0].get("repo") if data_rows else None),
                                eval_method=method,
                                model_name=model_name,
                                num_files=len(data_rows),
                                exact_match_overall=exact_overall,
                                processing_time_s=processing_time,
                                llm_calls=llm_calls,
                            )
                            return per_file_results
                        except Exception as exc:  # pragma: no cover – runtime resilience
                            elapsed = time.perf_counter() - row_start
                            tb = traceback.format_exc()
                            logger.exception(
                                "[run_all] Error processing scenario %s (%s)",
                                scenario_key,
                                method,
                            )
                            llm_calls = get_llm_calls()
                            ledger.record_failure(
                                scenario_id=scenario_key,
                                repo=repo_slug or None,
                                eval_method=method,
                                model_name=model_name,
                                error=exc,
                                traceback_text=tb,
                                captured_logs=list(captured),
                                processing_time_s=round(elapsed, 3),
                                failure_trace_path=getattr(exc, "failure_trace_path", None),
                                llm_calls=llm_calls,
                            )
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
                try:
                    # Prefer the same root used for cloning
                    from src.merge_pipeline.pipeline_clone import _checkout_root, _robust_rmtree  # type: ignore
                    repos_root = _checkout_root()
                except Exception:
                    repos_root = Path.cwd() / "repos"
                for _, row in batch_df.iterrows():
                    name = str(row.get("name", "")).replace("/", "___")
                    if not name:
                        continue
                    repo_dir = repos_root / name
                    if repo_dir.exists():
                        try:
                            _robust_rmtree(repo_dir)
                            logger.info("Cleaned cloned repo: %s", repo_dir)
                        except Exception:
                            logger.exception("Failed to remove repo directory: %s", repo_dir)

    ledger.record_summary(
        results_path=str(results_path),
        total_scenarios=total,
        methods=list(methods_to_run),
        model_name=model_name,
    )
    logger.info(
        "All evaluations complete. Consolidated results saved to %s | run ledger: %s (success=%d, failure=%d)",
        results_path,
        ledger_path,
        ledger.success_count,
        ledger.failure_count,
    )


if __name__ == "__main__":
    tyro.cli(main)

