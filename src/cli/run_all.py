from __future__ import annotations

import asyncio
import copy
import datetime
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

# Ensure one-time global startup (env, logging, tracing) before anything else
import src.startup  # noqa: F401

import pandas as pd
import tyro

from src.dataset.loader import load_benchmark
from src.config.settings import BATCH_SIZE
from src.config.eval_methods import EvalMethod, ALL_EVAL_METHODS
from src.cache.scenario_context import ensure_prepared
from src.utils.logger import setup_logger
from src.utils.degradation import (
    clear_degradations,
    get_degradations,
    has_degradations,
    primary_degradation_category,
)
from src.utils.failure_classify import classify_failure
from src.utils.run_ledger import RunLedger, capture_logs, load_successful_units
from src.utils.run_progress import (
    RunProgress,
    reset_current_worker_id,
    set_current_worker_id,
)
from src.agents.graph_router import build_graph
from src.agents.observability import get_llm_calls
from src.cli.runner import run_and_save_report, RESULTS_SCHEMA_COLUMNS


ProcessMode = Literal["clone"]

_DEFAULT_CONCURRENCY = 8
_DEFAULT_METHOD_CONCURRENCY_CAP = 8


class CreditsExhaustedAbort(RuntimeError):
    """Raised to stop a batch run after an insufficient-credits / 402 failure."""

    pass


def _resolve_concurrency(concurrency: int | None) -> int:
    """Resolve scenario worker-pool size from CLI arg or INFERENCE_CONCURRENCY env."""
    if concurrency is not None:
        return max(1, int(concurrency))
    env = (os.getenv("INFERENCE_CONCURRENCY") or "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return _DEFAULT_CONCURRENCY


def _resolve_method_concurrency(
    method_concurrency: int | None, n_methods: int
) -> int:
    """Max parallel methods per scenario (default: min(n_methods, 8))."""
    if method_concurrency is not None:
        return max(1, int(method_concurrency))
    return max(1, min(int(n_methods), _DEFAULT_METHOD_CONCURRENCY_CAP))


def _stamp_soft_degradation_flags(
    rows: list[dict[str, Any]],
    *,
    soft_degraded: bool,
    degradation_category: str,
    num_degradations: int,
) -> None:
    """Stamp soft-degradation columns onto result rows (in place)."""
    for row in rows:
        row["soft_degraded"] = soft_degraded
        row["degradation_category"] = degradation_category
        row["num_degradations"] = num_degradations


def _scenario_key_from_row(row) -> str:
    key = row.get("id")
    if key is None:
        return str(row.name)
    return str(key)


def _row_to_sample(row) -> dict[str, Any]:
    """Convert a benchmark DataFrame row into a sample_row dict for prep/cache."""
    sample = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    sample["df_index"] = getattr(row, "name", sample.get("id"))
    if "id" not in sample or sample.get("id") is None:
        sample["id"] = _scenario_key_from_row(row)
    sj = sample.get("scenario_json")
    if isinstance(sj, str):
        sample["scenario_json"] = json.loads(sj)
    elif sj is None and isinstance(sample.get("scenario"), str):
        from src.dataset.loader import _parse_scenario

        sample["scenario_json"] = _parse_scenario(sample["scenario"])
    return sample


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
    concurrency: int | None = None,
    method_concurrency: int | None = None,
    resume: bool = False,
    trace_replay: bool = False,
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
        Subset of evaluation methods to run. Defaults to all methods in
        ALL_EVAL_METHODS (baselines, agent, bypass7, better_judge, bj_* ablations,
        force_mix).
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
    concurrency
        Max concurrent **scenarios** within a batch (default: 8, or
        INFERENCE_CONCURRENCY). Peak LLM pressure ≈ concurrency ×
        method_concurrency; lower this when running many methods in parallel.
    method_concurrency
        Max parallel methods per scenario (default: min(len(methods), 8)).
    resume
        If True, keep existing results CSV / run ledger / failures log and skip
        ``(scenario_id, eval_method)`` units already recorded as success or
        soft-degraded (flagged CSV already written). Hard failures and
        never-attempted units are re-run.
    trace_replay
        If True, ``bj_*`` ablations reuse a canonical ``better_judge`` trace
        (same-run snapshot or existing on-disk artifacts) and only execute the
        stages the ablation changes. Schedules ``better_judge`` before ablations
        within each scenario. Provenance is written to artifact metadata, the
        run ledger, and results CSV columns.

    Notes
    -----
    Clones under ``GHACR_CLONE_DIR`` / ``./repos`` and prepared context under
    ``data/context_cache`` (or ``GHACR_CONTEXT_CACHE_DIR``) persist across
    batches and model runs. Prep runs once per scenario, then pending methods
    execute in parallel.
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
            concurrency=concurrency,
            method_concurrency=method_concurrency,
            resume=resume,
            trace_replay=trace_replay,
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
    concurrency: int | None = None,
    method_concurrency: int | None = None,
    resume: bool = False,
    trace_replay: bool = False,
):
    # Configure root logger so all modules propagate here
    logger = setup_logger()
    methods_to_run: list[EvalMethod] = methods or ALL_EVAL_METHODS
    workers = _resolve_concurrency(concurrency)
    method_workers = _resolve_method_concurrency(method_concurrency, len(methods_to_run))
    run_t0 = time.perf_counter()

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
    logger.info("  scenario_concurrency: %s", workers)
    logger.info("  method_concurrency: %s", method_workers)
    logger.info(
        "  peak_llm_slots≈%s (scenario_concurrency × method_concurrency)",
        workers * method_workers,
    )
    logger.info("  resume: %s", resume)
    logger.info("  trace_replay: %s", trace_replay)
    logger.info("  wall_clock_start: %.6f (perf_counter)", run_t0)
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

    n_scenarios = len(benchmark_df)
    n_methods = len(methods_to_run)
    total_units = n_scenarios * n_methods
    logger.info(
        "Total work units: %d scenarios × %d methods = %d",
        n_scenarios,
        n_methods,
        total_units,
    )

    # Nest outputs under data/<model>/<id>
    output_root = Path.cwd() / "data"

    # Aggregate results into a single CSV per run (allow override)
    if results_filename:
        rp = Path(results_filename)
        results_path = rp if rp.is_absolute() else (Path.cwd() / "data" / rp.name)
    else:
        date_str = datetime.date.today().strftime("%Y_%m_%d")
        results_path = Path.cwd() / "data" / f"{date_str}_results_all.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Success/failure/degraded ledger next to the results CSV (crash-resilient JSONL)
    ledger_path = results_path.with_name(f"{results_path.stem}_run_log.jsonl")
    failures_path = results_path.with_name(f"{results_path.stem}_failures.jsonl")

    done_units: set[tuple[str, str]] = set()
    if resume:
        if not results_path.exists() and not ledger_path.exists():
            logger.warning(
                "Resume requested but no existing results CSV or ledger at %s / %s; "
                "starting a fresh run.",
                results_path,
                ledger_path,
            )
        done_units = load_successful_units(ledger_path)
        # Only count successes that apply to this run's scenario×method grid
        scenario_ids = {
            str(row.get("id") if row.get("id") is not None else row.name)
            for _, row in benchmark_df.iterrows()
        }
        method_set = set(methods_to_run)
        done_in_scope = {
            (sid, m) for (sid, m) in done_units if sid in scenario_ids and m in method_set
        }
        remaining_units = max(0, total_units - len(done_in_scope))
        logger.info(
            "Resume: %d/%d units already successful; running %d remaining",
            len(done_in_scope),
            total_units,
            remaining_units,
        )
        done_units = done_in_scope
        ledger = RunLedger.from_existing(ledger_path, failures_path=failures_path)
        progress = RunProgress(remaining_units).activate()
    else:
        if results_path.exists():
            logger.info("Removing existing results file: %s", results_path)
            results_path.unlink()
        if ledger_path.exists():
            logger.info("Removing existing run ledger: %s", ledger_path)
            ledger_path.unlink()
        if failures_path.exists():
            logger.info("Removing existing failures log: %s", failures_path)
            failures_path.unlink()
        ledger = RunLedger(ledger_path, failures_path=failures_path)
        progress = RunProgress(total_units).activate()

    logger.info("Run ledger: %s", ledger_path)
    logger.info("Failures log: %s", failures_path)

    csv_lock = threading.Lock()
    credits_abort = threading.Event()

    def _append_results(per_file_results: list) -> None:
        """Append scenario rows to the results CSV under the shared lock."""
        if not per_file_results:
            return
        df = pd.DataFrame(per_file_results)
        # Enforce unified column order/schema
        df = df.reindex(columns=RESULTS_SCHEMA_COLUMNS)
        with csv_lock:
            if results_path.exists() and results_path.stat().st_size > 0:
                existing_cols = list(pd.read_csv(results_path, nrows=0).columns)
                if existing_cols != RESULTS_SCHEMA_COLUMNS:
                    # Schema evolved (e.g. soft-degradation flags); rewrite unified.
                    old = pd.read_csv(results_path).reindex(columns=RESULTS_SCHEMA_COLUMNS)
                    combined = pd.concat([old, df], ignore_index=True)
                    combined.to_csv(results_path, index=False)
                else:
                    df.to_csv(results_path, mode="a", header=False, index=False)
            else:
                df.to_csv(results_path, mode="w", header=True, index=False)
            logger.info(
                "Appended %s rows → %s | %s",
                len(per_file_results),
                results_path,
                progress.snapshot_line(),
            )

    # Build one graph per method (shared across scenarios in this process).
    method_apps = {
        method: build_graph(process_mode=mode, eval_method=method)
        for method in methods_to_run
    }

    try:
        total = len(benchmark_df)
        for start in range(0, total, BATCH_SIZE):
            batch_df = benchmark_df.iloc[start : start + BATCH_SIZE]
            logger.info(
                "=== Batch scenarios %s-%s | scenario_concurrency=%s "
                "method_concurrency=%s | %s ===",
                start + 1,
                min(start + BATCH_SIZE, total),
                workers,
                method_workers,
                progress.snapshot_line(),
            )

            async def process_method(
                *,
                scenario_key: str,
                repo_slug: str,
                method: str,
                prepared: dict[str, Any],
            ) -> list:
                if credits_abort.is_set():
                    return []
                row_start = time.perf_counter()
                worker_id = progress.acquire_worker()
                worker_token = set_current_worker_id(worker_id)
                progress.mark_started(worker_id, scenario_key, method)
                clear_degradations()
                with capture_logs() as captured:
                    try:
                        per_file_results = await run_and_save_report(
                            method_apps[method],
                            scenario_key,
                            output_root,
                            eval_method=method,
                            model_name=model_name,
                            process_mode=mode,
                            write_prep=False,
                            prepared_state=copy.deepcopy(prepared),
                            trace_replay=trace_replay,
                        )
                        elapsed = time.perf_counter() - row_start
                        data_rows = [
                            r
                            for r in (per_file_results or [])
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
                        prompt_total = sum(
                            int(c.get("prompt_tokens") or 0) for c in llm_calls
                        )
                        completion_total = sum(
                            int(c.get("completion_tokens") or 0) for c in llm_calls
                        )
                        # Replay provenance from first data row (scenario-level)
                        replay_meta = {
                            "trace_replay_enabled": bool(
                                data_rows[0].get("trace_replay_enabled")
                            )
                            if data_rows
                            else bool(trace_replay),
                            "trace_replay_strategy": (
                                data_rows[0].get("trace_replay_strategy") or ""
                            )
                            if data_rows
                            else "",
                            "trace_replay_fallback": (
                                data_rows[0].get("trace_replay_fallback") or ""
                            )
                            if data_rows
                            else "",
                        }
                        logger.info(
                            "[run_all] scenario=%s method=%s llm_calls=%d "
                            "prompt_tokens=%d completion_tokens=%d processing_time_s=%.3f "
                            "trace_replay=%s strategy=%s",
                            scenario_key,
                            method,
                            len(llm_calls),
                            prompt_total,
                            completion_total,
                            float(processing_time)
                            if processing_time is not None
                            else elapsed,
                            replay_meta["trace_replay_enabled"],
                            replay_meta["trace_replay_strategy"] or "-",
                        )
                        if has_degradations():
                            events = get_degradations()
                            primary = primary_degradation_category(events)
                            _stamp_soft_degradation_flags(
                                per_file_results or [],
                                soft_degraded=True,
                                degradation_category=primary or "other",
                                num_degradations=len(events),
                            )
                            logger.warning(
                                "[run_all] scenario=%s method=%s completed with "
                                "%d soft degradation(s); writing CSV rows",
                                scenario_key,
                                method,
                                len(events),
                            )
                            ledger.record_degraded(
                                scenario_id=scenario_key,
                                df_index=df_index,
                                repo=repo_slug
                                or (data_rows[0].get("repo") if data_rows else None),
                                eval_method=method,
                                model_name=model_name,
                                degradation_events=events,
                                failure_category=primary,
                                processing_time_s=processing_time,
                                llm_calls=llm_calls,
                                num_files=len(data_rows),
                                exact_match_overall=exact_overall,
                                captured_logs=list(captured),
                                **replay_meta,
                            )
                            progress.mark_done(worker_id, ok=True, elapsed_s=elapsed)
                            return per_file_results or []
                        _stamp_soft_degradation_flags(
                            per_file_results or [],
                            soft_degraded=False,
                            degradation_category="",
                            num_degradations=0,
                        )
                        ledger.record_success(
                            scenario_id=scenario_key,
                            df_index=df_index,
                            repo=repo_slug
                            or (data_rows[0].get("repo") if data_rows else None),
                            eval_method=method,
                            model_name=model_name,
                            num_files=len(data_rows),
                            exact_match_overall=exact_overall,
                            processing_time_s=processing_time,
                            llm_calls=llm_calls,
                            **replay_meta,
                        )
                        progress.mark_done(worker_id, ok=True, elapsed_s=elapsed)
                        return per_file_results or []
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
                            trace_replay_enabled=bool(trace_replay),
                        )
                        progress.mark_done(worker_id, ok=False, elapsed_s=elapsed)
                        if classify_failure(exc) == "credits":
                            credits_abort.set()
                            logger.error(
                                "[run_all] Insufficient API credits (402); aborting "
                                "remaining work. Add credits at "
                                "https://openrouter.ai/settings/credits then resume."
                            )
                            raise CreditsExhaustedAbort(str(exc)) from exc
                        return []
                    finally:
                        progress.release_worker(worker_id)
                        reset_current_worker_id(worker_token)

            async def process_scenario(row) -> list:
                """Prep once, then run pending methods (possibly in parallel)."""
                if credits_abort.is_set():
                    return []
                scenario_key = _scenario_key_from_row(row)
                repo_slug = str(row.get("name", "") or "")
                pending_methods = [
                    m
                    for m in methods_to_run
                    if (scenario_key, m) not in done_units
                ]
                if not pending_methods:
                    logger.info(
                        "Scenario %s: all methods already done; skipping",
                        scenario_key,
                    )
                    return []

                skipped = len(methods_to_run) - len(pending_methods)
                if skipped:
                    logger.info(
                        "Scenario %s: skipping %d already-successful method(s); "
                        "running %s",
                        scenario_key,
                        skipped,
                        pending_methods,
                    )

                sample = _row_to_sample(row)
                try:
                    # Blocking prep/clone off the event loop
                    prepared = await asyncio.to_thread(
                        ensure_prepared, scenario_key, sample
                    )
                except Exception as exc:
                    logger.exception(
                        "[run_all] Prep failed for scenario %s: %s",
                        scenario_key,
                        exc,
                    )
                    # Record failure for each pending method so resume can retry
                    for method in pending_methods:
                        worker_id = progress.acquire_worker()
                        worker_token = set_current_worker_id(worker_id)
                        progress.mark_started(worker_id, scenario_key, method)
                        try:
                            ledger.record_failure(
                                scenario_id=scenario_key,
                                repo=repo_slug or None,
                                eval_method=method,
                                model_name=model_name,
                                error=exc,
                                traceback_text=traceback.format_exc(),
                                captured_logs=[],
                                processing_time_s=0.0,
                                llm_calls=[],
                                prep=True,
                            )
                            progress.mark_done(worker_id, ok=False, elapsed_s=0.0)
                        finally:
                            progress.release_worker(worker_id)
                            reset_current_worker_id(worker_token)
                    return []

                out: list = []

                def _partition_methods_for_replay(
                    pending: list[str],
                ) -> tuple[list[str], list[str]]:
                    """When trace_replay is on, run better_judge before bj_* ablations."""
                    if not trace_replay:
                        return list(pending), []
                    from src.agents.trace_replay import BJ_ABLATION_METHODS

                    canonical = [m for m in pending if m == "better_judge"]
                    ablations = [m for m in pending if m in BJ_ABLATION_METHODS]
                    others = [
                        m
                        for m in pending
                        if m != "better_judge" and m not in BJ_ABLATION_METHODS
                    ]
                    # Phase 1: non-ablation methods including better_judge (canonical first)
                    phase1 = canonical + others
                    return phase1, ablations

                phase1_methods, ablation_methods = _partition_methods_for_replay(
                    pending_methods
                )

                async def _run_method_list(methods_list: list[str]) -> list:
                    local_out: list = []
                    if not methods_list:
                        return local_out
                    if method_workers <= 1 or len(methods_list) <= 1:
                        for method in methods_list:
                            if credits_abort.is_set():
                                break
                            rows = await process_method(
                                scenario_key=scenario_key,
                                repo_slug=repo_slug,
                                method=method,
                                prepared=prepared,
                            )
                            if rows:
                                local_out.extend(rows)
                        return local_out

                    def _run_method_sync(method: str):
                        if credits_abort.is_set():
                            return []
                        return asyncio.run(
                            process_method(
                                scenario_key=scenario_key,
                                repo_slug=repo_slug,
                                method=method,
                                prepared=prepared,
                            )
                        )

                    pool_size = min(method_workers, len(methods_list))
                    with ThreadPoolExecutor(max_workers=pool_size) as pool:
                        futures = [
                            pool.submit(_run_method_sync, method)
                            for method in methods_list
                        ]
                        for fut in as_completed(futures):
                            try:
                                rows = fut.result()
                            except CreditsExhaustedAbort:
                                for other in futures:
                                    other.cancel()
                                raise
                            if rows:
                                local_out.extend(rows)
                    return local_out

                if trace_replay and ablation_methods:
                    logger.info(
                        "Scenario %s: trace_replay scheduling phase1=%s then ablations=%s",
                        scenario_key,
                        phase1_methods,
                        ablation_methods,
                    )
                phase1_rows = await _run_method_list(phase1_methods)
                if phase1_rows:
                    out.extend(phase1_rows)
                ablation_rows = await _run_method_list(ablation_methods)
                if ablation_rows:
                    out.extend(ablation_rows)
                return out

            if credits_abort.is_set():
                logger.error(
                    "[run_all] Aborting remaining batches due to insufficient credits."
                )
                break

            pending_rows = [
                row
                for _, row in batch_df.iterrows()
                if any(
                    (_scenario_key_from_row(row), m) not in done_units
                    for m in methods_to_run
                )
            ]
            if not pending_rows:
                logger.info(
                    "Batch %s-%s: all scenario×method units already done; skipping",
                    start + 1,
                    min(start + BATCH_SIZE, total),
                )
                continue

            batch_appended = 0
            try:
                if workers <= 1:
                    for row in pending_rows:
                        if credits_abort.is_set():
                            break
                        per_file_results = await process_scenario(row)
                        if per_file_results:
                            _append_results(per_file_results)
                            batch_appended += 1
                else:
                    def _run_scenario_sync(row_data):
                        if credits_abort.is_set():
                            return []
                        return asyncio.run(process_scenario(row_data))

                    pool_size = min(workers, len(pending_rows))
                    with ThreadPoolExecutor(max_workers=pool_size) as pool:
                        futures = [
                            pool.submit(_run_scenario_sync, row) for row in pending_rows
                        ]
                        for fut in as_completed(futures):
                            try:
                                per_file_results = fut.result()
                            except CreditsExhaustedAbort:
                                for other in futures:
                                    other.cancel()
                                credits_abort.set()
                                raise
                            if per_file_results:
                                _append_results(per_file_results)
                                batch_appended += 1
            except CreditsExhaustedAbort:
                logger.error(
                    "[run_all] Aborting remaining batches due to insufficient credits."
                )
                break

            if batch_appended == 0:
                logger.warning(
                    "Batch %s-%s: no results to append.",
                    start + 1,
                    min(start + BATCH_SIZE, total),
                )
            # Clones persist across batches and model runs (no rmtree cleanup).

        ledger.record_summary(
            results_path=str(results_path),
            failures_path=str(failures_path),
            total_scenarios=total,
            methods=list(methods_to_run),
            model_name=model_name,
            trace_replay=trace_replay,
        )
        run_elapsed = time.perf_counter() - run_t0
        logger.info("%s", progress.summary_line())
        logger.info(
            "All evaluations complete in %.3fs. Consolidated results saved to %s | "
            "run ledger: %s | failures log: %s "
            "(success=%d, failure=%d, degraded=%d)",
            run_elapsed,
            results_path,
            ledger_path,
            failures_path,
            ledger.success_count,
            ledger.failure_count,
            ledger.degraded_count,
        )
    finally:
        progress.deactivate()


if __name__ == "__main__":
    tyro.cli(main)
