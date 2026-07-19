from __future__ import annotations

"""Shared logic for running the merge-resolution pipeline.

This module is the single source of truth for processing a scenario and
writing its output files. It is used by the `run_all` entrypoint.
"""
from pathlib import Path
import src.startup  # noqa: F401  # Ensure startup side-effects apply when runner is imported
from typing import Dict, Any
import os
import time
import traceback
from src.config.model_costs import estimate_usd_cost
from src.config.eval_methods import MULTI_AGENT_EVAL_METHODS
from src.utils.rate_limiter import LimiterRegistry
from src.utils.logger import setup_logger
from src.utils.run_progress import set_stage


def _log_pipeline_diagnostics(logger, context: str, state: Dict[str, Any]) -> None:
    """Log pipeline state diagnostics for debugging."""
    try:
        scenario_id = state.get("scenario_id", "unknown")
        status = state.get("status", "unknown")
        
        # Count files and content sizes
        diffs_a = state.get("diffs_a", {}) or {}
        diffs_b = state.get("diffs_b", {}) or {}
        resolved = state.get("resolved_contents", {}) or {}
        parent_a = state.get("parent_a_contents", {}) or {}
        parent_b = state.get("parent_b_contents", {}) or {}
        ancestor = state.get("ancestor_contents", {}) or {}
        resolution_history = state.get("resolution_history", {}) or {}
        
        total_diff_a_chars = sum(len(v) for v in diffs_a.values())
        total_diff_b_chars = sum(len(v) for v in diffs_b.values())
        total_resolved_chars = sum(len(v) for v in resolved.values())
        total_parent_a_chars = sum(len(v) for v in parent_a.values())
        total_parent_b_chars = sum(len(v) for v in parent_b.values())
        
        logger.debug(
            "[%s] scenario=%s, status=%s, files=%d, diff_a_chars=%d, diff_b_chars=%d, resolved_chars=%d",
            context, scenario_id, status, len(diffs_a), total_diff_a_chars, total_diff_b_chars, total_resolved_chars
        )
        
        # Log bypass-specific info
        bypass_decision = state.get("bypass_decision", "")
        bypass_method = state.get("bypass_method", "")
        if bypass_decision or bypass_method:
            logger.debug(
                "[%s] bypass_decision=%s, bypass_method=%s, resolution_history_files=%d",
                context, bypass_decision, bypass_method, len(resolution_history)
            )
        
        # Log parent contents availability (important for ALL_A/ALL_B bypass)
        logger.debug(
            "[%s] parent_a_files=%d (chars=%d), parent_b_files=%d (chars=%d), ancestor_files=%d",
            context, len(parent_a), total_parent_a_chars, len(parent_b), total_parent_b_chars, len(ancestor)
        )
        
        # Log per-file resolved content sizes (helps identify empty files)
        if resolved:
            for fpath, content in resolved.items():
                content_len = len(content) if content else 0
                if content_len == 0:
                    logger.warning("[%s] EMPTY resolved_contents for file: %s", context, fpath)
                else:
                    logger.debug("[%s] resolved_contents[%s] = %d chars", context, fpath, content_len)
        
        # Log memory usage
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.debug(
                "[%s] Memory: rss=%.1fMB, vms=%.1fMB",
                context, mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024
            )
        except Exception:
            pass
    except Exception as e:
        logger.warning("[%s] Diagnostics failed: %s", context, e)

# Unified results schema (column order) for all methods
RESULTS_SCHEMA_COLUMNS = [
    "id",
    "repo",
    "file_name",
    "exact_match",
    "similarity",
    "bleu3",
    "rouge_l",
    "eval_method",
    "bypass_method",
    "model_name",
    "tokens_system_prompt",
    "tokens_original",
    "tokens_diff_a",
    "tokens_diff_b",
    "tokens_output",
    "tokens_total",
    "tokens_in",
    "tokens_out",
    "cost_in",
    "cost_out",
    "total_cost",
    "processing_time_s",
    "difficulty",
    "project_size",
    "trace_replay_enabled",
    "trace_replay_strategy",
    "trace_replay_fallback",
]

async def run_and_save_report(
    app,
    scenario_id: str,
    output_root: Path,
    *,
    eval_method: str,
    model_name: str | None = None,
    process_mode: str | None = None,
    write_prep: bool = True,
    prepared_state: Dict[str, Any] | None = None,
    trace_replay: bool = False,
    trace_replay_source: str = "better_judge",
):
    """Run the pipeline for one scenario and save its report to disk.

    When ``prepared_state`` is provided (from the scenario context cache /
    ``ensure_prepared``), it is merged into the graph init state and the
    redundant prep clone is skipped. Shared conflict input files under
    ``data/<model>/<id>/`` are written only if missing.

    When ``trace_replay`` is True and ``eval_method`` is a ``bj_*`` ablation,
    the resolver hydrates reusable stages from a canonical
    ``better_judge`` snapshot (same-run or existing on disk).
    """

    logger = setup_logger(__name__)

    # Place outputs under data/<model_name>/<id>/<eval_method>/ — compute early so
    # agents can write per-call artifacts during invoke.
    if eval_method in ("base_a", "base_b", "base", "prep"):
        raw_model_dir = "nan"
    else:
        raw_model_dir = (model_name or os.getenv("OPENAI_MODEL", "")).strip() or "nan"
    safe_model_dir = (
        raw_model_dir.replace("/", "_").replace("\\", "_").replace(":", "_").strip() or "nan"
    )
    scenario_dir = output_root / safe_model_dir / str(scenario_id)
    artifact_root: Path | None = None
    if eval_method not in ("base", "base_a", "base_b", "prep"):
        artifact_root = scenario_dir / eval_method
        artifact_root.mkdir(parents=True, exist_ok=True)

    source_root = scenario_dir / (trace_replay_source or "better_judge")
    init_state: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "status": "start",
        "logs": [],
        "model_name": model_name,
        "eval_method": eval_method,
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "trace_replay": {
            "enabled": bool(trace_replay),
            "source_method": trace_replay_source or "better_judge",
            "source_root": str(source_root),
        },
    }
    if prepared_state:
        # Prefer caller-prepared context; keep run-specific keys above.
        merged = dict(prepared_state)
        merged.update(init_state)
        # Preserve sample_row / contents from prepared_state after update
        for key in (
            "sample_row",
            "repo_path",
            "ancestor_contents",
            "parent_a_contents",
            "parent_b_contents",
            "diffs_a",
            "diffs_b",
            "truth_contents",
            "diffs_truth",
            "commit_messages_a",
            "commit_messages_b",
            "status",
        ):
            if key in prepared_state:
                merged[key] = prepared_state[key]
        init_state = merged
        write_prep = False

    # ---------------------------------------------------------------------------
    # Optional pre-run preparation: clone repository (clone mode only)
    # This time is recorded separately as a 'prep' line item and excluded from
    # the main processing time measurement.
    # Skipped when prepared_state was supplied (prep owned by ensure_prepared).
    # ---------------------------------------------------------------------------
    prep_row: Dict[str, Any] | None = None
    if write_prep and (process_mode or "").strip().lower() == "clone":
        try:
            from src.dataset.loader import load_benchmark  # local import to avoid cyclic deps
            from src.merge_pipeline.pipeline_clone import _clone_repo, _checkout_root  # type: ignore

            # Locate scenario row similar to pipeline's load_sample_node
            df = load_benchmark()
            raw_id = str(scenario_id)
            row_series = None
            try:
                row_series = df.loc[int(raw_id)]
            except Exception:
                try:
                    row_series = df.loc[raw_id]
                except Exception:
                    row_series = None
            if row_series is None and "id" in df.columns:
                match = df.loc[df["id"].astype(str) == raw_id]
                if not match.empty:
                    row_series = match.iloc[0]
            if row_series is not None:
                sample = row_series.to_dict()
                prep_start = time.perf_counter()
                set_stage("clone", detail=str(sample.get("name", "")))
                _clone_repo(sample, checkout_dir=_checkout_root())
                prep_elapsed = time.perf_counter() - prep_start
                logger.info(
                    "Prep clone for scenario %s finished in %.1fs",
                    scenario_id,
                    prep_elapsed,
                )
                df_index = row_series.name
                repo_slug = str(sample.get("name", ""))
                difficulty = sample.get("difficulty", "unknown")
                # Create a separate line item recording clone time only
                prep_row = {
                    "id": df_index,
                    "repo": repo_slug,
                    "file_name": "",
                    "exact_match": "",
                    "similarity": "",
                    "bleu3": "",
                    "rouge_l": "",
                    "eval_method": "prep",
                    "bypass_method": "NA",
                    "model_name": "NA",
                    # Tokens/costs zeroed for prep
                    "tokens_system_prompt": 0,
                    "tokens_original": 0,
                    "tokens_diff_a": 0,
                    "tokens_diff_b": 0,
                    "tokens_output": 0,
                    "tokens_total": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_in": 0.0,
                    "cost_out": 0.0,
                    "total_cost": 0.0,
                    "processing_time_s": round(prep_elapsed, 3),
                    "difficulty": difficulty,
                    "project_size": sample.get("project_size", ""),
                    "trace_replay_enabled": False,
                    "trace_replay_strategy": "",
                    "trace_replay_fallback": "",
                }
        except Exception:
            # Best-effort: prep failures shouldn't block main processing
            prep_row = None

    start_ts = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Starting scenario %s with method=%s", scenario_id, eval_method)
    logger.info("=" * 60)
    
    # Log initial state diagnostics
    _log_pipeline_diagnostics(logger, "PRE-INVOKE", init_state)

    from src.agents.observability import (
        build_langfuse_invoke_config,
        clear_run_context,
        flush_langfuse,
        make_trace_name,
        scenario_observation,
        set_run_context,
        update_observation_cost_metadata,
    )

    set_run_context(
        eval_method=eval_method,
        scenario_id=str(scenario_id),
        model_name=model_name,
    )
    # Attach LangFuse CallbackHandler + langfuse_trace_name so the graph root
    # trace is named by eval method (passthrough when disabled).
    # Scenario id is carried via langfuse_session_id + tags, not the trace name.
    invoke_cfg = build_langfuse_invoke_config(
        {
            "configurable": {"thread_id": f"scn-{scenario_id}"},
            "run_name": make_trace_name(eval_method),
        },
        model_name=model_name,
    )
    try:
        with scenario_observation(make_trace_name(eval_method)) as scenario_obs:
            result = await app.ainvoke(init_state, config=invoke_cfg)
            # Attach scenario-level cost totals while the parent observation is open
            tok_map = result.get("token_counts", {}) or {}
            meta_in = meta_out = 0
            for v in tok_map.values():
                if not isinstance(v, dict):
                    continue
                for k, val in v.items():
                    try:
                        num = int(val)
                    except Exception:
                        continue
                    if k == "output":
                        meta_out += num
                    else:
                        meta_in += num
            c_in, c_out, c_tot = estimate_usd_cost(model_name or "", meta_in, meta_out)
            update_observation_cost_metadata(
                scenario_obs,
                tokens_in=meta_in,
                tokens_out=meta_out,
                cost_in=c_in,
                cost_out=c_out,
                total_cost=c_tot,
                model_name=model_name,
            )
            # Persist a versioned replay snapshot after a successful better_judge run
            # so same-run and future ablation replays can hydrate from it.
            if eval_method == "better_judge":
                try:
                    from src.agents.trace_replay import save_snapshot_from_state

                    save_snapshot_from_state(
                        result,
                        source_root=artifact_root or source_root,
                    )
                except Exception as snap_exc:  # pragma: no cover – best-effort
                    logger.warning(
                        "[%s] Failed to write replay snapshot for %s: %s",
                        eval_method,
                        scenario_id,
                        snap_exc,
                    )
    except Exception as e:
        elapsed_sec = time.perf_counter() - start_ts
        logger.error(
            "Pipeline failed for scenario=%s, method=%s after %.3fs: %s",
            scenario_id, eval_method, elapsed_sec, e
        )
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise
    finally:
        clear_run_context()
        flush_langfuse()

    elapsed_sec = time.perf_counter() - start_ts
    
    # Log final state diagnostics
    _log_pipeline_diagnostics(logger, "POST-INVOKE", result)

    # ---------------------------------------------------------------------------
    # Write full report to files
    # ---------------------------------------------------------------------------

    sample_row = result["sample_row"]
    df_index = sample_row["df_index"]
    # Keep the pre-invoke scenario_dir (keyed by scenario_id) so agent-written
    # artifacts stay co-located with shared inputs.
    files = sample_row["scenario_json"]["files_in_merge_conflict"]

    if eval_method != "base" and artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    # Log file processing summary
    logger.info(
        "[%s] Processing %d files for scenario %s: %s",
        eval_method, len(files), scenario_id, files
    )

    # Log resolved_contents keys for debugging mismatches
    resolved_keys = list(result.get("resolved_contents", {}).keys())
    if set(files) != set(resolved_keys):
        logger.warning(
            "[%s] File key mismatch! Expected files: %s, resolved_contents keys: %s",
            eval_method, files, resolved_keys
        )

    from src.agents.artifact_io import file_path_to_slug, write_final_artifacts

    for file_path in files:
        file_slug = file_path_to_slug(file_path)
        file_dir = scenario_dir / file_slug
        file_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("[%s] Processing file: %s → slug: %s", eval_method, file_path, file_slug)

        # Write shared conflict inputs once (parallel methods share scenario_dir).
        shared_marker = file_dir / "original.txt"
        if not shared_marker.exists():
            (file_dir / "original.txt").write_text(
                result["ancestor_contents"].get(file_path, ""), encoding="utf-8"
            )
            (file_dir / "a.txt").write_text(
                result["parent_a_contents"].get(file_path, ""), encoding="utf-8"
            )
            (file_dir / "b.txt").write_text(
                result["parent_b_contents"].get(file_path, ""), encoding="utf-8"
            )
            (file_dir / "a.diff").write_text(
                result["diffs_a"].get(file_path, ""), encoding="utf-8"
            )
            (file_dir / "b.diff").write_text(
                result["diffs_b"].get(file_path, ""), encoding="utf-8"
            )
            (file_dir / "ground_truth.txt").write_text(
                result["truth_contents"].get(file_path, ""), encoding="utf-8"
            )
            (file_dir / "ground_truth.diff").write_text(
                result.get("diffs_truth", {}).get(file_path, ""), encoding="utf-8"
            )

            try:
                cm_a = str(result.get("commit_messages_a", "")).strip()
                cm_b = str(result.get("commit_messages_b", "")).strip()
                if cm_a:
                    (file_dir / "a_commit_message.txt").write_text(cm_a, encoding="utf-8")
                if cm_b:
                    (file_dir / "b_commit_message.txt").write_text(cm_b, encoding="utf-8")
            except Exception as e:
                logger.warning(
                    "[%s] Failed to write commit messages for %s: %s",
                    eval_method,
                    file_path,
                    e,
                )

        # Ensure final/ exists for methods that produce resolved contents
        if eval_method not in ("base", "base_a", "base_b", "prep") and artifact_root is not None:
            merged_out = result.get("resolved_contents", {}).get(file_path, "")
            final_diff = (result.get("final_diffs", {}) or {}).get(file_path, "")
            final_dir = artifact_root / file_slug / "final"
            if not (final_dir / "resolved.txt").exists():
                write_final_artifacts(
                    artifact_root,
                    file_path=file_path,
                    resolved_text=merged_out or "",
                    final_diff=final_diff or "",
                )

    # Summary log for multi-agent methods
    if eval_method in MULTI_AGENT_EVAL_METHODS:
        bypass_decision = result.get("bypass_decision", "unknown")
        resolved_count = len(result.get("resolved_contents", {}))
        res_hist_count = len(result.get("resolution_history", {}))
        logger.info(
            "[%s] File writing complete for scenario %s: bypass_decision=%s, resolved_files=%d, resolution_history_files=%d, output_dir=%s",
            eval_method, scenario_id, bypass_decision, resolved_count, res_hist_count, scenario_dir / eval_method
        )

    eval_ = result.get("evaluation", {})

    method_tag = f"[method={eval_method}]"
    logger.info("%s Report for %s (%s)", method_tag, scenario_id, df_index)
    logger.info("%s     - Full report saved to: %s", method_tag, scenario_dir)
    logger.info("%s     - Overall exact match: %s", method_tag, eval_["overall_exact_match"])
    logger.info("%s     - Overall BLEU-3: %s", method_tag, eval_.get("overall_bleu3", "N/A"))
    logger.info("%s     - Overall ROUGE-L: %s", method_tag, eval_.get("overall_rouge_l", "N/A"))

    # -------------------------------------------------------------------
    # Token usage / cost estimation (if available)
    # -------------------------------------------------------------------

    token_map: Dict[str, Dict[str, int]] = result.get("token_counts", {})  # per-file token stats
    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")

    if token_map:
        # Accumulate across all files; include any numeric inputs from any agent stage
        total_in_tokens = 0
        total_out_tokens = 0
        for v in token_map.values():
            if not isinstance(v, dict):
                continue
            for k, val in v.items():
                try:
                    num = int(val)
                except Exception:
                    continue
                if k == "output":
                    total_out_tokens += num
                else:
                    total_in_tokens += num
    else:
        total_in_tokens = total_out_tokens = 0

    cost_in, cost_out, total_cost = estimate_usd_cost(
        model_name, total_in_tokens, total_out_tokens
    )

    if eval_method not in ("base_a", "base_b"):
        logger.info("%s     - Tokens in: %s  | cost: $%.4f", method_tag, total_in_tokens, cost_in)
        logger.info("%s     - Tokens out: %s | cost: $%.4f", method_tag, total_out_tokens, cost_out)
        logger.info("%s     - Estimated total LLM cost: $%.4f (model: %s)", method_tag, total_cost, model_name)

    logger.info("%s     - Processing time: %.2f s", method_tag, elapsed_sec)

    # -------------------------------------------------------------------
    # Rate limiter metrics (if any LLM calls were made)
    # -------------------------------------------------------------------
    rl_metrics = LimiterRegistry.metrics()
    if rl_metrics:
        logger.info("%s     - Rate limit activity:", method_tag)
        for key, m in rl_metrics.items():
            logger.info(
                "%s         · %s: waits=%s total_wait=%.2fs retries=%s last_delay=%.2fs",
                method_tag,
                key,
                m['wait_events'],
                m['total_wait_time_s'],
                m['total_retries'],
                m['last_retry_delay_s'],
            )

    # -------------------------------------------------------------------
    # Transform evaluation into **per-file** records so that downstream
    # aggregation is simpler and CSVs remain in a tidy, row-oriented form.
    # -------------------------------------------------------------------
    per_file_rows = []
    # Determine bypass method label for this scenario (A/B/MIX) or NA for others
    if eval_method in MULTI_AGENT_EVAL_METHODS:
        bypass_label = str(result.get("bypass_method") or result.get("bypass_decision", "MIX")).upper()
        # Normalize to short form if full form present
        if bypass_label in ("ALL_A", "A"):
            bypass_label = "A"
        elif bypass_label in ("ALL_B", "B"):
            bypass_label = "B"
        else:
            bypass_label = "MIX"
    else:
        bypass_label = "NA"
    repo_slug: str = sample_row["name"]  # e.g. "owner/repo"
    model_for_row = model_name if eval_method not in ("base_a", "base_b") else "NA"
    for file_path in sample_row["scenario_json"]["files_in_merge_conflict"]:
        exact_match_bool = bool(eval_["exact_match"].get(file_path, False))
        similarity_score = float(eval_["similarity"].get(file_path, 0.0))
        bleu3_score = float(eval_.get("bleu3", {}).get(file_path, 0.0))
        rouge_l_score = float(eval_.get("rouge_l", {}).get(file_path, 0.0))

        difficulty = sample_row.get("difficulty", "unknown")
        tok_stats = token_map.get(file_path, {}) if token_map else {}
        in_tok_file = tok_stats.get("system_prompt", 0) + tok_stats.get("original", 0) + tok_stats.get("diff_a", 0) + tok_stats.get("diff_b", 0)
        out_tok_file = tok_stats.get("output", 0)

        cost_in_file, cost_out_file, total_cost_file = estimate_usd_cost(
            model_name, in_tok_file, out_tok_file
        )

        per_file_rows.append({
            "id": df_index,
            "repo": repo_slug,
            "file_name": file_path,
            "exact_match": exact_match_bool,
            "similarity": similarity_score,
            "bleu3": bleu3_score,
            "rouge_l": rouge_l_score,
            "eval_method": eval_method,
            "bypass_method": bypass_label,
            "model_name": model_for_row,
            # Individual token categories
            "tokens_system_prompt": tok_stats.get("system_prompt", 0),
            "tokens_original": tok_stats.get("original", 0),
            "tokens_diff_a": tok_stats.get("diff_a", 0),
            "tokens_diff_b": tok_stats.get("diff_b", 0),
            "tokens_output": tok_stats.get("output", 0),

            # Aggregated counts
            "tokens_total": in_tok_file + out_tok_file,

            # Back-compat combined fields
            "tokens_in": in_tok_file,
            "tokens_out": out_tok_file,
            "cost_in": round(cost_in_file, 6),
            "cost_out": round(cost_out_file, 6),
            "total_cost": round(total_cost_file, 6),
            "processing_time_s": round(elapsed_sec, 3),
            "difficulty": difficulty,
            "project_size": sample_row.get("project_size", ""),
            "trace_replay_enabled": bool(
                (result.get("trace_replay_provenance") or {}).get("enabled")
                or (trace_replay and eval_method.startswith("bj_"))
            ),
            "trace_replay_strategy": (
                (result.get("trace_replay_provenance") or {}).get("strategy") or ""
            ),
            "trace_replay_fallback": (
                (result.get("trace_replay_provenance") or {}).get("fallback_reason")
                or ""
            ),
        })

    # If we recorded a prep item, prepend it so it's written before file rows
    if write_prep and prep_row is not None:
        return [prep_row] + per_file_rows
    return per_file_rows 