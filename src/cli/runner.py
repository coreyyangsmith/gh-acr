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
from src.config.model_costs import MODEL_COSTS
from src.utils.rate_limiter import LimiterRegistry
from src.utils.logger import setup_logger


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
        
        logger.info(
            "[%s] scenario=%s, status=%s, files=%d, diff_a_chars=%d, diff_b_chars=%d, resolved_chars=%d",
            context, scenario_id, status, len(diffs_a), total_diff_a_chars, total_diff_b_chars, total_resolved_chars
        )
        
        # Log bypass-specific info
        bypass_decision = state.get("bypass_decision", "")
        bypass_method = state.get("bypass_method", "")
        if bypass_decision or bypass_method:
            logger.info(
                "[%s] bypass_decision=%s, bypass_method=%s, resolution_history_files=%d",
                context, bypass_decision, bypass_method, len(resolution_history)
            )
        
        # Log parent contents availability (important for ALL_A/ALL_B bypass)
        logger.info(
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
            logger.info(
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
]

async def run_and_save_report(app, scenario_id: str, output_root: Path, *, eval_method: str, model_name: str | None = None, process_mode: str | None = None, write_prep: bool = True):
    """Run the pipeline for one scenario and save its report to disk.

    The console output will be the final evaluation summary, plus a confirmation
    of where the full report was saved.
    """

    logger = setup_logger(__name__)

    init_state = {
        "scenario_id": scenario_id,
        "status": "start",
        "logs": [],
        "model_name": model_name,
        "eval_method": eval_method,
    }

    # ---------------------------------------------------------------------------
    # Optional pre-run preparation: clone repository (clone mode only)
    # This time is recorded separately as a 'prep' line item and excluded from
    # the main processing time measurement.
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
                _clone_repo(sample, checkout_dir=_checkout_root())
                prep_elapsed = time.perf_counter() - prep_start
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
    )

    set_run_context(
        eval_method=eval_method,
        scenario_id=str(scenario_id),
        model_name=model_name,
    )
    # Attach LangFuse CallbackHandler + langfuse_trace_name so the graph root
    # trace is named "{eval_method}-scenario-{id}" (passthrough when disabled).
    invoke_cfg = build_langfuse_invoke_config(
        {
            "configurable": {"thread_id": f"scn-{scenario_id}"},
            "run_name": make_trace_name(eval_method, str(scenario_id)),
        },
        model_name=model_name,
    )
    try:
        with scenario_observation(make_trace_name(eval_method, str(scenario_id))):
            result = await app.ainvoke(init_state, config=invoke_cfg)
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
    # Place outputs under data/<model_name>/<id>
    if eval_method in ("base_a", "base_b", "base", "prep"):
        raw_model_dir = "nan"
    else:
        raw_model_dir = (model_name or os.getenv("OPENAI_MODEL", "")).strip() or "nan"
    # Also replace ':' so Windows paths are safe (e.g., 'groq:llama' → 'groq_llama')
    safe_model_dir = (
        raw_model_dir.replace("/", "_").replace("\\", "_").replace(":", "_").strip() or "nan"
    )
    scenario_dir = output_root / safe_model_dir / str(df_index)
    files = sample_row["scenario_json"]["files_in_merge_conflict"]

    # Note: commit messages are written inside each per-file directory below

    # -------------------------------------------------------------------
    # Prepare per-eval-method LLM output directory (simple/, multi/, bypass/ …)
    # -------------------------------------------------------------------
    if eval_method != "base":
        llm_out_dir = scenario_dir / eval_method
        llm_out_dir.mkdir(parents=True, exist_ok=True)

        # Multi-style outputs: bypass7 / force_mix write per-file artifacts under <llm_out_dir>/<file_slug>/
        is_multi_like = eval_method in ("bypass7", "force_mix")

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
    
    for file_path in files:
        file_slug = file_path.replace("/", "_").replace("\\", "_")
        file_dir = scenario_dir / file_slug
        file_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug("[%s] Processing file: %s → slug: %s", eval_method, file_path, file_slug)

        # Write content and diff files
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
        # Persist diff between ancestor and ground-truth merge result
        (file_dir / "ground_truth.diff").write_text(
            result.get("diffs_truth", {}).get(file_path, ""), encoding="utf-8"
        )

        # Write commit messages into each file directory (clone mode only fields)
        try:
            cm_a = str(result.get("commit_messages_a", "")).strip()
            cm_b = str(result.get("commit_messages_b", "")).strip()
            if cm_a:
                (file_dir / "a_commit_message.txt").write_text(cm_a, encoding="utf-8")
            if cm_b:
                (file_dir / "b_commit_message.txt").write_text(cm_b, encoding="utf-8")
        except Exception as e:
            logger.warning("[%s] Failed to write commit messages for %s: %s", eval_method, file_path, e)

        # Duplicate LLM output into central per-method directory (simple/, multi/, bypass/)
        if eval_method != "base":
            # Agent-specific file base within its directory
            is_bypass_like = eval_method in ("bypass7", "force_mix")
            base_name = f"bypass_{file_slug}" if is_bypass_like else file_slug

            # ---------------- bypass7 / force_mix extra outputs -------------------
            if eval_method in ("bypass7", "force_mix"):
                summaries = result.get("summaries", {}).get(file_path, {})
                # Write per-file artifacts inside a per-file subdirectory
                per_file_agent_dir = llm_out_dir / file_slug
                try:
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    logger.debug("[%s] Created per-file dir: %s", eval_method, per_file_agent_dir)
                except Exception as e:
                    logger.error("[%s] Failed to create per-file dir %s: %s", eval_method, per_file_agent_dir, e)

                summary_a = summaries.get("summary_a", "")
                summary_b = summaries.get("summary_b", "")
                if not summary_a:
                    logger.warning("[%s] EMPTY summary_a for %s", eval_method, file_path)
                if not summary_b:
                    logger.warning("[%s] EMPTY summary_b for %s", eval_method, file_path)

                try:
                    (per_file_agent_dir / "a_summary.txt").write_text(summary_a, encoding="utf-8")
                    (per_file_agent_dir / "b_summary.txt").write_text(summary_b, encoding="utf-8")
                    logger.debug("[%s] Wrote summaries for %s (a=%d, b=%d chars)",
                                 eval_method, file_path, len(summary_a), len(summary_b))
                except Exception as e:
                    logger.warning("[%s] Failed to write summaries for %s: %s", eval_method, file_path, e)

                try:
                    import json
                    plan_obj = result.get("conflict_plan", {}) or {}
                    single_plan = {file_path: plan_obj.get(file_path, "merge")}
                    (per_file_agent_dir / "plan.txt").write_text(
                        json.dumps(single_plan, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    if plan_obj:
                        (per_file_agent_dir / "agent_plan.txt").write_text(
                            json.dumps(plan_obj, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                    logger.debug("[%s] Wrote plan for %s", eval_method, file_path)
                except Exception as e:
                    logger.warning("[%s] Failed to write plan for %s: %s", eval_method, file_path, e)

                reviews = result.get("reviews", {})
                if reviews:
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    hist = result.get("review_history", {}) or {}
                    items = hist.get(file_path, [])
                    if items:
                        for idx, txt in enumerate(items, start=1):
                            (per_file_agent_dir / f"review{idx}.txt").write_text(txt, encoding="utf-8")
                    else:
                        (per_file_agent_dir / "review.txt").write_text(
                            reviews.get(file_path, ""), encoding="utf-8"
                        )

                rr = result.get("review_results", {}) or {}
                rr_item = rr.get(file_path)
                if isinstance(rr_item, dict):
                    outcome = str(rr_item.get("outcome", "")).strip()
                    rationale = str(rr_item.get("rationale", "")).strip()
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    (per_file_agent_dir / "review_results.txt").write_text(
                        f"outcome: {outcome}\n\nrationale:\n{rationale}\n", encoding="utf-8"
                    )

                feedback_map = result.get("review_feedback", {}) or {}
                fb_text = str(feedback_map.get(file_path, "")).strip()
                if fb_text:
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    (per_file_agent_dir / "review_feedback.txt").write_text(fb_text, encoding="utf-8")

                fb_hist = result.get("review_feedback_history", {}) or {}
                hist_entries = fb_hist.get(file_path, [])
                if hist_entries:
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    (per_file_agent_dir / "review_feedback_history.txt").write_text(
                        "\n\n".join(hist_entries), encoding="utf-8"
                    )

                # Also persist resolution iterations if present
                if True:
                    res_hist = result.get("resolution_history", {}) or {}
                    r_items = res_hist.get(file_path, [])
                    per_file_agent_dir = llm_out_dir / file_slug
                    
                    # Log resolution history state for debugging
                    bypass_decision = result.get("bypass_decision", "unknown")
                    logger.info(
                        "[%s] Writing resolution for %s: bypass_decision=%s, resolution_history_items=%d",
                        eval_method, file_path, bypass_decision, len(r_items)
                    )
                    
                    try:
                        per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        logger.error("[%s] Failed to create resolution dir %s: %s", eval_method, per_file_agent_dir, e)
                    
                    if r_items:
                        for idx, txt in enumerate(r_items, start=1):
                            txt_len = len(txt) if txt else 0
                            if txt_len == 0:
                                logger.warning("[%s] EMPTY resolution_history[%d] for %s", eval_method, idx, file_path)
                            try:
                                (per_file_agent_dir / f"resolution{idx}.txt").write_text(txt, encoding="utf-8")
                                logger.debug("[%s] Wrote resolution%d.txt for %s (%d chars)", eval_method, idx, file_path, txt_len)
                            except Exception as e:
                                logger.error("[%s] Failed to write resolution%d.txt for %s: %s", eval_method, idx, file_path, e)
                    else:
                        # Single resolution fallback name - this happens for ALL_A/ALL_B bypass
                        resolved_content = result["resolved_contents"].get(file_path, "")
                        resolved_len = len(resolved_content) if resolved_content else 0
                        if resolved_len == 0:
                            # This is the key diagnostic for missing files
                            logger.warning(
                                "[%s] EMPTY resolved_contents for %s (bypass_decision=%s). "
                                "Check parent_a_contents/parent_b_contents population.",
                                eval_method, file_path, bypass_decision
                            )
                            # Log what keys exist in resolved_contents for debugging
                            resolved_keys = list(result.get("resolved_contents", {}).keys())
                            logger.warning("[%s] Available resolved_contents keys: %s", eval_method, resolved_keys[:10])
                        try:
                            (per_file_agent_dir / "resolution1.txt").write_text(resolved_content, encoding="utf-8")
                            logger.info("[%s] Wrote resolution1.txt for %s (%d chars)", eval_method, file_path, resolved_len)
                        except Exception as e:
                            logger.error("[%s] Failed to write resolution1.txt for %s: %s", eval_method, file_path, e)

                    # Also include the final merged and diff artifacts inside the per-file directory
                    merged_out = result["resolved_contents"].get(file_path, "")
                    merged_len = len(merged_out) if merged_out else 0
                    try:
                        (per_file_agent_dir / f"bypass_{file_slug}.txt").write_text(merged_out, encoding="utf-8")
                        logger.info("[%s] Wrote bypass_%s.txt (%d chars)", eval_method, file_slug, merged_len)
                    except Exception as e:
                        logger.error("[%s] Failed to write bypass_%s.txt for %s: %s", eval_method, file_slug, file_path, e)
                    
                    final_diff_map = result.get("final_diffs", {})
                    if final_diff_map:
                        diff_text = final_diff_map.get(file_path, "")
                        diff_len = len(diff_text) if diff_text else 0
                        try:
                            (per_file_agent_dir / f"bypass_{file_slug}.diff").write_text(diff_text, encoding="utf-8")
                            (per_file_agent_dir / f"bypass_{file_slug}_final_diff.txt").write_text(diff_text, encoding="utf-8")
                            # Convenience duplicate: final_diff.txt without prefix
                            (per_file_agent_dir / "final_diff.txt").write_text(diff_text, encoding="utf-8")
                            logger.debug("[%s] Wrote diff files for %s (%d chars)", eval_method, file_path, diff_len)
                        except Exception as e:
                            logger.error("[%s] Failed to write diff files for %s: %s", eval_method, file_path, e)
                    else:
                        logger.debug("[%s] No final_diffs available for %s", eval_method, file_path)

    # Summary log for bypass7 / force_mix file writing
    if eval_method in ("bypass7", "force_mix"):
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
    # Normalize for cost lookup: support openai/, groq: and local:
    def _price_key(name: str) -> str:
        try:
            if name.startswith("openai/"):
                return name
            if name.startswith("groq:"):
                return "groq/" + name.split(":", 1)[1]
            if name.startswith("local:"):
                return name
            return f"openai/{name}"
        except Exception:
            return name

    model_cfg = MODEL_COSTS.get(_price_key(model_name), {})
    input_cost_rate = float(model_cfg.get("input_cost_per_1k", 0.0))
    output_cost_rate = float(model_cfg.get("output_cost_per_1k", 0.0))

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

    # Compute cost using the model that actually ran for this scenario
    # Normalize model_name to the key used in MODEL_COSTS
    price_key = _price_key(model_name) if model_name else ""
    model_cfg = MODEL_COSTS.get(price_key or "", {}) or MODEL_COSTS.get(model_name or "", {})
    input_cost_rate = float(model_cfg.get("input_cost_per_1k", 0.0))
    output_cost_rate = float(model_cfg.get("output_cost_per_1k", 0.0))

    cost_in = (float(total_in_tokens) / 1000.0) * input_cost_rate
    cost_out = (float(total_out_tokens) / 1000.0) * output_cost_rate
    total_cost = cost_in + cost_out

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
    if eval_method in ("bypass7", "force_mix"):
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

        cost_in_file = (in_tok_file / 1000) * input_cost_rate
        cost_out_file = (out_tok_file / 1000) * output_cost_rate

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
            "total_cost": round(cost_in_file + cost_out_file, 6),
            "processing_time_s": round(elapsed_sec, 3),
            "difficulty": difficulty,
            "project_size": sample_row.get("project_size", ""),
        })

    # If we recorded a prep item, prepend it so it's written before file rows
    if write_prep and prep_row is not None:
        return [prep_row] + per_file_rows
    return per_file_rows 