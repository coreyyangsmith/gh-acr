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
from src.config.model_costs import MODEL_COSTS
from src.utils.rate_limiter import LimiterRegistry
from src.utils.logger import setup_logger

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

    # Attach Langfuse callback only if explicitly enabled AND startup marked it ready
    langfuse_handler = None
    if str(os.getenv("LANGFUSE_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on") and str(os.getenv("LANGFUSE_READY", "0")).strip() in ("1", "true", "TRUE"):
        try:
            from langfuse.langchain import CallbackHandler as LangfuseCallback  # type: ignore
            langfuse_handler = LangfuseCallback()
            try:
                app = app.with_config({"callbacks": [langfuse_handler]})
            except Exception:
                pass
        except Exception:
            langfuse_handler = None

    init_state = {
        "scenario_id": scenario_id,
        "status": "start",
        "logs": [],
        "model_name": model_name,
    }

    # Derive a session id for tracing (per conversation/scenario)
    session_id = f"{scenario_id}:{eval_method}"
    if langfuse_handler is not None:
        try:
            from langfuse.decorators import langfuse_context  # type: ignore
            try:
                langfuse_context.update_current_trace(session_id=session_id)  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Optional pre-run preparation: clone repository (clone mode only)
    # This time is recorded separately as a 'prep' line item and excluded from
    # the main processing time measurement.
    # ---------------------------------------------------------------------------
    prep_row: Dict[str, Any] | None = None
    if write_prep and (process_mode or "").strip().lower() == "clone":
        try:
            from src.dataset.loader import load_benchmark  # local import to avoid cyclic deps
            from src.merge_pipeline.pipeline_clone import _clone_repo  # type: ignore

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
                _clone_repo(sample, checkout_dir=Path.cwd() / "repos")
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

    logger.info("Starting scenario %s with method=%s", scenario_id, eval_method)
    invoke_cfg: Dict[str, Any] = {
        "configurable": {"thread_id": f"scn-{scenario_id}"},
        "metadata": {
            "langfuse_session_id": session_id,
        },
        "run_name": f"{eval_method}-scenario-{scenario_id}",
    }
    if langfuse_handler is not None:
        invoke_cfg["callbacks"] = [langfuse_handler]

    result = await app.ainvoke(init_state, config=invoke_cfg)

    elapsed_sec = time.perf_counter() - start_ts

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

        # Multi-style outputs: include bypass to mirror multi folder structure
        is_multi_like = eval_method in ("multi", "bypass", "bypass_multi", "bypass2", "bypass3", "bypass4", "dynamic", "bypass_only", "bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5")
        if is_multi_like:
            # Skip redundant folders for bypass7; it writes per-file artifacts under <llm_out_dir>/<file_slug>/
            if eval_method not in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                (llm_out_dir / "summaries").mkdir(exist_ok=True)
                (llm_out_dir / "reviews").mkdir(exist_ok=True)
            if eval_method == "dynamic":
                (llm_out_dir / "prompts").mkdir(exist_ok=True)
            # Persist the merge plan as text (skip for bypass_only and bypass7 – bypass7 writes per-file only)
            if eval_method not in ("bypass_only", "bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                try:
                    import json
                    plan_obj = result.get("conflict_plan", {})
                    # Only write if a plan exists (avoid empty file spam)
                    if plan_obj:
                        (llm_out_dir / "plan.txt").write_text(
                            json.dumps(plan_obj, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                    if eval_method == "dynamic":
                        dyn_prompts = result.get("dynamic_prompts", {}) or {}
                        if dyn_prompts:
                            (llm_out_dir / "prompts.json").write_text(
                                json.dumps(dyn_prompts, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                except Exception:
                    pass
            # Persist bypass analyzer output if present
            if eval_method in ("bypass", "bypass_multi", "bypass_only"):
                decision = str(result.get("bypass_decision", "")).strip()
                analyzer_raw = str(result.get("bypass_analyzer_output", "")).strip()
                try:
                    (llm_out_dir / "bypass_analyzer.txt").write_text(
                        f"decision: {decision}\n\nraw:\n{analyzer_raw}\n", encoding="utf-8"
                    )
                except Exception:
                    pass
                # If bypass_only defaulted to 'A' after invalid outputs, persist a flag
                if eval_method == "bypass_only" and result.get("bypass_only_defaulted"):
                    try:
                        (llm_out_dir / "bypass_only_defaulted.flag").write_text("defaulted_to_A", encoding="utf-8")
                    except Exception:
                        pass

    for file_path in files:
        file_slug = file_path.replace("/", "_").replace("\\", "_")
        file_dir = scenario_dir / file_slug
        file_dir.mkdir(parents=True, exist_ok=True)

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
        except Exception:
            pass

        # Duplicate LLM output into central per-method directory (simple/, multi/, bypass/)
        if eval_method != "base":
            # Agent-specific file base within its directory
            is_bypass_like = eval_method in ("bypass", "bypass_multi", "bypass2", "bypass3", "bypass4", "bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5")
            base_name = f"bypass_{file_slug}" if is_bypass_like else file_slug

            if eval_method not in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                (llm_out_dir / f"{base_name}.txt").write_text(
                    result["resolved_contents"].get(file_path, ""), encoding="utf-8"
                )
                # Also persist final diff if available from multi/bypass resolver
                final_diff_map = result.get("final_diffs", {})
                if final_diff_map:
                    diff_text = final_diff_map.get(file_path, "")
                    (llm_out_dir / f"{base_name}.diff").write_text(diff_text, encoding="utf-8")
                    # Also provide a .txt variant to ensure all agent outputs are available as text files
                    (llm_out_dir / f"{base_name}_final_diff.txt").write_text(diff_text, encoding="utf-8")

            # Provide clearly named final artifacts for bypass family
            if eval_method in ("bypass", "bypass2", "bypass3", "bypass4"):
                merged_out = result["resolved_contents"].get(file_path, "")
                try:
                    (llm_out_dir / f"{file_slug}__FINAL_MERGED.txt").write_text(merged_out, encoding="utf-8")
                except Exception:
                    pass
                final_diff_map = result.get("final_diffs", {})
                if final_diff_map:
                    diff_text = final_diff_map.get(file_path, "")
                    try:
                        (llm_out_dir / f"{file_slug}__FINAL_DIFF.diff").write_text(diff_text, encoding="utf-8")
                        (llm_out_dir / f"{file_slug}__FINAL_DIFF.txt").write_text(diff_text, encoding="utf-8")
                    except Exception:
                        pass

            # ---------------- multi/bypass extra outputs -------------------
            if eval_method in ("multi", "bypass", "bypass_multi", "dynamic", "bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                summaries = result.get("summaries", {}).get(file_path, {})
                if eval_method in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                    # Write simplified names for bypass7 inside a per-file subdirectory
                    per_file_agent_dir = llm_out_dir / file_slug
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    (per_file_agent_dir / "a_summary.txt").write_text(
                        summaries.get("summary_a", ""), encoding="utf-8"
                    )
                    (per_file_agent_dir / "b_summary.txt").write_text(
                        summaries.get("summary_b", ""), encoding="utf-8"
                    )
                    # Also write plan at per-file level if useful (JSON subset per file)
                    try:
                        import json
                        plan_obj = result.get("conflict_plan", {}) or {}
                        # Always write a per-file plan; default to "merge" if missing
                        single_plan = {file_path: plan_obj.get(file_path, "merge")}
                        (per_file_agent_dir / "plan.txt").write_text(
                            json.dumps(single_plan, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        # Also include the full agent plan for convenience under the slug folder
                        if plan_obj:
                            (per_file_agent_dir / "agent_plan.txt").write_text(
                                json.dumps(plan_obj, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                    except Exception:
                        pass
                else:
                    prefix = "bypass_" if is_bypass_like else ""
                    (llm_out_dir / "summaries" / f"{prefix}{file_slug}_A.txt").write_text(
                        summaries.get("summary_a", ""), encoding="utf-8"
                    )
                    (llm_out_dir / "summaries" / f"{prefix}{file_slug}_B.txt").write_text(
                        summaries.get("summary_b", ""), encoding="utf-8"
                    )
                if eval_method == "dynamic":
                    dyn_prompts = result.get("dynamic_prompts", {}) or {}
                    dyn_text = str(dyn_prompts.get(file_path, "")).strip()
                    try:
                        (llm_out_dir / "prompts" / f"{file_slug}.txt").write_text(dyn_text, encoding="utf-8")
                    except Exception:
                        pass
                reviews = result.get("reviews", {})
                if reviews:
                    if eval_method in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                        # Write review iterations if present
                        per_file_agent_dir = llm_out_dir / file_slug
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
                    else:
                        (llm_out_dir / "reviews" / f"{prefix}{file_slug}.txt").write_text(
                            reviews.get(file_path, ""), encoding="utf-8"
                        )
                # Persist structured review results and accumulated feedback history as txt
                rr = result.get("review_results", {}) or {}
                rr_item = rr.get(file_path)
                if isinstance(rr_item, dict):
                    outcome = str(rr_item.get("outcome", "")).strip()
                    rationale = str(rr_item.get("rationale", "")).strip()
                    if eval_method in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                        per_file_agent_dir = llm_out_dir / file_slug
                        per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                        (per_file_agent_dir / "review_results.txt").write_text(
                            f"outcome: {outcome}\n\nrationale:\n{rationale}\n", encoding="utf-8"
                        )
                    else:
                        (llm_out_dir / "reviews" / f"{prefix}{file_slug}_results.txt").write_text(
                            f"outcome: {outcome}\n\nrationale:\n{rationale}\n", encoding="utf-8"
                        )
                feedback_map = result.get("review_feedback", {}) or {}
                fb_text = str(feedback_map.get(file_path, "")).strip()
                if fb_text:
                    if eval_method in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                        per_file_agent_dir = llm_out_dir / file_slug
                        per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                        (per_file_agent_dir / "review_feedback.txt").write_text(
                            fb_text, encoding="utf-8"
                        )
                    else:
                        (llm_out_dir / "reviews" / f"{prefix}{file_slug}_feedback.txt").write_text(
                            fb_text, encoding="utf-8"
                        )
                fb_hist = result.get("review_feedback_history", {}) or {}
                hist_entries = fb_hist.get(file_path, [])
                if hist_entries:
                    if eval_method in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                        per_file_agent_dir = llm_out_dir / file_slug
                        per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                        (per_file_agent_dir / "review_feedback_history.txt").write_text(
                            "\n\n".join(hist_entries), encoding="utf-8"
                        )
                    else:
                        (llm_out_dir / "reviews" / f"{prefix}{file_slug}_feedback_history.txt").write_text(
                            "\n\n".join(hist_entries), encoding="utf-8"
                        )

                # For bypass7, also persist resolution iterations if present
                if eval_method in ("bypass7", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
                    res_hist = result.get("resolution_history", {}) or {}
                    r_items = res_hist.get(file_path, [])
                    per_file_agent_dir = llm_out_dir / file_slug
                    per_file_agent_dir.mkdir(parents=True, exist_ok=True)
                    if r_items:
                        for idx, txt in enumerate(r_items, start=1):
                            (per_file_agent_dir / f"resolution{idx}.txt").write_text(txt, encoding="utf-8")
                    else:
                        # Single resolution fallback name
                        (per_file_agent_dir / "resolution1.txt").write_text(
                            result["resolved_contents"].get(file_path, ""), encoding="utf-8"
                        )

                    # Also include the final merged and diff artifacts inside the per-file directory
                    merged_out = result["resolved_contents"].get(file_path, "")
                    try:
                        (per_file_agent_dir / f"bypass_{file_slug}.txt").write_text(merged_out, encoding="utf-8")
                    except Exception:
                        pass
                    final_diff_map = result.get("final_diffs", {})
                    if final_diff_map:
                        diff_text = final_diff_map.get(file_path, "")
                        try:
                            (per_file_agent_dir / f"bypass_{file_slug}.diff").write_text(diff_text, encoding="utf-8")
                            (per_file_agent_dir / f"bypass_{file_slug}_final_diff.txt").write_text(diff_text, encoding="utf-8")
                            # Convenience duplicate: final_diff.txt without prefix
                            (per_file_agent_dir / "final_diff.txt").write_text(diff_text, encoding="utf-8")
                        except Exception:
                            pass

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
    if eval_method in ("bypass", "bypass_multi", "bypass2", "bypass3", "bypass4", "bypass_only", "bypass_only2", "bypass5", "bypass6", "bypass7", "bypass8", "new_bypass", "new_bypass2", "new_bypass3", "new_bypass4", "new_bypass5"):
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