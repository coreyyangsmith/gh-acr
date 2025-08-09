from __future__ import annotations

"""Shared logic for running the merge-resolution pipeline.

This module is the single source of truth for processing a scenario and
writing its output files. It is imported by both `run_single` and `run_batch`.
"""
from pathlib import Path
from typing import Dict, Any
import os
import time
from src.config.model_costs import MODEL_COSTS
from src.utils.rate_limiter import LimiterRegistry
from src.utils.logger import setup_logger

async def run_and_save_report(app, scenario_id: str, output_root: Path, *, eval_method: str, model_name: str | None = None):
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
    }

    start_ts = time.perf_counter()

    logger.info("Starting scenario %s with method=%s", scenario_id, eval_method)
    result = await app.ainvoke(
        init_state,
        config={"configurable": {"thread_id": f"scn-{scenario_id}"}},
    )

    elapsed_sec = time.perf_counter() - start_ts

    # ---------------------------------------------------------------------------
    # Write full report to files
    # ---------------------------------------------------------------------------

    sample_row = result["sample_row"]
    df_index = sample_row["df_index"]
    scenario_dir = output_root / str(df_index)
    files = sample_row["scenario_json"]["files_in_merge_conflict"]

    # -------------------------------------------------------------------
    # Prepare per-eval-method LLM output directory (simple/, multi/, …)
    # -------------------------------------------------------------------
    if eval_method != "base":
        llm_out_dir = scenario_dir / eval_method
        llm_out_dir.mkdir(parents=True, exist_ok=True)

        # For multi-agent we will create sub-folders for each stage
        if eval_method == "multi":
            (llm_out_dir / "summaries").mkdir(exist_ok=True)
            (llm_out_dir / "reviews").mkdir(exist_ok=True)

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

        # Duplicate LLM output into central per-method directory (simple/, multi/)
        if eval_method != "base":
            (llm_out_dir / f"{file_slug}.txt").write_text(
                result["resolved_contents"].get(file_path, ""), encoding="utf-8"
            )

            # ---------------- multi-agent extra outputs -------------------
            if eval_method == "multi":
                summaries = result.get("summaries", {}).get(file_path, {})
                (llm_out_dir / "summaries" / f"{file_slug}_A.txt").write_text(
                    summaries.get("summary_a", ""), encoding="utf-8"
                )
                (llm_out_dir / "summaries" / f"{file_slug}_B.txt").write_text(
                    summaries.get("summary_b", ""), encoding="utf-8"
                )
                reviews = result.get("reviews", {})
                if reviews:
                    (llm_out_dir / "reviews" / f"{file_slug}.txt").write_text(
                        reviews.get(file_path, ""), encoding="utf-8"
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
    model_cfg = MODEL_COSTS.get(model_name, {})
    if not model_cfg and not model_name.startswith("openai/"):
        model_cfg = MODEL_COSTS.get(f"openai/{model_name}", {})
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
    if model_name and not model_name.startswith("openai/"):
        price_key = f"openai/{model_name}"
    else:
        price_key = model_name
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
    repo_slug: str = sample_row["name"]  # e.g. "owner/repo"
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
            # Individual token categories
            "tokens_system_prompt": tok_stats.get("system_prompt", 0),
            "tokens_original": tok_stats.get("original", 0),
            "tokens_diff_a": tok_stats.get("diff_a", 0),
            "tokens_diff_b": tok_stats.get("diff_b", 0),
            "tokens_output": tok_stats.get("output", 0),

            # Aggregated counts
            "tokens_total_input": in_tok_file,
            "tokens_total": in_tok_file + out_tok_file,

            # Back-compat combined fields
            "tokens_in": in_tok_file,
            "tokens_out": out_tok_file,
            "cost_in": round(cost_in_file, 6),
            "cost_out": round(cost_out_file, 6),
            "total_cost": round(cost_in_file + cost_out_file, 6),
            "processing_time_s": round(elapsed_sec, 3),
            "difficulty": difficulty,
        })

    return per_file_rows 