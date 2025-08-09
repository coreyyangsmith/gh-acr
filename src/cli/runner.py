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

async def run_and_save_report(app, scenario_id: str, output_root: Path, *, eval_method: str, model_name: str | None = None):
    """Run the pipeline for one scenario and save its report to disk.

    The console output will be the final evaluation summary, plus a confirmation
    of where the full report was saved.
    """

    init_state = {
        "scenario_id": scenario_id,
        "status": "start",
        "logs": [],
        "model_name": model_name,
    }

    start_ts = time.perf_counter()

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

    print(f"\n\n--- Report for {scenario_id} ({df_index}) ---")
    print(f"    - Full report saved to: {scenario_dir}")
    print(f"    - Overall exact match: {eval_['overall_exact_match']}")
    print(f"    - Overall BLEU-3: {eval_.get('overall_bleu3', 'N/A')}")
    print(f"    - Overall ROUGE-L: {eval_.get('overall_rouge_l', 'N/A')}")

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
        total_in_tokens = sum(
            v.get("system_prompt", 0)
            + v.get("original", 0)
            + v.get("diff_a", 0)
            + v.get("diff_b", 0)
            for v in token_map.values()
        )
        total_out_tokens = sum(v.get("output", 0) for v in token_map.values())
    else:
        total_in_tokens = total_out_tokens = 0

    cost_in = (total_in_tokens / 1000) * input_cost_rate
    cost_out = (total_out_tokens / 1000) * output_cost_rate
    total_cost = cost_in + cost_out

    print(f"    - Tokens in: {total_in_tokens}  | cost: ${cost_in:.4f}")
    print(f"    - Tokens out: {total_out_tokens} | cost: ${cost_out:.4f}")
    print(f"    - Estimated total LLM cost: ${total_cost:.4f} (model: {model_name})")

    print(f"    - Processing time: {elapsed_sec:.2f} s")

    # -------------------------------------------------------------------
    # Rate limiter metrics (if any LLM calls were made)
    # -------------------------------------------------------------------
    rl_metrics = LimiterRegistry.metrics()
    if rl_metrics:
        print("    - Rate limit activity:")
        for key, m in rl_metrics.items():
            print(
                f"        · {key}: waits={m['wait_events']} total_wait={m['total_wait_time_s']:.2f}s retries={m['total_retries']} last_delay={m['last_retry_delay_s']:.2f}s"
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