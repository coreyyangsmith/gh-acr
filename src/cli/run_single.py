from __future__ import annotations

import asyncio
from typing import Annotated, Literal

# Ensure one-time global startup (env, logging, tracing) before anything else
import src.startup  # noqa: F401

import tyro
import pandas as pd
from pathlib import Path


# Graph builder will be selected at runtime based on --mode flag
from src.cli.runner import run_and_save_report
from src.agents.graph_router import build_graph


def main(
    scenario_id: Annotated[
        str,
        tyro.conf.Positional,
        "Identifier matching the **id** column (integer) *or* the scenario's slug.",
    ],
    mode: Literal["api", "clone"] = "api",
    eval_method: Literal["base_a", "base_b", "agent", "multi"] = "agent",
    model_name: str | None = None,
):
    """Run the merge-resolution pipeline for a single *scenario_id*.

    Parameters
    ----------
    scenario_id
        Identifier of the dataset scenario (index or slug).
    mode
        "api"  – use the lightweight GitHub API pipeline (default).
        "clone" – clone the full repository locally before processing.
    eval_method
        "agent"  (default) – use the LLM merge agent.
        "base_a" (alias: "base") – baseline parent-A resolver.
        "base_b" – baseline parent-B resolver.
    """
    asyncio.run(
        _run(scenario_id=scenario_id, mode=mode, eval_method=eval_method, model_name=model_name),
    )


async def _run(scenario_id: str, mode: str, eval_method: str, model_name: str | None):
    """Internal async runner."""

    app = build_graph(process_mode=mode, eval_method=eval_method)
    output_root = Path.cwd() / "data" / "output"
    
    per_file_results = await run_and_save_report(app, scenario_id, output_root, eval_method=eval_method, model_name=model_name)

    # Append the evaluation to a CSV (one row per file)
    results_path = Path.cwd() / "data" / "results.csv"
    df = pd.DataFrame(per_file_results)

    header = not results_path.exists() or results_path.stat().st_size == 0
    df.to_csv(results_path, mode="a", header=header, index=False)

    print(f"\nEvaluation metrics appended to {results_path}")


if __name__ == "__main__":
    tyro.cli(main)
