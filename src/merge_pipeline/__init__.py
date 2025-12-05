"""Merge pipeline package for Git merge conflict resolution.

This package contains the core LangGraph pipeline that orchestrates the
entire merge conflict resolution process, from loading scenarios to
evaluating results.

Pipeline Stages
---------------
1. **load_sample**: Load a scenario from the benchmark CSV
2. **prepare_context**: Clone repo, read files, compute diffs
3. **resolve**: Apply the selected resolution strategy
4. **evaluate**: Compare results against ground truth

Modules
-------
- **pipeline_clone**: Main pipeline implementation using git clone

The pipeline uses LangGraph's StateGraph to define a directed acyclic graph
of processing nodes. Each node receives a state dict and returns an updated
state dict.

State Keys
----------
Input:
- scenario_id: str - ID of the scenario to process
- model_name: str - LLM model to use (optional)

Intermediate:
- sample_row: dict - Loaded scenario data from CSV
- ancestor_contents: Dict[str, str] - Base file versions
- parent_a_contents: Dict[str, str] - Parent A versions
- parent_b_contents: Dict[str, str] - Parent B versions
- diffs_a: Dict[str, str] - Parent A diffs
- diffs_b: Dict[str, str] - Parent B diffs

Output:
- resolved_contents: Dict[str, str] - Merged file versions
- truth_contents: Dict[str, str] - Ground truth versions
- evaluation: dict - Metric scores (exact_match, bleu, rouge)

Example Usage
-------------
>>> from src.merge_pipeline.pipeline_clone import build_graph
>>> 
>>> # Build and run the pipeline
>>> app = build_graph(eval_method="bypass7")
>>> result = await app.ainvoke({
...     "scenario_id": "123",
...     "model_name": "openai/gpt-4o-mini",
...     "status": "start",
... })
>>> 
>>> print(result["evaluation"]["overall_exact_match"])
True
"""

from .pipeline_clone import build_graph, make_graph

__all__ = [
    "build_graph",
    "make_graph",
]
