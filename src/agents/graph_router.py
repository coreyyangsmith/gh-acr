from __future__ import annotations

"""Central graph builder/router allowing selection of processing and evaluation modes.

Usage
-----
>>> from src.agents.graph_router import build_graph
>>> app = build_graph(process_mode="clone", eval_method="agent")
"""

from typing import Literal
from src.config.eval_methods import EvalMethod

ProcessMode = Literal["clone"]

def build_graph(*, process_mode: ProcessMode = "clone", eval_method: EvalMethod = "agent"):
    """Return a compiled LangGraph application.

    Parameters
    ----------
    process_mode
        "clone" – clone the full repository locally.
    eval_method
        "agent"  – LLM-based resolver (default)
        "base_a" – baseline Parent-A resolver (alias: "base")
        "base_b" – baseline Parent-B resolver
        "multi"  – multi-agent resolver
        "bypass" – bypass resolver
    """

    if process_mode == "clone":
        from ..merge_pipeline import pipeline_clone as _pipe
    else:
        raise ValueError(f"Unknown process_mode {process_mode!r}; choose 'clone'.")

    return _pipe.build_graph(eval_method=eval_method) 