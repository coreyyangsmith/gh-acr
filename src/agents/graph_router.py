from __future__ import annotations

"""Central graph builder/router allowing selection of processing and evaluation modes.

Usage
-----
>>> from src.agents.graph_router import build_graph
>>> app = build_graph(process_mode="api", eval_method="agent")
"""

from typing import Literal

ProcessMode = Literal["api", "clone"]
EvalMethod = Literal["agent", "base"]


def build_graph(*, process_mode: ProcessMode = "api", eval_method: EvalMethod = "agent"):
    """Return a compiled LangGraph application.

    Parameters
    ----------
    process_mode
        "api"   – use GitHub API for fetching file contents (lightweight).
        "clone" – clone the full repository locally.
    eval_method
        "agent" – use the LLM-based merge resolver (default).
        "base"  – use the parent-A stub resolver.
    """

    if process_mode == "api":
        from ..merge_pipeline import pipeline_api as _pipe
    elif process_mode == "clone":
        from ..merge_pipeline import pipeline_clone as _pipe
    else:
        raise ValueError(f"Unknown process_mode {process_mode!r}; choose 'api' or 'clone'.")

    return _pipe.build_graph(eval_method=eval_method) 