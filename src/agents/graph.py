"""Public facade & entrypoints for serving LangGraph apps.

This keeps higher-level agent abstractions decoupled from the low-level
merge-resolution logic located in :pymod:`src.merge_pipeline`.
"""

from typing import Any
from langchain_core.runnables import RunnableConfig

from .graph_router import build_graph


def make_graph(config: RunnableConfig | None = None) -> Any:  # noqa: D401
    """Return a compiled LangGraph app using values from ``config``.

    Expected configurable keys (with defaults):
    - process_mode: "clone" (default: "clone")
    - eval_method:  "agent" | "base_a" | "base_b" | "multi" | "bypass_multi" (default: "agent")
    """

    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    process_mode = cfg.get("process_mode", "clone")
    eval_method = cfg.get("eval_method", "agent")
    return build_graph(process_mode=process_mode, eval_method=eval_method)
