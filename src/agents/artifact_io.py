"""Per-agent inference artifact helpers.

Writes a consistent on-disk layout for each agent call::

    <call_dir>/
      input.txt
      output.txt
      artifacts/<name>
      metadata.json
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def file_path_to_slug(file_path: str | None) -> str:
    """Convert a conflicted file path into a filesystem-safe slug."""
    if not file_path:
        return "nofil"
    return str(file_path).replace("/", "_").replace("\\", "_")


def safe_slug(value: Any, *, max_len: int = 80) -> str:
    """Sanitize an arbitrary value for use in a directory name."""
    text = re.sub(r"[^\w.\-]+", "_", str(value if value is not None else "unknown"))
    return (text[:max_len] or "unknown").strip("_") or "unknown"


def get_artifact_root(state: Mapping[str, Any]) -> Path | None:
    """Return the method-level artifact root from pipeline state, if set."""
    raw = state.get("artifact_root")
    if not raw:
        return None
    try:
        return Path(str(raw))
    except Exception:
        return None


def agent_call_dir(
    artifact_root: Path | str | None,
    *,
    agent: str,
    file_slug: str | None = None,
    call_id: str | None = None,
) -> Path | None:
    """Build ``<root>/[file_slug/]<agent>/[call_id/]``.

    Scenario-level agents (e.g. analyzer, planner) omit ``file_slug``.
    """
    if artifact_root is None:
        return None
    root = Path(artifact_root)
    parts: list[str] = []
    if file_slug:
        parts.append(file_slug)
    parts.append(agent)
    if call_id:
        parts.append(str(call_id))
    return root.joinpath(*parts)


def write_agent_call(
    call_dir: Path | str | None,
    *,
    input_text: str = "",
    output_text: str = "",
    artifacts: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist one agent call's input, output, artifacts, and metadata JSON.

    Best-effort: failures are logged and never raised to the caller.
    """
    if call_dir is None:
        return None
    try:
        dest = Path(call_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "input.txt").write_text(input_text or "", encoding="utf-8")
        (dest / "output.txt").write_text(output_text or "", encoding="utf-8")

        art_dir = dest / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (artifacts or {}).items():
            safe_name = safe_slug(name, max_len=120)
            if not safe_name:
                continue
            (art_dir / safe_name).write_text(content if content is not None else "", encoding="utf-8")

        meta = dict(metadata or {})
        meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        (dest / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return dest
    except Exception as exc:
        logger.warning("Failed to write agent artifacts under %s: %s", call_dir, exc)
        return None


def write_final_artifacts(
    artifact_root: Path | str | None,
    *,
    file_path: str,
    resolved_text: str,
    final_diff: str = "",
) -> Path | None:
    """Write ``<root>/<file_slug>/final/{resolved.txt,final_diff.txt}``."""
    if artifact_root is None:
        return None
    try:
        dest = Path(artifact_root) / file_path_to_slug(file_path) / "final"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "resolved.txt").write_text(resolved_text or "", encoding="utf-8")
        (dest / "final_diff.txt").write_text(final_diff or "", encoding="utf-8")
        return dest
    except Exception as exc:
        logger.warning("Failed to write final artifacts for %s: %s", file_path, exc)
        return None


def base_metadata(
    *,
    agent: str,
    node: str,
    state: Mapping[str, Any],
    file_path: str | None = None,
    call_id: str | None = None,
    llm_used: bool = True,
    elapsed_s: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard metadata.json payload (tokens/costs filled later)."""
    meta: dict[str, Any] = {
        "agent": agent,
        "node": node,
        "model_name": state.get("model_name"),
        "elapsed_s": elapsed_s,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_in": None,
        "cost_out": None,
        "total_cost": None,
        "usage_from_api": None,
        "scenario_id": state.get("scenario_id"),
        "eval_method": state.get("eval_method"),
        "file_path": file_path,
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_used": llm_used,
    }
    if extra:
        meta.update(dict(extra))
    return meta


__all__ = [
    "agent_call_dir",
    "base_metadata",
    "file_path_to_slug",
    "get_artifact_root",
    "safe_slug",
    "write_agent_call",
    "write_final_artifacts",
]
