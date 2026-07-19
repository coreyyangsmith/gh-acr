"""Persistent on-disk cache of prepared merge-conflict scenario context.

Caches ancestor/parent contents, diffs, commit messages, and ground truth so
future model runs can skip clone+prepare when the clone and cache already exist.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_VERSION = 1

_CONTENT_KEYS = (
    "ancestor_contents",
    "parent_a_contents",
    "parent_b_contents",
    "diffs_a",
    "diffs_b",
    "truth_contents",
    "diffs_truth",
)

_FILE_SUFFIX = {
    "ancestor_contents": "ancestor.txt",
    "parent_a_contents": "a.txt",
    "parent_b_contents": "b.txt",
    "diffs_a": "a.diff",
    "diffs_b": "b.diff",
    "truth_contents": "truth.txt",
    "diffs_truth": "truth.diff",
}


def context_cache_root() -> Path:
    """Return the root directory for scenario context caches."""
    env = (os.getenv("GHACR_CONTEXT_CACHE_DIR") or "").strip()
    if env:
        return Path(env)
    return Path.cwd() / "data" / "context_cache"


def safe_scenario_slug(scenario_id: Any, *, max_len: int = 160) -> str:
    """Filesystem-safe directory name for a scenario id."""
    text = str(scenario_id if scenario_id is not None else "unknown")
    text = text.replace("/", "__").replace("\\", "__").replace(":", "_")
    text = re.sub(r"[^\w.\-]+", "_", text)
    return (text[:max_len] or "unknown").strip("_") or "unknown"


def _cache_dir(scenario_id: Any) -> Path:
    return context_cache_root() / safe_scenario_slug(scenario_id)


def _file_slug(file_path: str) -> str:
    return str(file_path).replace("/", "_").replace("\\", "_")


def _scenario_fingerprint(sample_row: dict[str, Any]) -> dict[str, Any]:
    scenario = sample_row.get("scenario_json") or {}
    if isinstance(scenario, str):
        scenario = json.loads(scenario)
    return {
        "scenario_id": str(sample_row.get("id", "")),
        "repo": str(sample_row.get("name", "")),
        "merge_commit_hash": str(scenario.get("merge_commit_hash", "")),
        "parents": list(scenario.get("parents") or []),
        "files_in_merge_conflict": list(scenario.get("files_in_merge_conflict") or []),
        "cache_version": CACHE_VERSION,
    }


def _meta_matches(meta: dict[str, Any], sample_row: dict[str, Any]) -> bool:
    expected = _scenario_fingerprint(sample_row)
    if int(meta.get("cache_version", -1)) != CACHE_VERSION:
        return False
    for key in ("merge_commit_hash", "parents", "files_in_merge_conflict"):
        if meta.get(key) != expected.get(key):
            return False
    return True


def load_context(
    scenario_id: Any,
    sample_row: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load a prepared context from disk, or ``None`` on miss/mismatch."""
    root = _cache_dir(scenario_id)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if sample_row is not None and not _meta_matches(meta, sample_row):
        logger.info(
            "[context_cache] Invalidating cache for %s (fingerprint mismatch)",
            scenario_id,
        )
        return None

    files = list(meta.get("files_in_merge_conflict") or [])
    state: dict[str, Any] = {
        "scenario_id": scenario_id,
        "status": "context_prepared",
        "from_context_cache": True,
    }
    for key in _CONTENT_KEYS:
        state[key] = {}

    files_dir = root / "files"
    for path in files:
        slug = _file_slug(path)
        file_dir = files_dir / slug
        for key, suffix in _FILE_SUFFIX.items():
            fp = file_dir / suffix
            if fp.is_file():
                try:
                    state[key][path] = fp.read_text(encoding="utf-8")
                except OSError:
                    state[key][path] = ""
            else:
                state[key][path] = ""

    for name in ("commit_messages_a", "commit_messages_b"):
        fp = root / f"{name}.txt"
        state[name] = fp.read_text(encoding="utf-8") if fp.is_file() else ""

    sample_path = root / "sample_row.json"
    if sample_path.is_file():
        try:
            state["sample_row"] = json.loads(sample_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if sample_row is not None:
                state["sample_row"] = sample_row
            else:
                return None
    elif sample_row is not None:
        state["sample_row"] = sample_row
    else:
        return None

    repo_path = meta.get("repo_path")
    if repo_path and Path(repo_path).exists():
        state["repo_path"] = repo_path
    else:
        # Best-effort: reconstruct expected clone path without requiring git
        try:
            from src.merge_pipeline.pipeline_clone import _checkout_root

            name = str((state.get("sample_row") or {}).get("name", "")).replace("/", "___")
            candidate = _checkout_root() / name if name else None
            if candidate is not None and candidate.exists():
                state["repo_path"] = str(candidate)
        except Exception:
            pass

    # Require core content maps to be present for all conflict files
    required = ("ancestor_contents", "parent_a_contents", "parent_b_contents", "truth_contents")
    for key in required:
        if set(files) - set((state.get(key) or {}).keys()):
            logger.info("[context_cache] Incomplete cache for %s (%s)", scenario_id, key)
            return None

    logger.info("[context_cache] HIT scenario=%s path=%s", scenario_id, root)
    return state


def save_context(scenario_id: Any, state: dict[str, Any]) -> Path:
    """Persist prepared context to disk; returns the cache directory."""
    sample_row = state.get("sample_row") or {}
    meta = _scenario_fingerprint(sample_row)
    meta["scenario_id"] = str(scenario_id)
    if state.get("repo_path"):
        meta["repo_path"] = str(state["repo_path"])

    root = _cache_dir(scenario_id)
    files_dir = root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    files = list(meta.get("files_in_merge_conflict") or [])
    for path in files:
        slug = _file_slug(path)
        file_dir = files_dir / slug
        file_dir.mkdir(parents=True, exist_ok=True)
        for key, suffix in _FILE_SUFFIX.items():
            content = (state.get(key) or {}).get(path, "")
            (file_dir / suffix).write_text(content if content is not None else "", encoding="utf-8")

    for name in ("commit_messages_a", "commit_messages_b"):
        (root / f"{name}.txt").write_text(str(state.get(name, "") or ""), encoding="utf-8")

    # Drop huge nested objects that aren't needed; keep scenario_json + ids
    sample_out = dict(sample_row)
    (root / "sample_row.json").write_text(
        json.dumps(sample_out, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[context_cache] SAVED scenario=%s path=%s", scenario_id, root)
    return root


def ensure_prepared(
    scenario_id: Any,
    sample_row: dict[str, Any],
) -> dict[str, Any]:
    """Return prepared scenario state from cache or by cloning + preparing.

    On cache miss: clone (or reuse), run prepare_context, load ground truth,
    persist the cache, and return the state dict.
    """
    import difflib

    from git import Repo

    from src.merge_pipeline.pipeline_clone import (
        _checkout_root,
        _clone_repo,
        _read_files_at_commit,
        prepare_context_node,
    )

    # Ensure sample_row has id + df_index for downstream CSV rows
    sample = dict(sample_row)
    if "id" not in sample:
        sample["id"] = scenario_id
    if "df_index" not in sample:
        sample["df_index"] = sample.get("id", scenario_id)
    # scenario_json may be a string in some CSV loads
    sj = sample.get("scenario_json")
    if isinstance(sj, str):
        sample["scenario_json"] = json.loads(sj)

    cached = load_context(scenario_id, sample)
    if cached is not None:
        cached["sample_row"] = sample
        cached["scenario_id"] = scenario_id
        return cached

    logger.info("[context_cache] MISS scenario=%s — preparing", scenario_id)
    # Ensure clone exists first (reuse if present)
    repo = _clone_repo(sample, checkout_dir=_checkout_root())
    state: dict[str, Any] = {
        "scenario_id": scenario_id,
        "sample_row": sample,
        "repo_path": repo.working_dir,
        "status": "sample_loaded",
    }
    state = prepare_context_node(state)

    scenario = sample["scenario_json"]
    files = scenario["files_in_merge_conflict"]
    repo = Repo(state["repo_path"])
    truth_contents = _read_files_at_commit(
        repo, scenario["merge_commit_hash"], files
    )
    ancestor_contents = state.get("ancestor_contents", {}) or {}
    diffs_truth: dict[str, str] = {}
    for path in files:
        anc_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
        truth_lines = truth_contents.get(path, "").splitlines(keepends=True)
        diffs_truth[path] = "".join(
            difflib.unified_diff(
                anc_lines,
                truth_lines,
                fromfile=f"ancestor/{path}",
                tofile=f"ground_truth/{path}",
            )
        )
    state["truth_contents"] = truth_contents
    state["diffs_truth"] = diffs_truth
    save_context(scenario_id, state)
    return state


__all__ = [
    "CACHE_VERSION",
    "context_cache_root",
    "safe_scenario_slug",
    "load_context",
    "save_context",
    "ensure_prepared",
]
