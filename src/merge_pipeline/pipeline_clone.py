from __future__ import annotations

"""End-to-end pipeline for resolving Git merge conflicts using **LangGraph**.

Only the data loading and *ground-truth* comparison are implemented with full
accuracy – the LLM portion is **stubbed out** and simply selects the first
parent's version of each conflicted file.  This keeps the code lightweight
and free of proprietary model dependencies while still demonstrating the
complete flow.
"""

from pathlib import Path
import difflib
import logging
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
import os
import time
import shutil
import stat

from git import Repo, GitCommandError
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from ..dataset.loader import load_benchmark

# Agent resolver node
from ..agents.base_agent import (
    resolve_conflict_base_a_node,
    resolve_conflict_base_b_node,
)
from ..agents.agent import resolve_conflict_agent_node
from ..agents.agent2 import resolve_conflict_agent2_node
from ..agents.agent3 import resolve_conflict_agent3_node
from ..agents.agent4 import resolve_conflict_agent4_node
from ..agents.multi import resolve_conflict_multi_agent_node
from ..agents.bypass import resolve_conflict_bypass_multi_agent_node
from ..agents.bypass2 import resolve_conflict_bypass2_multi_agent_node
from ..agents.bypass3 import resolve_conflict_bypass3_multi_agent_node
from ..agents.dynamic import resolve_conflict_dynamic_agent_node
from ..agents.bypass_only import resolve_conflict_bypass_only_multi_agent_node
from ..agents.bypass4 import resolve_conflict_bypass4_multi_agent_node
from ..agents.bypass5 import resolve_conflict_bypass5_multi_agent_node
from ..agents.bypass6 import resolve_conflict_bypass6_multi_agent_node
from ..agents.bypass7 import resolve_conflict_bypass7_multi_agent_node
from ..agents.new_bypass import resolve_conflict_new_bypass_multi_agent_node
from ..agents.new_bypass2 import resolve_conflict_new_bypass2_multi_agent_node
from ..agents.new_bypass3 import resolve_conflict_new_bypass3_multi_agent_node
from ..agents.new_bypass4 import resolve_conflict_new_bypass4_multi_agent_node
from ..agents.new_bypass5 import resolve_conflict_new_bypass5_multi_agent_node
from ..agents.bypass8 import resolve_conflict_bypass8_multi_agent_node
from ..agents.bypass_only2 import resolve_conflict_bypass_only2_multi_agent_node

# Evaluation
from ..eval.exact_match import per_file as em_per_file, overall as em_overall
from ..eval.bleu import per_file as bleu_per_file, overall as bleu_overall
from ..eval.rouge_l import per_file as rouge_per_file, overall as rouge_overall
from rapidfuzz.distance import Levenshtein as RFLevenshtein  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SampleRow = Dict[str, Any]
FileContents = Dict[str, str]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _clone_repo(sample: SampleRow, checkout_dir: Path) -> Repo:
    """Clone *or* reuse the repository referenced by *sample*.

    The repo is fetched into **<checkout_dir>/<sample['name']>**.
    """
    checkout_dir.mkdir(exist_ok=True)
    repo_url = f"https://github.com/{sample['name']}.git"
    dest = checkout_dir / sample["name"].replace("/", "___")

    # Per-repo lock to avoid concurrent clones into the same destination
    locks_dir = checkout_dir / "_locks"
    locks_dir.mkdir(exist_ok=True)
    lock_path = locks_dir / (dest.name + ".lock")

    acquired = False
    for _ in range(240):  # ~60s timeout (240 * 0.25s)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.25)
    if not acquired:
        logger.warning("Timeout acquiring lock for %s; proceeding without lock", dest)

    def _force_remove_dir(
        path: Path, *, retries: int = 20, delay_s: float = 0.25
    ) -> bool:
        def _on_rm_error(func, p, exc_info):  # make read-only writable and retry
            try:
                os.chmod(p, stat.S_IWRITE)
            except Exception:
                pass
            try:
                func(p)
            except Exception:
                pass

        for _ in range(retries):
            try:
                shutil.rmtree(path, onerror=_on_rm_error)
            except FileNotFoundError:
                return True
            except Exception:
                time.sleep(delay_s)
            if not path.exists():
                return True
            time.sleep(delay_s)
        return not path.exists()

    try:
        if dest.exists():
            try:
                repo = Repo(dest)
                # Ensure origin URL matches expected; if not, reclone
                try:
                    origin_url = next(repo.remote("origin").urls)
                except Exception:
                    origin_url = ""
                if origin_url and origin_url.endswith(f"{sample['name']}.git"):
                    logger.info("Reusing existing clone at %s", dest)
                    return repo
                logger.warning(
                    "Existing directory at %s has unexpected origin; deleting and recloning",
                    dest,
                )
            except Exception:
                logger.warning(
                    "Existing directory at %s is not a valid git repo; deleting and recloning",
                    dest,
                )
            removed = _force_remove_dir(dest)
            if not removed and dest.exists():
                logger.error(
                    "Failed to remove invalid repo directory after retries: %s", dest
                )
                raise RuntimeError(f"Cannot remove stale repo directory: {dest}")

        logger.info("Cloning %s → %s", repo_url, dest)

        try:
            return Repo.clone_from(repo_url, dest)
        except GitCommandError as exc:
            msg = str(exc)
            # Windows path length issues – retry with longpaths enabled
            if "Filename too long" in msg:
                logger.warning(
                    "Encountered 'Filename too long'. Retrying with core.longpaths=true …"
                )
                env = os.environ.copy()
                env["GIT_CONFIG_PARAMETERS"] = "core.longpaths=true"
                return Repo.clone_from(repo_url, dest, env=env)
            # Handle race: destination appeared during clone
            if "already exists and is not an empty directory" in msg:
                try:
                    repo = Repo(dest)
                    logger.info(
                        "Detected concurrent clone completion; reusing %s", dest
                    )
                    return repo
                except Exception:
                    logger.warning(
                        "Destination exists but not a valid repo; removing and retrying: %s",
                        dest,
                    )
                    removed = _force_remove_dir(dest)
                    if not removed and dest.exists():
                        logger.error(
                            "Failed to remove directory before retry after multiple attempts: %s",
                            dest,
                        )
                        raise
                    # Second attempt wrapped to re-handle race conditions
                    try:
                        return Repo.clone_from(repo_url, dest)
                    except GitCommandError as exc2:
                        msg2 = str(exc2)
                        if "already exists and is not an empty directory" in msg2:
                            try:
                                repo = Repo(dest)
                                logger.info(
                                    "Concurrent clone finished during retry; reusing %s",
                                    dest,
                                )
                                return repo
                            except Exception:
                                logger.error(
                                    "Clone retry failed and repo still invalid at %s",
                                    dest,
                                )
                        raise
            raise
    finally:
        # Release lock
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            logger.warning("Failed to remove repo lock file: %s", lock_path)


def _read_files_at_commit(
    repo: Repo, commit_sha: str, paths: List[str]
) -> FileContents:
    """Return the UTF-8 contents of *paths* at *commit_sha*.

    Any binary files (decode error) are silently skipped.
    """

    commit = repo.commit(commit_sha)
    tree = commit.tree
    out: FileContents = {}
    for path in paths:
        try:
            blob = tree / path
            out[path] = blob.data_stream.read().decode("utf-8", errors="ignore")
        except KeyError:
            # File may have been deleted or renamed – ignore for this simple demo
            logger.warning("%s missing at commit %s", path, commit_sha)
    return out


def _diff_ratio(a: str, b: str) -> float:
    """Return normalized Levenshtein similarity in [0,1] between *a* and *b*."""

    try:
        return float(RFLevenshtein.normalized_similarity(a, b))
    except Exception:
        # Fallback to 1.0 if both empty, else 0.0 for degenerate cases
        if not a and not b:
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# LangGraph nodes (stateless callables)
# ---------------------------------------------------------------------------


def load_sample_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Load the CSV row given a *scenario_id* in *state*."""

    df = load_benchmark()
    raw_id = str(state["scenario_id"])  # ensure string for comparison
    row_series = None

    # Try index match first (handles numeric random indices in CSV)
    try:
        row_series = df.loc[int(raw_id)]
    except Exception:
        try:
            row_series = df.loc[raw_id]
        except Exception:
            row_series = None

    # Fallback: match against slug id column
    if row_series is None and "id" in df.columns:
        match = df.loc[df["id"].astype(str) == raw_id]
        if not match.empty:
            row_series = match.iloc[0]

    if row_series is None:
        raise ValueError(
            f"Scenario id {raw_id!r} not found in dataset (checked index and 'id' column)"
        )

    sample_dict = row_series.to_dict()
    sample_dict["df_index"] = row_series.name  # Add the numeric index to the dict

    state["sample_row"] = sample_dict
    state["status"] = "sample_loaded"
    return state


def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Prepare all data needed for resolution and evaluation.

    This node:
    1. Clones the repo.
    2. Finds the merge base.
    3. Reads file contents for ancestor, parent A, and parent B.
    4. Generates diffs between the ancestor and each parent.
    """
    sample = state["sample_row"]
    scenario = sample["scenario_json"]
    files = scenario["files_in_merge_conflict"]
    parents = scenario["parents"]

    repo = _clone_repo(sample, checkout_dir=Path.cwd() / "repos")

    # Ensure all necessary commits are available locally
    for sha in [scenario["merge_commit_hash"], *parents]:
        try:
            repo.git.fetch("origin", sha)
        except GitCommandError as exc:
            logger.warning("Fetch failed for %s: %s (continuing)", sha, exc)

    # Find merge base and read file contents
    try:
        merge_base_commit = repo.merge_base(parents[0], parents[1])[0]
    except Exception as exc:
        logger.exception(
            "Failed to compute merge-base for %s vs %s", parents[0], parents[1]
        )
        raise
    ancestor_contents = _read_files_at_commit(repo, merge_base_commit.hexsha, files)
    parent_a_contents = _read_files_at_commit(repo, parents[0], files)
    parent_b_contents = _read_files_at_commit(repo, parents[1], files)

    # Generate diffs
    diffs_a: Dict[str, str] = {}
    diffs_b: Dict[str, str] = {}
    for path in files:
        a_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
        p_a_lines = parent_a_contents.get(path, "").splitlines(keepends=True)
        p_b_lines = parent_b_contents.get(path, "").splitlines(keepends=True)

        diffs_a[path] = "".join(
            difflib.unified_diff(
                a_lines,
                p_a_lines,
                fromfile=f"ancestor/{path}",
                tofile=f"parent_a/{path}",
            )
        )
        diffs_b[path] = "".join(
            difflib.unified_diff(
                a_lines,
                p_b_lines,
                fromfile=f"ancestor/{path}",
                tofile=f"parent_b/{path}",
            )
        )

    # Collect commit messages for the commits contributing to each side (A/B).
    # We restrict to commits that touched the conflicted files for focus.
    commit_messages_a = ""
    commit_messages_b = ""
    try:
        commits_a = list(
            repo.iter_commits(
                f"{merge_base_commit.hexsha}..{parents[0]}", paths=files
            )
        )
        if commits_a:
            lines_a = []
            for c in reversed(commits_a):  # oldest → newest
                summary = getattr(c, "summary", "")
                message = getattr(c, "message", "") or ""
                lines_a.append(f"SHA: {c.hexsha}\nTitle: {summary}\n\n{message.strip()}")
            commit_messages_a = "\n\n-----\n\n".join(lines_a)
        else:
            # Fallback to the parent commit's own message
            commit_messages_a = (repo.commit(parents[0]).message or "").strip()
    except Exception:
        commit_messages_a = ""

    try:
        commits_b = list(
            repo.iter_commits(
                f"{merge_base_commit.hexsha}..{parents[1]}", paths=files
            )
        )
        if commits_b:
            lines_b = []
            for c in reversed(commits_b):  # oldest → newest
                summary = getattr(c, "summary", "")
                message = getattr(c, "message", "") or ""
                lines_b.append(f"SHA: {c.hexsha}\nTitle: {summary}\n\n{message.strip()}")
            commit_messages_b = "\n\n-----\n\n".join(lines_b)
        else:
            commit_messages_b = (repo.commit(parents[1]).message or "").strip()
    except Exception:
        commit_messages_b = ""

    return {
        **state,
        "repo_path": repo.working_dir,
        "ancestor_contents": ancestor_contents,
        "parent_a_contents": parent_a_contents,
        "parent_b_contents": parent_b_contents,
        "diffs_a": diffs_a,
        "diffs_b": diffs_b,
        "commit_messages_a": commit_messages_a,
        "commit_messages_b": commit_messages_b,
        "status": "context_prepared",
    }


def resolve_conflict_stub_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Stub LLM that *selects parent[0]* version for each conflicted file."""
    state["resolved_contents"] = state["parent_a_contents"]
    state["status"] = "resolved_stub"
    return state


def evaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Compare stub *resolution* to the *ground truth* merge commit."""

    sample = state["sample_row"]
    scenario = sample["scenario_json"]
    repo = Repo(state["repo_path"])

    truth_contents = _read_files_at_commit(
        repo, scenario["merge_commit_hash"], scenario["files_in_merge_conflict"]
    )
    pred_contents: FileContents = state["resolved_contents"]

    # -------------------------------------------------------------------
    # Compute diffs between the ancestor version and the ground-truth merge
    # result so that we can persist them to disk later (ground_truth.diff).
    # -------------------------------------------------------------------
    diffs_truth: Dict[str, str] = {}
    ancestor_contents: FileContents = state.get("ancestor_contents", {})
    for path in scenario["files_in_merge_conflict"]:
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

    ratios = {
        path: _diff_ratio(pred_contents.get(path, ""), truth)
        for path, truth in truth_contents.items()
    }

    # Store results back into state for downstream consumers (runner.py)
    state["truth_contents"] = truth_contents
    state["diffs_truth"] = diffs_truth
    state["evaluation"] = {
        "similarity": ratios,
        "exact_match": em_per_file(pred_contents, truth_contents),
        "overall_exact_match": em_overall(pred_contents, truth_contents),
        "bleu3": bleu_per_file(pred_contents, truth_contents),
        "overall_bleu3": bleu_overall(pred_contents, truth_contents),
        "rouge_l": rouge_per_file(pred_contents, truth_contents),
        "overall_rouge_l": rouge_overall(pred_contents, truth_contents),
    }
    state["status"] = "evaluated"
    return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(eval_method: str = "agent") -> Pregel:  # noqa: D401
    """Return a LangGraph *Pregel* application.

    Parameters
    ----------
    eval_method
        "agent" – use LLM-based resolver.
        "base"  – use parent-A stub.
        "bypass2"/"bypass3" – alternative bypass agent variants.
    """

    sg = StateGraph(dict)

    sg.add_node("load_sample", load_sample_node)
    sg.add_node("prepare_context", prepare_context_node)

    if eval_method == "agent":
        resolver_node_name = "resolve_agent"
        sg.add_node(resolver_node_name, resolve_conflict_agent_node)
    elif eval_method == "agent2":
        resolver_node_name = "resolve_agent2"
        sg.add_node(resolver_node_name, resolve_conflict_agent2_node)
    elif eval_method == "agent3":
        resolver_node_name = "resolve_agent3"
        sg.add_node(resolver_node_name, resolve_conflict_agent3_node)
    elif eval_method == "agent4":
        resolver_node_name = "resolve_agent4"
        sg.add_node(resolver_node_name, resolve_conflict_agent4_node)
    elif eval_method == "base_a":
        resolver_node_name = "resolve_base_a"
        sg.add_node(resolver_node_name, resolve_conflict_base_a_node)
    elif eval_method == "base_b":
        resolver_node_name = "resolve_base_b"
        sg.add_node(resolver_node_name, resolve_conflict_base_b_node)
    elif eval_method == "multi":
        resolver_node_name = "resolve_multi"
        sg.add_node(resolver_node_name, resolve_conflict_multi_agent_node)
    elif eval_method == "bypass":
        resolver_node_name = "resolve_bypass_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass_multi_agent_node)
    elif eval_method == "bypass2":
        resolver_node_name = "resolve_bypass2_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass2_multi_agent_node)
    elif eval_method == "bypass3":
        resolver_node_name = "resolve_bypass3_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass3_multi_agent_node)
    elif eval_method == "bypass4":
        resolver_node_name = "resolve_bypass4_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass4_multi_agent_node)
    elif eval_method == "bypass5":
        resolver_node_name = "resolve_bypass5_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass5_multi_agent_node)
    elif eval_method == "bypass6":
        resolver_node_name = "resolve_bypass6_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass6_multi_agent_node)
    elif eval_method == "bypass7":
        resolver_node_name = "resolve_bypass7_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass7_multi_agent_node)
    elif eval_method == "new_bypass":
        resolver_node_name = "resolve_new_bypass_multi"
        sg.add_node(resolver_node_name, resolve_conflict_new_bypass_multi_agent_node)
    elif eval_method == "new_bypass2":
        resolver_node_name = "resolve_new_bypass2_multi"
        sg.add_node(resolver_node_name, resolve_conflict_new_bypass2_multi_agent_node)
    elif eval_method == "new_bypass3":
        resolver_node_name = "resolve_new_bypass3_multi"
        sg.add_node(resolver_node_name, resolve_conflict_new_bypass3_multi_agent_node)
    elif eval_method == "new_bypass4":
        resolver_node_name = "resolve_new_bypass4_multi"
        sg.add_node(resolver_node_name, resolve_conflict_new_bypass4_multi_agent_node)
    elif eval_method == "new_bypass5":
        resolver_node_name = "resolve_new_bypass5_multi"
        sg.add_node(resolver_node_name, resolve_conflict_new_bypass5_multi_agent_node)
    elif eval_method == "bypass8":
        resolver_node_name = "resolve_bypass8_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass8_multi_agent_node)
    elif eval_method == "dynamic":
        resolver_node_name = "resolve_dynamic"
        sg.add_node(resolver_node_name, resolve_conflict_dynamic_agent_node)
    elif eval_method == "bypass_only":
        resolver_node_name = "resolve_bypass_only_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass_only_multi_agent_node)
    elif eval_method == "bypass_only2":
        resolver_node_name = "resolve_bypass_only2_multi"
        sg.add_node(resolver_node_name, resolve_conflict_bypass_only2_multi_agent_node)
    else:
        raise ValueError(
            f"Unknown eval_method {eval_method!r}; choose 'agent', 'base_a', 'base_b', 'multi', 'bypass', 'bypass2', 'bypass3', 'bypass4', 'bypass5', 'bypass6', 'bypass7', 'bypass8', 'new_bypass', 'new_bypass2', 'new_bypass3', 'new_bypass4', 'new_bypass5', 'bypass_only', 'bypass_only2', or 'dynamic'."
        )

    sg.add_node("evaluate", evaluate_node)

    sg.set_entry_point("load_sample")

    sg.add_edge("load_sample", "prepare_context")
    sg.add_edge("prepare_context", resolver_node_name)
    sg.add_edge(resolver_node_name, "evaluate")
    sg.add_edge("evaluate", END)

    return sg.compile()


def make_graph(config: RunnableConfig | None = None) -> Pregel:  # noqa: D401
    """LangGraph entrypoint: build a compiled app from ``config``.

    Recognised configurable keys:
    - eval_method: "agent" | "base_a" | "base_b" | "multi" | "bypass" | "bypass2" | "bypass3" | "bypass4" | "bypass5" | "bypass6" | "bypass7" | "bypass8" | "bypass_only" | "bypass_only2" (default: "agent")
    """
    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    eval_method = cfg.get("eval_method", "agent")
    return build_graph(eval_method=eval_method)
