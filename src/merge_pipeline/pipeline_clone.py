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
import os

from git import Repo, GitCommandError
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

# Agent resolver node
from ..agents.simple_agent import resolve_conflict_agent_node
from ..agents.base_agent import resolve_conflict_base_node
from ..agents.multi_agent import resolve_conflict_multi_agent_node
from ..dataset.loader import DATA_PATH, load_benchmark
from ..eval.exact_match import per_file as em_per_file, overall as em_overall
from ..eval.bleu import per_file as bleu_per_file, overall as bleu_overall
from ..eval.rouge_l import per_file as rouge_per_file, overall as rouge_overall

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
    if dest.exists():
        return Repo(dest)

    logger.info("Cloning %s → %s", repo_url, dest)

    try:
        return Repo.clone_from(repo_url, dest)
    except GitCommandError as exc:
        # Windows path length issues – retry with longpaths enabled
        if "Filename too long" in str(exc):
            logger.warning("Encountered 'Filename too long'. Retrying with core.longpaths=true …")
            env = os.environ.copy()
            env["GIT_CONFIG_PARAMETERS"] = "core.longpaths=true"
            return Repo.clone_from(repo_url, dest, env=env)
        raise


def _read_files_at_commit(repo: Repo, commit_sha: str, paths: List[str]) -> FileContents:
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
    """Return a similarity ratio between *a* and *b* using difflib."""

    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# LangGraph nodes (stateless callables)
# ---------------------------------------------------------------------------

def load_sample_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Load the CSV row given a *scenario_id* in *state*."""

    df = load_benchmark()
    raw_id = state["scenario_id"]
    row_series = None

    # Attempt to treat raw_id as a numeric index first.
    try:
        # This will work for integer-like strings ("1505") or actual ints.
        numeric_id = int(raw_id)
        row_series = df.loc[numeric_id]
    except (ValueError, TypeError, KeyError):
        # ValueError/TypeError if raw_id isn't integer-like.
        # KeyError if df.loc[numeric_id] fails to find the index.
        pass

    # If numeric lookup failed, try matching against the 'id' column (slug).
    if row_series is None:
        match = df.loc[df["id"] == raw_id]
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
        except GitCommandError:
            pass  # Already present

    # Find merge base and read file contents
    merge_base_commit = repo.merge_base(parents[0], parents[1])[0]
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
                a_lines, p_a_lines, fromfile=f"ancestor/{path}", tofile=f"parent_a/{path}"
            )
        )
        diffs_b[path] = "".join(
            difflib.unified_diff(
                a_lines, p_b_lines, fromfile=f"ancestor/{path}", tofile=f"parent_b/{path}"
            )
        )

    return {
        **state,
        "repo_path": repo.working_dir,
        "ancestor_contents": ancestor_contents,
        "parent_a_contents": parent_a_contents,
        "parent_b_contents": parent_b_contents,
        "diffs_a": diffs_a,
        "diffs_b": diffs_b,
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

    ratios = {path: _diff_ratio(pred_contents.get(path, ""), truth) for path, truth in truth_contents.items()}

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
    """

    sg = StateGraph(dict)

    sg.add_node("load_sample", load_sample_node)
    sg.add_node("prepare_context", prepare_context_node)

    if eval_method == "agent":
        resolver_node_name = "resolve_agent"
        sg.add_node(resolver_node_name, resolve_conflict_agent_node)
    elif eval_method == "base":
        resolver_node_name = "resolve_base"
        sg.add_node(resolver_node_name, resolve_conflict_base_node)
    elif eval_method == "multi":
        resolver_node_name = "resolve_multi"
        sg.add_node(resolver_node_name, resolve_conflict_multi_agent_node)
    else:
        raise ValueError(f"Unknown eval_method {eval_method!r}; choose 'agent', 'base', or 'multi'.")

    sg.add_node("evaluate", evaluate_node)

    sg.set_entry_point("load_sample")

    sg.add_edge("load_sample", "prepare_context")
    sg.add_edge("prepare_context", resolver_node_name)
    sg.add_edge(resolver_node_name, "evaluate")
    sg.add_edge("evaluate", END)

    return sg.compile() 