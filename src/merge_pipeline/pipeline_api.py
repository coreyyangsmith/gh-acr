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

from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from ..dataset.loader import load_benchmark
from ..tools.github_api import GithubClient
from ..eval.exact_match import per_file as em_per_file, overall as em_overall
from ..eval.bleu import per_file as bleu_per_file, overall as bleu_overall
from ..eval.rouge_l import per_file as rouge_per_file, overall as rouge_overall

# Import the dedicated merge agent node
from ..agents.simple_agent import resolve_conflict_agent_node
from ..agents.base_agent import resolve_conflict_base_node
from ..agents.multi_agent import resolve_conflict_multi_agent_node

# Local logger for this module
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SampleRow = Dict[str, Any]
FileContents = Dict[str, str]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _read_files_via_api(client: GithubClient, owner: str, repo: str, commit_sha: str, paths: List[str]) -> FileContents:
    """Return the UTF-8 contents of *paths* at *commit_sha* using the GitHub API."""
    out: FileContents = {}
    for path in paths:
        try:
            out[path] = client.get_file_content(owner, repo, path, commit_sha)
        except Exception:
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

    try:
        numeric_id = int(raw_id)
        row_series = df.loc[numeric_id]
    except (ValueError, TypeError, KeyError):
        pass

    if row_series is None:
        match = df.loc[df["id"] == raw_id]
        if not match.empty:
            row_series = match.iloc[0]

    if row_series is None:
        raise ValueError(
            f"Scenario id {raw_id!r} not found in dataset (checked index and 'id' column)"
        )

    sample_dict = row_series.to_dict()
    sample_dict["df_index"] = row_series.name
    state["sample_row"] = sample_dict
    state["status"] = "sample_loaded"
    return state


def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Prepare all data needed for resolution and evaluation using the GitHub API."""
    client = GithubClient()
    sample = state["sample_row"]
    owner, repo_name = sample["name"].split("/")
    scenario = sample["scenario_json"]
    files = scenario["files_in_merge_conflict"]
    parents = scenario["parents"]

    merge_base_commit = client.get_merge_base(owner, repo_name, parents[0], parents[1])
    ancestor_sha = merge_base_commit["sha"]

    ancestor_contents = _read_files_via_api(client, owner, repo_name, ancestor_sha, files)
    parent_a_contents = _read_files_via_api(client, owner, repo_name, parents[0], files)
    parent_b_contents = _read_files_via_api(client, owner, repo_name, parents[1], files)

    diffs_a: Dict[str, str] = {}
    diffs_b: Dict[str, str] = {}
    for path in files:
        a_lines = ancestor_contents.get(path, "").splitlines(keepends=True)
        p_a_lines = parent_a_contents.get(path, "").splitlines(keepends=True)
        p_b_lines = parent_b_contents.get(path, "").splitlines(keepends=True)

        diffs_a[path] = "".join(
            difflib.unified_diff(a_lines, p_a_lines, fromfile=f"ancestor/{path}", tofile=f"parent_a/{path}")
        )
        diffs_b[path] = "".join(
            difflib.unified_diff(a_lines, p_b_lines, fromfile=f"ancestor/{path}", tofile=f"parent_b/{path}")
        )

    truth_contents = _read_files_via_api(client, owner, repo_name, scenario["merge_commit_hash"], files)

    return {
        **state,
        "ancestor_contents": ancestor_contents,
        "parent_a_contents": parent_a_contents,
        "parent_b_contents": parent_b_contents,
        "diffs_a": diffs_a,
        "diffs_b": diffs_b,
        "truth_contents": truth_contents,
        "status": "context_prepared",
    }


def resolve_conflict_stub_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Stub retained for backward-compatibility (falls back to Parent-A)."""
    state["resolved_contents"] = state["parent_a_contents"]
    state["status"] = "resolved_stub"
    return state


def evaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Compare stub *resolution* to the *ground truth* merge commit."""
    pred_contents: FileContents = state["resolved_contents"]
    truth_contents: FileContents = state["truth_contents"]
    
    em_results = em_per_file(pred_contents, truth_contents)

    ratios = {
        path: _diff_ratio(pred_contents.get(path, ""), truth_contents.get(path, ""))
        for path in truth_contents
    }

    state["evaluation"] = {
        "similarity": ratios,
        "exact_match": em_results,
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

def build_graph(eval_method: str = "agent") -> Pregel:  # noqa: D401 – builder function
    """Return a LangGraph *Pregel* application.

    Parameters
    ----------
    eval_method
        "agent" (default) – use the `resolve_conflict_agent_node` (LLM-based).
        "base"  – use the simple parent-A stub.
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
        raise ValueError(f"Unknown eval_method {eval_method!r}; choose 'agent' or 'base'.")

    sg.add_node("evaluate", evaluate_node)

    sg.set_entry_point("load_sample")

    sg.add_edge("load_sample", "prepare_context")
    sg.add_edge("prepare_context", resolver_node_name)
    sg.add_edge(resolver_node_name, "evaluate")
    sg.add_edge("evaluate", END)

    return sg.compile() 