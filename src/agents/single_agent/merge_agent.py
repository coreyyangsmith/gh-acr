"""LangChain-powered merge-conflict resolver (simple agent).

This is a verbatim copy of the original `agents/merge_agent.py` moved into the
`agents.simple_agent` namespace during the repository refactor.  Only one path
adjustment is made for the prompt template (parents[2] instead of parents[1]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import logging
import time
from dotenv import load_dotenv

# We intentionally import tiktoken and langchain lazily (inside try/except)
import tiktoken  # type: ignore

# Updated import location per latest LangChain split
from ..llm_base import get_backend, count_tokens
from ..resilient_invoke import resilient_invoke
from ..artifact_io import (
    agent_call_dir,
    base_metadata,
    file_path_to_slug,
    get_artifact_root,
    write_agent_call,
    write_final_artifacts,
)
from ..prompt_budget import EvidenceBlock, fit_variable_blocks
from ...utils.degradation import record_degradation
from ...utils.run_progress import set_stage

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt & LLM setup
# ---------------------------------------------------------------------------

# Load prompt from prompts/agent/
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "agent" / "merge_prompt.txt"
if not _PROMPT_PATH.exists():
    logger.warning("Prompt template missing at %s – falling back to default prompt string.", _PROMPT_PATH)
    _DEFAULT_PROMPT_STR = (
        "You are a helpful assistant specialized in resolving Git merge conflicts for source code files.\n\n"
        "Given two versions of the same file from Parent A and Parent B, produce a single merged version that incorporates the correct changes from both parents and resolves any conflicting edits.\n\n"
        "Guidelines:\n"
        "1. If the parents modified different parts of the file, include both changes.\n"
        "2. If both parents changed the very same lines in incompatible ways, decide on the best resolution (prefer readability and correctness).\n"
        "3. Preserve the original formatting and indentation style.\n"
        "4. Output ONLY the fully merged file content – no explanations, comments, or conflict markers.\n\n"
        "Context:\nFile path: {file_path}\n\n"
        "--- Parent A version ---\n{version_a}\n--- End Parent A ---\n\n"
        "--- Parent B version ---\n{version_b}\n--- End Parent B ---\n\n"
        "[Your answer below – merged file content starts on the next line]"
    )
    _PROMPT_STR = _DEFAULT_PROMPT_STR
else:
    _PROMPT_STR = _PROMPT_PATH.read_text(encoding="utf-8")

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4.1-nano-2025-04-14")

# Cache for model→(encoder, runnable)
_RUNNABLE_CACHE: dict[str, tuple[tiktoken.Encoding, Any]] = {}
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _get_encoder(model_name: str):  # noqa: D401
    """Return a tiktoken encoder for *model_name*, with simple fallback mapping."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:  # pragma: no cover – unknown model
        return tiktoken.get_encoding("cl100k_base")


def _get_runnable(model_name: str):  # noqa: D401
    """Return (encoder, prompt|LLM runnable) for *model_name* (cache-aware)."""
    if model_name in _RUNNABLE_CACHE:
        return _RUNNABLE_CACHE[model_name]

    if _OPENAI_API_KEY is None:
        _RUNNABLE_CACHE[model_name] = (None, None)  # type: ignore
        return _RUNNABLE_CACHE[model_name]

    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError:  # pragma: no cover – fallback for minimal installs
        from langchain_community.chat_models import ChatOpenAI  # type: ignore

    # Omit temperature for GPT-5 variants which do not support it
    if model_name.split("/", 1)[-1].startswith("gpt-5"):
        llm = ChatOpenAI(api_key=_OPENAI_API_KEY, model=model_name)  # type: ignore[call-arg]
    else:
        llm = ChatOpenAI(api_key=_OPENAI_API_KEY, model=model_name, temperature=0)  # type: ignore[call-arg]
    runnable = llm
    encoder = _get_encoder(model_name)
    _RUNNABLE_CACHE[model_name] = (encoder, runnable)
    return _RUNNABLE_CACHE[model_name]


def _count_tokens(encoder, text: str) -> int:  # noqa: D401
    return len(encoder.encode(text))

# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------

FileContents = Dict[str, str]


def resolve_conflict_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Merge conflicted files via a single-turn LLM prompt (or fallback)."""
    t0 = time.perf_counter()
    scenario_id = state.get("scenario_id", "unknown")
    parent_a: FileContents = state["parent_a_contents"]
    parent_b: FileContents = state["parent_b_contents"]

    model_name: str = state.get("model_name") or _DEFAULT_MODEL
    set_stage("resolve", detail=f"model={model_name}")
    logger.info(
        "[resolve_conflict_agent] scenario=%s Starting with model: %s",
        scenario_id,
        model_name,
    )

    encoder, llm_backend = get_backend(model_name)

    if llm_backend is not None:
        runnable = llm_backend
    else:
        runnable = None

    # No credentials or failed initialisation → naive fallback.
    if runnable is None:
        logger.warning("No LLM backend available, using naive fallback (parent_a)")
        record_degradation(
            "llm_unavailable_heuristic",
            "single-agent used parent_a stub (no LLM)",
            node="resolve_conflict_agent",
        )
        state["resolved_contents"] = parent_a
        state["status"] = "resolved_agent_stub"
        artifact_root = get_artifact_root(state)
        for path, content in parent_a.items():
            write_agent_call(
                agent_call_dir(
                    artifact_root,
                    agent="agent",
                    file_slug=file_path_to_slug(path),
                ),
                input_text="",
                output_text=content or "",
                artifacts={},
                metadata=base_metadata(
                    agent="agent",
                    node="resolve_conflict_agent",
                    state=state,
                    file_path=path,
                    llm_used=False,
                    extra={"reason": "no_llm"},
                ),
            )
            write_final_artifacts(
                artifact_root,
                file_path=path,
                resolved_text=content or "",
                final_diff="",
            )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[resolve_conflict_agent] scenario=%s Completed in %.3fs (status=%s)",
            scenario_id,
            elapsed,
            state["status"],
        )
        return state

    file_paths = list(parent_a.keys() | parent_b.keys())
    logger.info("Processing %d files for conflict resolution", len(file_paths))
    resolved: FileContents = {}
    artifact_root = get_artifact_root(state)

    for path in file_paths:
        logger.debug("Processing file: %s", path)
        original_text = state["ancestor_contents"].get(path, "")
        diff_a_text = state["diffs_a"].get(path, "")
        diff_b_text = state["diffs_b"].get(path, "")

        pre_token_usage = {
            "system_prompt": count_tokens(encoder, _PROMPT_STR),
            "original": count_tokens(encoder, original_text),
            "diff_a": count_tokens(encoder, diff_a_text),
            "diff_b": count_tokens(encoder, diff_b_text),
        }
        logger.debug("Token usage for %s: %s", path, pre_token_usage)

        # Structure-aware fit: keep instructions + file path; clip evidence.
        fit = fit_variable_blocks(
            template=_PROMPT_STR,
            render="format",
            fixed_variables={"file_path": path},
            blocks=[
                EvidenceBlock(
                    block_id="original",
                    text=original_text,
                    kind="context",
                    file_path=path,
                    priority=20,
                ),
                EvidenceBlock(
                    block_id="diff_a",
                    text=diff_a_text,
                    side="A",
                    kind="diff",
                    file_path=path,
                    priority=30,
                ),
                EvidenceBlock(
                    block_id="diff_b",
                    text=diff_b_text,
                    side="B",
                    kind="diff",
                    file_path=path,
                    priority=30,
                ),
            ],
            encoder=encoder,
            model_name=model_name,
            node="resolve_conflict_agent",
            file_path=path,
        )
        prompt_text = fit.prompt

        call_dir = agent_call_dir(
            artifact_root,
            agent="agent",
            file_slug=file_path_to_slug(path),
        )
        supporting: Dict[str, str] = {
            "original.txt": original_text,
            "a.diff": diff_a_text,
            "b.diff": diff_b_text,
        }
        if fit.was_clipped:
            supporting["truncation_report.json"] = fit.artifact_json()

        logger.debug("Invoking LLM for file: %s", path)
        result = resilient_invoke(
            runnable,
            prompt_text,
            context={
                "scenario_id": state.get("scenario_id"),
                "eval_method": state.get("eval_method", "agent"),
                "node": "resolve_conflict_agent",
                "agent": "agent",
                "file_path": path,
                "model_name": model_name,
            },
            artifact_dir=call_dir,
            artifacts=supporting,
        )

        merged_content = result.content if hasattr(result, "content") else str(result)
        merged_clean = merged_content.strip("\n")
        resolved[path] = merged_clean
        logger.debug("Successfully resolved file: %s (output length: %d chars)", path, len(merged_clean))

        token_usage = {**pre_token_usage, "output": count_tokens(encoder, merged_clean)}
        state.setdefault("token_counts", {})[path] = token_usage
        write_final_artifacts(
            artifact_root,
            file_path=path,
            resolved_text=merged_clean,
            final_diff="",
        )

    logger.info("Conflict resolution completed for %d files", len(resolved))
    state["resolved_contents"] = resolved
    state["status"] = "resolved_agent"
    elapsed = time.perf_counter() - t0
    logger.info(
        "[resolve_conflict_agent] scenario=%s Completed in %.3fs (status=%s)",
        scenario_id,
        elapsed,
        state["status"],
    )
    return state
