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
from dotenv import load_dotenv

# We intentionally import tiktoken and langchain lazily (inside try/except)
import tiktoken  # type: ignore

# Updated import location per latest LangChain split
from langchain_core.prompts import PromptTemplate
from ..llm_base import get_backend, count_tokens

load_dotenv()

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover – fallback for minimal installs
    from langchain_community.chat_models import ChatOpenAI  # type: ignore

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

    llm = ChatOpenAI(api_key=_OPENAI_API_KEY, model=model_name, temperature=0)  # type: ignore[call-arg]
    prompt = PromptTemplate.from_template(_PROMPT_STR)
    runnable = prompt | llm
    encoder = _get_encoder(model_name)
    _RUNNABLE_CACHE[model_name] = (encoder, runnable)
    return _RUNNABLE_CACHE[model_name]


def _count_tokens(encoder, text: str) -> int:  # noqa: D401
    return len(encoder.encode(text))

# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------

FileContents = Dict[str, str]

# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------

def resolve_conflict_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Merge conflicted files via a single-turn LLM prompt (or fallback)."""

    parent_a: FileContents = state["parent_a_contents"]
    parent_b: FileContents = state["parent_b_contents"]

    model_name: str = state.get("model_name") or _DEFAULT_MODEL

    encoder, llm_backend = get_backend(model_name)

    if llm_backend is not None:
        from langchain_core.prompts import PromptTemplate as _PT
        runnable = _PT.from_template(_PROMPT_STR) | llm_backend
    else:
        runnable = None

    # No credentials or failed initialisation → naive fallback.
    if runnable is None:
        state["resolved_contents"] = parent_a
        state["status"] = "resolved_agent_stub"
        return state

    file_paths = list(parent_a.keys() | parent_b.keys())
    resolved: FileContents = {}

    for path in file_paths:
        original_text = state["ancestor_contents"].get(path, "")
        diff_a_text = state["diffs_a"].get(path, "")
        diff_b_text = state["diffs_b"].get(path, "")

        pre_token_usage = {
            "system_prompt": count_tokens(encoder, _PROMPT_STR),
            "original": count_tokens(encoder, original_text),
            "diff_a": count_tokens(encoder, diff_a_text),
            "diff_b": count_tokens(encoder, diff_b_text),
        }

        result = runnable.invoke(  # type: ignore[attr-defined]
            {
                "file_path": path,
                "original": original_text,
                "diff_a": diff_a_text,
                "diff_b": diff_b_text,
            }
        )

        merged_content = result.content if hasattr(result, "content") else str(result)
        merged_clean = merged_content.strip("\n")
        resolved[path] = merged_clean

        token_usage = {**pre_token_usage, "output": count_tokens(encoder, merged_clean)}
        state.setdefault("token_counts", {})[path] = token_usage

    state["resolved_contents"] = resolved
    state["status"] = "resolved_agent"
    return state
