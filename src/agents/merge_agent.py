from __future__ import annotations

"""Lightweight LangChain-powered merge-conflict resolver.

This module exposes :func:`resolve_conflict_agent_node`, a LangGraph-compatible
stateful callable that can be plugged directly into the merge-pipeline graph.

Keeping agent logic in its own module isolates any heavyweight dependencies
(LangChain, OpenAI) from the core pipeline code and allows future swapping of
LLM providers or prompt engineering iterations without touching graph wiring.
"""
from pathlib import Path
from typing import Any, Dict
import os
import logging
import tiktoken
from dotenv import load_dotenv
# Updated LangChain imports per latest documentation (PromptTemplate now in langchain_core.prompts)
from langchain_core.prompts import PromptTemplate
# Build pipelines via the new Runnable interfaces rather than deprecated *LLMChain*.
# No additional import needed because the `|` operator constructs a RunnableSequence.
load_dotenv()

# Prefer the dedicated langchain_openai package for ChatOpenAI; fall back to
# community stub if the import fails (e.g. user hasn't installed it yet).
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover – best-effort fallback
    from langchain_community.chat_models import ChatOpenAI  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt & LLM setup
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "merge_prompt.txt"
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

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")

# ---------------------------------------------------------------------------
# Tokenizer setup (tiktoken)
# ---------------------------------------------------------------------------

import tiktoken

_CUSTOM_ENCODING_MAP = {
    "gpt-4.1-nano-2025-04-14": "o200k_base",  # Manual mapping per documentation
}

try:
    _ENCODER = tiktoken.encoding_for_model(_DEFAULT_MODEL)
except KeyError:
    _ENCODER = tiktoken.get_encoding(_CUSTOM_ENCODING_MAP.get(_DEFAULT_MODEL, "cl100k_base"))

# Helper for token counts
def _count_tokens(text: str) -> int:  # noqa: D401
    return len(_ENCODER.encode(text))

# Attempt to construct ChatOpenAI only if an API key is present; otherwise we
# will fall back to a simple parent-A passthrough.
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(_OPENAI_API_KEY)
if _OPENAI_API_KEY:
    _llm = ChatOpenAI(api_key=_OPENAI_API_KEY, model=_DEFAULT_MODEL, temperature=0)  # type: ignore[call-arg]
    _prompt = PromptTemplate.from_template(_PROMPT_STR)
    # The modern pattern chains the prompt directly into the LLM using the
    # pipe (|) operator which returns a RunnableSequence.
    _merge_runnable = _prompt | _llm
else:
    logger.warning("OPENAI_API_KEY not found – merge agent will default to Parent-A contents.")
    _llm = None  # type: ignore
    _merge_runnable = None  # type: ignore

FileContents = Dict[str, str]

# ---------------------------------------------------------------------------
# Public LangGraph-node callable
# ---------------------------------------------------------------------------

def resolve_conflict_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """LangGraph node that merges conflicting files via a single-turn LLM prompt."""

    parent_a: FileContents = state["parent_a_contents"]
    parent_b: FileContents = state["parent_b_contents"]

    # If no LLM credentials, fall back to Parent-A versions.
    if _merge_runnable is None:
        print("[merge_agent] OPENAI_API_KEY not set – falling back to baseline resolution (parent-A).")
        state["resolved_contents"] = parent_a
        state["status"] = "resolved_agent_stub"
        return state

    file_paths = list(parent_a.keys() | parent_b.keys())
    print(f"[merge_agent] Resolving {len(file_paths)} conflicted file(s) via LLM…")

    resolved: FileContents = {}

    for idx, path in enumerate(file_paths, start=1):
        print(f"[merge_agent] ({idx}/{len(file_paths)}) Merging {path}…")

        original_text = state["ancestor_contents"].get(path, "")
        diff_a_text = state["diffs_a"].get(path, "")
        diff_b_text = state["diffs_b"].get(path, "")

        pre_token_usage = {
            "system_prompt": _count_tokens(_PROMPT_STR),
            "original": _count_tokens(original_text),
            "diff_a": _count_tokens(diff_a_text),
            "diff_b": _count_tokens(diff_b_text),
        }

        print(f"[merge_agent]     → pre-token counts: {pre_token_usage}")

        result = _merge_runnable.invoke({
            "file_path": path,
            "original": original_text,
            "diff_a": diff_a_text,
            "diff_b": diff_b_text,
        })

        # The runnable may return a ChatMessage or plain string; handle both.
        if hasattr(result, "content"):
            merged_content: str = result.content  # type: ignore[attr-defined]
        else:
            merged_content = str(result)

        merged_clean = merged_content.strip("\n")
        resolved[path] = merged_clean

        # Token accounting
        token_usage = {
            **pre_token_usage,
            "output": _count_tokens(merged_clean),
        }
        # Store per-path token stats
        state.setdefault("token_counts", {})[path] = token_usage

        print(f"[merge_agent]     → tokens: {token_usage}")

    state["resolved_contents"] = resolved
    state["status"] = "resolved_agent"
    print("[merge_agent] Resolution complete.")
    return state 