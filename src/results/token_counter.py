"""Count tokens across prompt .txt files under `src/prompts` and export JSON.

This script groups files by their top-level folder within `src/prompts`, counts
tokens per file using `tiktoken` when available (falling back to a simple word
count), and writes a JSON summary. The output path can be overridden with the
`OUTPUT_JSON` environment variable.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List


logger = logging.getLogger(__name__)

try:  # Best-effort tiktoken import
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


def get_repo_root() -> str:
    """Return repository root assuming this file lives at src/results/."""
    # This file lives at src/results/token_counter.py → repo root is two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_prompts_root(repo_root: str) -> str:
    """Return absolute path to the `src/prompts` directory."""
    return os.path.join(repo_root, "src", "prompts")


def find_txt_files_grouped(prompts_root: str) -> Dict[str, List[Dict[str, str]]]:
    """Return mapping: top-level folder → list of {path, key} entries for .txt files.

    The `key` is the path relative to the top-level folder (or the filename if
    at the root). Files directly under `prompts_root` are grouped under
    "_root".
    """
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for dirpath, _, filenames in os.walk(prompts_root):
        for name in filenames:
            if not name.lower().endswith(".txt"):
                continue
            abs_path = os.path.join(dirpath, name)
            rel_from_prompts = os.path.relpath(abs_path, prompts_root)
            parts = rel_from_prompts.split(os.sep)
            if len(parts) == 1:
                folder = "_root"
                key = parts[0]
            else:
                folder = parts[0]
                key = os.path.join(*parts[1:])
            grouped.setdefault(folder, []).append({"path": abs_path, "key": key})
    # sort entries for deterministic output
    for folder, entries in grouped.items():
        entries.sort(key=lambda e: e["key"])  # type: ignore[index]
    return grouped


def load_text(path: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_encoder() -> Optional[Any]:  # type: ignore[override]
    """Return a tiktoken encoder based on env settings, else None.

    Env vars:
    - TOKENIZER_MODEL: if set, try tiktoken.encoding_for_model(model)
    - TOKENIZER_ENCODING: fallback encoding name (default: cl100k_base)
    """
    if tiktoken is None:
        return None
    model = os.getenv("TOKENIZER_MODEL", "").strip()
    encoding_name = os.getenv("TOKENIZER_ENCODING", "cl100k_base").strip()
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def count_tokens(text: str, encoder: Optional[Any]) -> int:
    """Count tokens via `tiktoken` if available; otherwise approximate via words."""
    if encoder is None:
        # Fallback: word count approximation
        return len(text.split())
    try:
        return len(encoder.encode(text))  # type: ignore[attr-defined]
    except Exception:
        return len(text.split())


def main() -> None:
    """Run the token counting pipeline and write the JSON report."""
    repo_root = get_repo_root()
    prompts_root = get_prompts_root(repo_root)
    output_path = os.getenv("OUTPUT_JSON", os.path.join(repo_root, "results", "review_prompt_token_counts.json"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    encoder = get_encoder()
    encoder_desc = "word-count" if encoder is None else os.getenv("TOKENIZER_MODEL", os.getenv("TOKENIZER_ENCODING", "cl100k_base"))

    grouped = find_txt_files_grouped(prompts_root)
    by_folder: Dict[str, Dict[str, int]] = {}
    total = 0
    file_count = 0

    for folder, entries in grouped.items():
        folder_map: Dict[str, int] = {}
        for ent in entries:
            path = ent["path"]
            key = ent["key"].replace("\\", "/")
            text = load_text(path)
            n = count_tokens(text, encoder)
            folder_map[key] = n
            total += n
            file_count += 1
        by_folder[folder] = folder_map

    payload: Dict[str, Any] = {
        "tokenizer": encoder_desc,
        "by_folder": by_folder,
        "total_tokens": total,
        "total_files": file_count,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({"wrote": output_path, "folders": len(by_folder), "files": file_count, "total_tokens": total}))


if __name__ == "__main__":
    main()


