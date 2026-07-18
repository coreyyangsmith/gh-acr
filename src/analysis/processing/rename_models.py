"""Rename/normalize model names in a results CSV using a built-in mapping.

Usage (from repo root, PowerShell one-liners):

- Module form (recommended):
  python -m src.analysis.processing.rename_models data\2025_10_18_ALL_RESULTS.csv data\2025_10_18_ALL_RESULTS_RENAMED.csv model_name

- Direct script:
  python src\results\processing\rename_models.py data\input.csv data\output_renamed.csv --column model_name

Notes:
- Only the specified column (defaults to `model_name`) is transformed.
- Unmapped values are left as-is (with a small heuristic normalization pass).
- A brief summary of changes is printed after writing the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import tyro


# Exact input -> output mappings for known variants. Keys are matched case-sensitively
# after trimming surrounding whitespace. If a name is not present here, we fall back to
# a light-weight normalization rule (see _normalize_provider_format).
#
# Add/adjust entries here as new model-name variants appear in incoming CSVs.
MODEL_NAME_MAP: Dict[str, str] = {
    # OpenAI
    "openai/gpt-5-nano": "gpt-5-nano",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4.1-nano-2025-04-14": "gpt-4.1-nano-2025-04-14",
    # Common OpenAI variants
    "openai:gpt-5-nano": "gpt-5-nano",
    "openai:gpt-5-mini": "gpt-5-mini",
    "openai:gpt-5": "gpt-5",
    "openai:gpt-4o-mini": "gpt-4o-mini",

    # Groq - collate llama variants
    "groq:llama-3.1-8b-instant": "llama-3.1-8b",
    "groq/llama-3.1-8b-instant": "llama-3.1-8b",  # unify slash->colon
    "groq_llama-3.1-8b-instant": "llama-3.1-8b",  # underscore variant

    "groq:qwen/qwen3-32b": "qwen3-32b",
    "groq/qwen/qwen3-32b": "qwen3-32b",  # unify slash->colon
    "qwen/qwen3-32b": "qwen3-32b",  # missing provider -> assume groq
    "qwen3-32b": "qwen3-32b",  # minimal alias -> assume groq

    # Local (Transformers/backends) - collate llama variants
    "local:meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "local/meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",

    "local:meta-llama/Llama-3.1-8B": "llama-3.1-8b",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b",

    "local:Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-8B": "qwen3-8b",

    "local:meta-llama/Llama-3.2-1B": "llama-3.2-1b",
    "meta-llama/Llama-3.2-1B": "llama-3.2-1b",

    "local:google/codegemma-7b-it": "codegemma-7b",
    "google/codegemma-7b-it": "codegemma-7b",

    "local:openai/gpt-oss-20b": "gpt-oss-20b",
    "openai/gpt-oss-20b": "gpt-oss-20b",

    # Misc local toy
    "local:distilbert/distilgpt2": "distilgpt2",
}

def _normalize_provider_format(name: str) -> str:
    """Apply light heuristics to normalize common provider prefixes.

    - Convert "groq/…" -> "groq:…"
    - Convert "openai:…" -> "openai/…"
    - Convert "local/…" -> "local:…"
    - Convert provider_… (underscore) -> provider:… or provider/… as above
    """
    s = name.strip()
    if not s:
        return s

    # Underscore variants used sometimes for filesystem-friendly names
    if s.startswith("groq_"):
        return "groq:" + s.split("_", 1)[1]
    if s.startswith("openai_"):
        return "openai/" + s.split("_", 1)[1]
    if s.startswith("local_"):
        return "local:" + s.split("_", 1)[1]

    # Slash/colon harmonization across providers
    if s.startswith("groq/"):
        return "groq:" + s.split("/", 1)[1]
    if s.startswith("openai:"):
        return "openai/" + s.split(":", 1)[1]
    if s.startswith("local/"):
        return "local:" + s.split("/", 1)[1]

    return s


def _apply_mapping(raw: str) -> Tuple[str, bool]:
    """Return (mapped_value, changed?).

    Mapping strategy:
    1) Try exact match in MODEL_NAME_MAP (after strip).
    2) If not found, attempt provider-format normalization, then lookup again.
    3) If still not found, return the normalized value (if changed) else original.
    """
    if pd.isna(raw):  # type: ignore
        return raw, False  # type: ignore

    original = str(raw)
    key = original.strip()

    # 1) Exact mapping
    if key in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[key], MODEL_NAME_MAP[key] != original

    # 2) Normalize provider format, then try mapping again
    normalized = _normalize_provider_format(key)
    if normalized in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[normalized], MODEL_NAME_MAP[normalized] != original

    # 3) Fallback: if normalization changed the string, return normalized
    if normalized != original:
        return normalized, True

    # No change
    return original, False


def main(input_file: Path, output_file: Path, *, column: str = "model_name") -> None:
    df = pd.read_csv(input_file)
    if column not in df.columns:
        # Best-effort auto-detect if the requested column is missing
        candidates = [c for c in df.columns if c.lower() in {"model", "model_name"} or "model" in c.lower()]
        if candidates:
            column = candidates[0]
        else:
            raise ValueError(f"Column '{column}' not found and no 'model*' column detected in input CSV.")

    work = df.copy()
    before = work[column].copy()
    mapped_values: list[Optional[str]] = []
    changed = 0
    missing = 0

    for v in before:
        new_v, is_changed = _apply_mapping(v)
        if pd.isna(v):  # type: ignore
            missing += 1
        if is_changed:
            changed += 1
        mapped_values.append(new_v)

    work[column] = mapped_values

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(output_file, index=False)

    total = int(len(work))
    print({
        "input": str(input_file),
        "output": str(output_file),
        "column": column,
        "rows": total,
        "changed": int(changed),
        "missing_in_input": int(missing),
        "unchanged": int(total - changed),
    })


if __name__ == "__main__":
    args = tyro.cli(tuple[Path, Path, Optional[str]])
    in_path, out_path, col = args
    main(in_path, out_path, column=col or "model_name")



