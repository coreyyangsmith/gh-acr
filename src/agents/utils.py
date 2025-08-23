from __future__ import annotations

from typing import Dict, Any, Iterable


def render_template(template: str, variables: Dict[str, str]) -> str:
    """Render a {{ var }} template using a simple replace.

    This mirrors the small helper duplicated in several agent modules.
    """
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", value)
    return rendered


def extract_text_content(result: Any) -> str:
    """Return text content from a LangChain model result or fallback to str()."""
    try:
        content = result.content if hasattr(result, "content") else str(result)
        return str(content)
    except Exception:
        return str(result)


def scenario_file_list(state: Dict[str, Any], fallback_paths: Iterable[str] | None = None) -> list[str]:
    """Return the list of files in the scenario, with robust fallbacks.

    - Reads `files_in_merge_conflict` from `state["sample_row"]["scenario_json"]`
    - If missing, returns the distinct union of keys from the provided fallback_paths
    """
    files = (
        (state.get("sample_row", {}) or {}).get("scenario_json", {}) or {}
    ).get("files_in_merge_conflict", [])
    if files:
        return list(files)
    if fallback_paths is None:
        return []
    try:
        return sorted(set(fallback_paths))
    except Exception:
        return list(fallback_paths)


