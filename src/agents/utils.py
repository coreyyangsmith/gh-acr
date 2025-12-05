"""Shared utilities for agent modules.

This module provides common helper functions used across all agent
implementations. It centralizes functionality that was previously
duplicated in individual agent files.

Functions
---------
- **render_template**: Simple Mustache-style template rendering
- **extract_text_content**: Extract text from LangChain model results
- **scenario_file_list**: Get list of files from scenario state

Template Syntax
---------------
Templates use `{{ variable }}` syntax (with spaces around the variable name).
This is a deliberately simple subset that avoids external templating engines::

    template = "Hello, {{ name }}! You have {{ count }} messages."
    result = render_template(template, {"name": "Alice", "count": "5"})
    # Result: "Hello, Alice! You have 5 messages."

Note: The syntax requires exactly one space after `{{` and one space before `}}`.

Example Usage
-------------
>>> from src.agents.utils import render_template, extract_text_content
>>> 
>>> # Template rendering
>>> prompt = render_template(
...     "Code: {{ code }}\\nPatch: {{ patch }}",
...     {"code": "def foo(): pass", "patch": "+    return 1"}
... )
>>> 
>>> # Extract content from LLM response
>>> content = extract_text_content(llm_result)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


def render_template(template: str, variables: Dict[str, str]) -> str:
    """Render a Mustache-style template with variable substitution.

    This function performs simple `{{ var }}` replacement without any
    external dependencies. It's designed for prompt templates where we
    control both the template and the variables.

    Parameters
    ----------
    template
        Template string containing `{{ variable }}` placeholders.
        Note: The syntax requires exactly one space after `{{` and
        one space before `}}`.
    variables
        Dictionary mapping variable names to their string values.

    Returns
    -------
    str
        The template with all placeholders replaced.

    Examples
    --------
    >>> render_template("Hello, {{ name }}!", {"name": "World"})
    'Hello, World!'

    >>> render_template(
    ...     "File: {{ path }}\\nDiff: {{ diff }}",
    ...     {"path": "main.py", "diff": "+print('hi')"}
    ... )
    'File: main.py\\nDiff: +print('hi')'

    Notes
    -----
    - Unmatched placeholders are left unchanged in the output.
    - This is intentionally simpler than Jinja2 or other engines.
    - All values are converted to strings before substitution.
    """
    rendered = template
    for key, value in variables.items():
        # Pattern: {{ key }} with exactly one space on each side
        placeholder = f"{{{{ {key} }}}}"
        rendered = rendered.replace(placeholder, str(value))
    return rendered


def extract_text_content(result: Any) -> str:
    """Extract text content from a LangChain model result.

    This function handles the various result types that LangChain models
    can return (AIMessage, string, etc.) and extracts the text content.

    Parameters
    ----------
    result
        A LangChain model result. Can be:
        - An AIMessage with a `.content` attribute
        - A string
        - Any object with a `.content` attribute
        - Any other object (converted via str())

    Returns
    -------
    str
        The extracted text content.

    Examples
    --------
    >>> from langchain_core.messages import AIMessage
    >>> extract_text_content(AIMessage(content="Hello!"))
    'Hello!'

    >>> extract_text_content("Direct string")
    'Direct string'
    """
    try:
        content = result.content if hasattr(result, "content") else str(result)
        return str(content)
    except Exception:
        return str(result)


def scenario_file_list(
    state: Dict[str, Any], 
    fallback_paths: Iterable[str] | None = None
) -> list[str]:
    """Extract the list of files in a merge conflict scenario.

    This function robustly extracts file paths from the scenario state,
    with fallback support for cases where the standard location is empty.

    Parameters
    ----------
    state
        Pipeline state dictionary. Expected to contain:
        - state["sample_row"]["scenario_json"]["files_in_merge_conflict"]
    fallback_paths
        Optional iterable of paths to use if the scenario doesn't specify
        files. Useful when you have file paths from other sources (e.g.,
        parent_a_contents.keys()).

    Returns
    -------
    list[str]
        List of file paths in the merge conflict.

    Examples
    --------
    >>> state = {
    ...     "sample_row": {
    ...         "scenario_json": {
    ...             "files_in_merge_conflict": ["src/main.py", "src/utils.py"]
    ...         }
    ...     }
    ... }
    >>> scenario_file_list(state)
    ['src/main.py', 'src/utils.py']

    >>> # With fallback when scenario is empty
    >>> empty_state = {"sample_row": {}}
    >>> scenario_file_list(empty_state, fallback_paths=["a.py", "b.py", "a.py"])
    ['a.py', 'b.py']
    """
    # Navigate nested structure safely
    files = (
        (state.get("sample_row", {}) or {})
        .get("scenario_json", {}) or {}
    ).get("files_in_merge_conflict", [])

    if files:
        return list(files)

    if fallback_paths is None:
        return []

    try:
        # Return sorted unique paths
        return sorted(set(fallback_paths))
    except Exception:
        return list(fallback_paths)


__all__ = [
    "render_template",
    "extract_text_content",
    "scenario_file_list",
]
