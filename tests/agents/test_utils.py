"""Unit tests for src.agents.utils helpers."""

from __future__ import annotations

from src.agents.utils import extract_text_content, render_template, scenario_file_list


def test_render_template_single_and_multiple():
    assert render_template("Hello, {{ name }}!", {"name": "World"}) == "Hello, World!"
    out = render_template(
        "File: {{ path }}\nDiff: {{ diff }}",
        {"path": "main.py", "diff": "+x"},
    )
    assert out == "File: main.py\nDiff: +x"


def test_render_template_unmatched_left_intact():
    assert render_template("{{ missing }}", {}) == "{{ missing }}"


def test_render_template_coerces_non_string():
    assert render_template("n={{ n }}", {"n": 42}) == "n=42"


def test_render_template_requires_exact_spacing():
    # Wrong spacing is not substituted
    assert render_template("{{name}}", {"name": "x"}) == "{{name}}"
    assert render_template("{{  name  }}", {"name": "x"}) == "{{  name  }}"


def test_extract_text_content_message_like():
    msg = type("Msg", (), {"content": "hello"})()
    assert extract_text_content(msg) == "hello"


def test_extract_text_content_plain_str():
    assert extract_text_content("direct") == "direct"


def test_extract_text_content_other_object():
    assert extract_text_content(123) == "123"


def test_scenario_file_list_from_state():
    state = {
        "sample_row": {
            "scenario_json": {"files_in_merge_conflict": ["a.py", "b.py"]}
        }
    }
    assert scenario_file_list(state) == ["a.py", "b.py"]


def test_scenario_file_list_fallback_dedupe_sort():
    empty = {"sample_row": {}}
    assert scenario_file_list(empty, fallback_paths=["b.py", "a.py", "a.py"]) == [
        "a.py",
        "b.py",
    ]


def test_scenario_file_list_empty_no_fallback():
    assert scenario_file_list({"sample_row": {}}) == []
