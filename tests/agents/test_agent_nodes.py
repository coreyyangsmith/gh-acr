"""Agent node tests with mocked LLM backends."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.helpers import FakeEncoder, RecordingLLM


def test_single_agent_uses_llm_output(synthetic_agent_state, mock_get_backend):
    from src.agents.single_agent.merge_agent import resolve_conflict_agent_node

    out = resolve_conflict_agent_node(synthetic_agent_state)
    assert out["status"] == "resolved_agent"
    assert "src/main.py" in out["resolved_contents"]
    assert out["resolved_contents"]["src/main.py"]


def test_single_agent_fallback_without_llm(synthetic_agent_state):
    from src.agents.single_agent.merge_agent import resolve_conflict_agent_node

    with patch(
        "src.agents.single_agent.merge_agent.get_backend",
        return_value=(None, None),
    ):
        out = resolve_conflict_agent_node(synthetic_agent_state)
    assert out["status"] == "resolved_agent_stub"
    assert out["resolved_contents"] == synthetic_agent_state["parent_a_contents"]


def test_single_agent_huge_inputs_budgeted_with_truncation_report(
    synthetic_agent_state, tmp_path, monkeypatch
):
    """Single-agent path fits evidence; over-budget prompts write truncation_report."""
    from src.utils.degradation import clear_degradations, get_degradations
    import src.agents.prompt_budget.fit as fit_mod

    clear_degradations()
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 80)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    from src.agents.single_agent.merge_agent import resolve_conflict_agent_node

    huge = " ".join(f"code{i}" for i in range(300))
    synthetic_agent_state["artifact_root"] = str(tmp_path / "agent_arts")
    synthetic_agent_state["model_name"] = "openrouter/meta-llama/llama-3.1-8b-instruct"
    synthetic_agent_state["ancestor_contents"] = {"src/main.py": huge}
    synthetic_agent_state["diffs_a"] = {"src/main.py": huge}
    synthetic_agent_state["diffs_b"] = {"src/main.py": huge}
    synthetic_agent_state["parent_a_contents"] = {"src/main.py": huge}
    synthetic_agent_state["parent_b_contents"] = {"src/main.py": huge}

    llm = RecordingLLM()

    def _invoke(prompt, config=None):
        llm.prompts.append(prompt)
        return type("Msg", (), {"content": "merged = True\n"})()

    llm.invoke = _invoke  # type: ignore[method-assign]

    with patch(
        "src.agents.single_agent.merge_agent.get_backend",
        return_value=(FakeEncoder(), llm),
    ):
        out = resolve_conflict_agent_node(synthetic_agent_state)

    assert out["status"] == "resolved_agent"
    assert llm.prompts
    assert "Output ONLY the merged file content" in llm.prompts[0]
    assert "src/main.py" in llm.prompts[0]
    events = get_degradations()
    assert any(e["category"] == "prompt_truncation" for e in events)
    reports = list((tmp_path / "agent_arts").rglob("truncation_report.json"))
    assert reports, "expected truncation_report.json from single-agent fit"
    import json
    from pathlib import Path

    data = json.loads(Path(reports[0]).read_text(encoding="utf-8"))
    assert data["was_clipped"] is True
    assert data["node"] == "resolve_conflict_agent"
    # Wrapper should not need to fire when structured fit already clipped.
    assert not any(
        "wrapper_fallback" in str(e.get("detail", "")) for e in events
    )


def test_summarizer_node_with_llm(synthetic_agent_state, mock_get_backend):
    from src.agents.multi_agent.nodes import create_summarizer_node

    node = create_summarizer_node("bypass7")
    out = node(synthetic_agent_state)
    assert out["status"] == "summarised"
    assert "src/main.py" in out["summaries"]
    assert "summary_a" in out["summaries"]["src/main.py"]


def test_summarizer_fallback_without_llm(synthetic_agent_state):
    from src.agents.multi_agent.nodes import create_summarizer_node

    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(FakeEncoder(), None),
    ):
        node = create_summarizer_node("bypass7")
        out = node(synthetic_agent_state)
    assert out["status"] == "summarised"
    assert "Adds" in out["summaries"]["src/main.py"]["summary_a"]


def test_analyzer_node_sets_bypass_decision(synthetic_agent_state, mock_get_backend):
    from src.agents.multi_agent.nodes import create_conflict_analyzer_node

    # Pre-seed summaries so analyzer has context
    synthetic_agent_state["summaries"] = {
        "src/main.py": {"summary_a": "changed a", "summary_b": "changed b"}
    }
    node = create_conflict_analyzer_node("bypass7")
    out = node(synthetic_agent_state)
    assert out["status"] == "analyzed"
    assert out["bypass_decision"] in {"ALL_A", "ALL_B", "MIX"}


def test_resolution_and_review_nodes(synthetic_agent_state, mock_get_backend):
    from src.agents.multi_agent.nodes import (
        create_conflict_agent_node,
        create_resolution_agent_node,
        create_review_agent_node,
    )

    synthetic_agent_state["summaries"] = {
        "src/main.py": {"summary_a": "a", "summary_b": "b"}
    }
    synthetic_agent_state["bypass_decision"] = "MIX"

    plan_node = create_conflict_agent_node("bypass7")
    state = plan_node(synthetic_agent_state)
    assert state["status"] == "planned"
    assert "conflict_plan" in state

    # Force a simple A choice so resolution is deterministic
    state["conflict_plan"] = {"src/main.py": "A"}
    resolve_node = create_resolution_agent_node("bypass7")
    state = resolve_node(state)
    assert state["resolved_contents"]["src/main.py"] == synthetic_agent_state["parent_a_contents"]["src/main.py"]

    review_node = create_review_agent_node("bypass7")
    state = review_node(state)
    assert "review_results" in state


def test_conflict_agent_parses_per_file_plan_schema(synthetic_agent_state):
    """The planner must map each conflicted file path to "A"/"B"/"merge",
    matching the contract documented in src/prompts/*/plan_prompt.txt and
    consumed by create_resolution_agent_node via plan.get(path, "merge")."""
    from src.agents.multi_agent.nodes import create_conflict_agent_node

    synthetic_agent_state["summaries"] = {
        "src/main.py": {"summary_a": "changed a", "summary_b": "changed b"}
    }

    class _StubLLM:
        def invoke(self, prompt, config=None):
            return type("Msg", (), {"content": '{"src/main.py": "A"}'})()

        async def ainvoke(self, prompt, config=None):
            return self.invoke(prompt, config=config)

    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(FakeEncoder(), _StubLLM()),
    ):
        node = create_conflict_agent_node("bypass7")
        out = node(synthetic_agent_state)

    assert out["conflict_plan"] == {"src/main.py": "A"}


def test_conflict_agent_rejects_schema_mismatched_plan(synthetic_agent_state):
    """Regression test for a real production bug: the plan prompt used to
    request a {strategy, steps, resolution} object instead of a per-file
    map, which parsed as valid JSON but silently defaulted every file to
    "merge" via plan.get(path, "merge"). The planner must now detect this
    shape mismatch and fall back explicitly instead of propagating an
    unusable plan downstream."""
    from src.agents.multi_agent.nodes import create_conflict_agent_node

    synthetic_agent_state["summaries"] = {
        "src/main.py": {"summary_a": "changed a", "summary_b": "changed b"}
    }

    class _StubLLM:
        def invoke(self, prompt, config=None):
            content = (
                '{"strategy": "merge both", "steps": ["do it"], '
                '"resolution": "done"}'
            )
            return type("Msg", (), {"content": content})()

        async def ainvoke(self, prompt, config=None):
            return self.invoke(prompt, config=config)

    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(FakeEncoder(), _StubLLM()),
    ):
        node = create_conflict_agent_node("bypass7")
        out = node(synthetic_agent_state)

    assert out["conflict_plan"] == {"src/main.py": "merge"}


def test_resolution_agent_bypasses_llm_for_per_file_ab_choice(synthetic_agent_state):
    """When the plan says "A" or "B" for a file, the resolver must copy that
    parent's content directly and must NOT invoke the LLM for that file."""
    from src.agents.multi_agent.nodes import create_resolution_agent_node

    synthetic_agent_state["conflict_plan"] = {"src/main.py": "A"}

    class _RaisingLLM:
        def invoke(self, prompt, config=None):
            raise AssertionError("LLM should not be called for an A/B plan choice")

        async def ainvoke(self, prompt, config=None):
            return self.invoke(prompt, config=config)

    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(FakeEncoder(), _RaisingLLM()),
    ):
        node = create_resolution_agent_node("bypass7")
        out = node(synthetic_agent_state)

    assert (
        out["resolved_contents"]["src/main.py"]
        == synthetic_agent_state["parent_a_contents"]["src/main.py"]
    )
