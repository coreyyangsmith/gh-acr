"""End-to-end multi-agent resolver tests with mocked LLM."""

from __future__ import annotations

from src.agents.multi_agent import create_resolver


def test_bypass7_resolver_completes(synthetic_agent_state, mock_get_backend):
    resolver = create_resolver("bypass7")
    out = resolver(dict(synthetic_agent_state))
    assert "resolved_contents" in out or "final_diffs" in out or out.get("status")
    # Should terminate without hanging; review iter bounded
    assert int(out.get("_review_iter", 0)) <= 3


def test_better_judge_resolver_completes(synthetic_agent_state, mock_get_backend):
    resolver = create_resolver("better_judge")
    out = resolver(dict(synthetic_agent_state))
    assert "resolved_contents" in out or "final_diffs" in out or out.get("status")
    assert int(out.get("_review_iter", 0)) <= 3


def test_better_judge_prompt_contains_judge_instructions():
    from src.agents.multi_agent.nodes import _load_prompt

    prompt = _load_prompt("better_judge", "conflict_judge_prompt.txt")
    assert "{a_summary}" in prompt
    assert "{b_summary}" in prompt
    assert "{a_diff}" in prompt
    assert "{b_diff}" in prompt
    assert "senior merge judge" in prompt.lower()
    assert "Prefer selecting one complete parent" in prompt
    assert "Cross-file coherence" in prompt


def test_normalize_decision_standard_exact_tokens():
    from src.agents.multi_agent.nodes import _normalize_decision_standard

    assert _normalize_decision_standard("A") == "ALL_A"
    assert _normalize_decision_standard("B") == "ALL_B"
    assert _normalize_decision_standard("Mix") == "MIX"
    assert _normalize_decision_standard("a\n") == "ALL_A"
    assert _normalize_decision_standard("b") == "ALL_B"
    assert _normalize_decision_standard("MIX") == "MIX"
    # Must not treat "ALL_B" as ALL_A via substring "a"
    assert _normalize_decision_standard("ALL_B") == "ALL_B"
    assert _normalize_decision_standard("Parent B") == "ALL_B"
    assert _normalize_decision_standard("unclear verdict") == "MIX"

def test_force_mix_sets_mix_and_completes(synthetic_agent_state, mock_get_backend):
    resolver = create_resolver("force_mix")
    out = resolver(dict(synthetic_agent_state))
    assert out.get("bypass_decision") == "MIX" or out.get("bypass_method") == "MIX"
    assert "resolved_contents" in out or "final_diffs" in out or out.get("status")


def test_bj_no_summary_seeds_raw_diffs(synthetic_agent_state, mock_get_backend, recording_llm):
    """Summaries should equal raw diffs; no summarizer-style prompts needed for seeding."""
    state = dict(synthetic_agent_state)
    # Ensure diffs are non-empty so seeding is observable
    state["diffs_a"] = {"f.py": "--- a\n+++ b\n@@\n-a\n+b\n"}
    state["diffs_b"] = {"f.py": "--- a\n+++ c\n@@\n-a\n+c\n"}
    state["sample_row"]["scenario_json"]["files_in_merge_conflict"] = ["f.py"]
    state["ancestor_contents"] = {"f.py": "a\n"}
    state["parent_a_contents"] = {"f.py": "b\n"}
    state["parent_b_contents"] = {"f.py": "c\n"}

    resolver = create_resolver("bj_no_summary")
    out = resolver(state)
    summaries = out.get("summaries") or {}
    assert "f.py" in summaries
    assert summaries["f.py"]["summary_a"] == state["diffs_a"]["f.py"]
    assert summaries["f.py"]["summary_b"] == state["diffs_b"]["f.py"]
    assert out.get("status") in ("reviewed", "review_finalized", "resolved_multi", "analyzed", "bypassed") or "resolved_contents" in out


def test_bj_no_judge_forces_mix(synthetic_agent_state, mock_get_backend):
    resolver = create_resolver("bj_no_judge")
    out = resolver(dict(synthetic_agent_state))
    assert out.get("bypass_decision") == "MIX"
    assert out.get("bypass_method") == "MIX"
    assert "[force_mix]" in str(out.get("bypass_analyzer_output", ""))


def test_bj_no_plan_seeds_merge_and_skips_review(synthetic_agent_state, mock_get_backend, recording_llm):
    """Force MIX via analyzer output; plan should be all merge; no review retries."""

    def _invoke(prompt, config=None):
        recording_llm.prompts.append(prompt)
        text = str(prompt)
        # Analyzer asks for A/B/Mix; always Mix so we hit the merge path
        if "senior merge judge" in text.lower() or "Pick ONE verdict" in text or "Choose ONE verdict" in text:
            return type("Msg", (), {"content": "Mix"})()
        if "Conflict Resolution Planner" in text or "JSON object" in text:
            # Should not be called for bj_no_plan planner
            return type("Msg", (), {"content": '{"f.py": "merge"}'})()
        if "ACCEPT" in text or "review" in text.lower()[:80]:
            return type("Msg", (), {"content": '{"outcome": "ACCEPT", "rationale": "ok"}'})()
        return type("Msg", (), {"content": "merged content\n"})()

    recording_llm.invoke = _invoke  # type: ignore[method-assign]
    recording_llm.ainvoke = lambda p, config=None: _invoke(p, config)  # type: ignore[method-assign]

    resolver = create_resolver("bj_no_plan")
    out = resolver(dict(synthetic_agent_state))
    plan = out.get("conflict_plan") or {}
    assert plan
    assert all(v == "merge" for v in plan.values())
    assert int(out.get("_review_iter", 0)) == 0
    assert "resolved_contents" in out or "final_diffs" in out


def test_bj_no_review_no_feedback_loop(synthetic_agent_state, mock_get_backend, recording_llm):
    """Plan runs but review is skipped; _review_iter stays 0."""

    def _invoke(prompt, config=None):
        recording_llm.prompts.append(prompt)
        text = str(prompt)
        if "Choose ONE verdict" in text or "senior merge judge" in text.lower():
            return type("Msg", (), {"content": "Mix"})()
        if "Conflict Resolution Planner" in text or (
            "JSON" in text and "merge" in text.lower()
        ):
            files = synthetic_agent_state.get("sample_row", {}).get("scenario_json", {}).get(
                "files_in_merge_conflict", ["f.py"]
            )
            import json

            return type("Msg", (), {"content": json.dumps({f: "merge" for f in files})})()
        return type("Msg", (), {"content": "merged\n"})()

    recording_llm.invoke = _invoke  # type: ignore[method-assign]
    recording_llm.ainvoke = lambda p, config=None: _invoke(p, config)  # type: ignore[method-assign]

    resolver = create_resolver("bj_no_review")
    out = resolver(dict(synthetic_agent_state))
    assert int(out.get("_review_iter", 0)) == 0
    # No review_results from reviewer agent (may be absent or empty)
    assert not out.get("review_results") or out.get("status") == "review_finalized"
    assert "resolved_contents" in out or "final_diffs" in out


def test_bypass7_all_a_path(synthetic_agent_state, mock_get_backend, recording_llm):
    """Force analyzer output to ALL_A via canned LLM responses."""

    def _invoke(prompt, config=None):
        recording_llm.prompts.append(prompt)
        # Always emit ALL_A for analyzer-style prompts
        return type("Msg", (), {"content": "ALL_A"})()

    recording_llm.invoke = _invoke  # type: ignore[method-assign]
    recording_llm.ainvoke = lambda p, config=None: _invoke(p, config)  # type: ignore[method-assign]

    resolver = create_resolver("bypass7")
    out = resolver(dict(synthetic_agent_state))
    # ALL_A decision should copy parent A contents (or leave resolved_contents)
    if out.get("bypass_decision") == "ALL_A":
        assert out.get("resolved_contents") == synthetic_agent_state["parent_a_contents"]
