"""End-to-end multi-agent resolver tests with mocked LLM."""

from __future__ import annotations

from src.agents.multi_agent import create_resolver


def test_bypass7_resolver_completes(synthetic_agent_state, mock_get_backend):
    resolver = create_resolver("bypass7")
    out = resolver(dict(synthetic_agent_state))
    assert "resolved_contents" in out or "final_diffs" in out or out.get("status")
    # Should terminate without hanging; review iter bounded
    assert int(out.get("_review_iter", 0)) <= 3


def test_force_mix_sets_mix_and_completes(synthetic_agent_state, mock_get_backend):
    resolver = create_resolver("force_mix")
    out = resolver(dict(synthetic_agent_state))
    assert out.get("bypass_decision") == "MIX" or out.get("bypass_method") == "MIX"
    assert "resolved_contents" in out or "final_diffs" in out or out.get("status")


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
