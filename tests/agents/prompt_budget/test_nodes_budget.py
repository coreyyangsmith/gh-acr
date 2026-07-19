"""Node-level tests for structured prompt budgeting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.utils.degradation import clear_degradations, get_degradations
from tests.helpers import FakeEncoder, RecordingLLM


def test_analyzer_over_budget_keeps_instructions_and_records_artifact(tmp_path, monkeypatch):
    clear_degradations()
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")

    from src.agents.multi_agent.nodes import create_conflict_analyzer_node
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 120)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    huge = " ".join(f"tok{i}" for i in range(200))
    state = {
        "scenario_id": "s1",
        "eval_method": "better_judge",
        "model_name": "openrouter/qwen/qwen3-32b",
        "artifact_root": str(tmp_path / "arts"),
        "summaries": {
            "src/a.py": {"summary_a": huge, "summary_b": huge},
            "src/b.py": {"summary_a": huge, "summary_b": huge},
        },
        "diffs_a": {"src/a.py": huge, "src/b.py": huge},
        "diffs_b": {"src/a.py": huge, "src/b.py": huge},
        "sample_row": {"scenario_json": {"files_in_merge_conflict": ["src/a.py", "src/b.py"]}},
    }

    llm = RecordingLLM()

    def _fake_backend(model_name):
        return FakeEncoder(), llm

    with patch("src.agents.multi_agent.nodes.get_backend", side_effect=_fake_backend):
        # Make analyzer parse succeed with Mix.
        llm_invoke_orig = llm.invoke

        def _invoke(prompt, config=None):
            llm_invoke_orig(prompt, config=config)
            return type("Msg", (), {"content": "Mix"})()

        llm.invoke = _invoke  # type: ignore[method-assign]
        node = create_conflict_analyzer_node("better_judge")
        out = node(state)

    assert out["bypass_decision"] == "MIX"
    assert llm.prompts
    prompt = llm.prompts[0]
    assert "Prefer selecting one complete parent" in prompt or "Choose ONE verdict" in prompt
    assert "Return exactly one string" in prompt
    assert len(prompt.split()) <= 200  # under tiny budget + template words

    events = get_degradations()
    assert any(e["category"] == "prompt_truncation" for e in events)
    detail = json.loads(next(e for e in events if e["category"] == "prompt_truncation")["detail"])
    assert detail["truncation_mode"] == "structured"

    report_path = tmp_path / "arts" / "analyzer" / "artifacts" / "truncation_report.json"
    # resilient_invoke may or may not write depending on success path; check artifacts dir
    art_dir = tmp_path / "arts" / "analyzer" / "artifacts"
    if art_dir.exists():
        reports = list(art_dir.glob("*truncation_report*"))
        assert reports, f"expected truncation report under {art_dir}"
        data = json.loads(Path(reports[0]).read_text(encoding="utf-8"))
        assert data["was_clipped"] is True


def test_bj_no_summary_raw_diffs_still_budgeted(monkeypatch, tmp_path):
    """bj_no_summary seeds summaries from raw diffs; budgeting must still apply."""
    clear_degradations()
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 100)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    from src.agents.multi_agent.nodes import create_conflict_analyzer_node

    raw = " ".join(f"d{i}" for i in range(150))
    state = {
        "scenario_id": "s2",
        "eval_method": "bj_no_summary",
        "model_name": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "artifact_root": str(tmp_path / "arts2"),
        "summaries": {
            "f.py": {"summary_a": raw, "summary_b": raw},
        },
        "diffs_a": {"f.py": raw},
        "diffs_b": {"f.py": raw},
        "sample_row": {"scenario_json": {"files_in_merge_conflict": ["f.py"]}},
    }
    llm = RecordingLLM()

    def _invoke(prompt, config=None):
        llm.prompts.append(prompt)
        return type("Msg", (), {"content": "A"})()

    llm.invoke = _invoke  # type: ignore[method-assign]

    with patch(
        "src.agents.multi_agent.nodes.get_backend",
        return_value=(FakeEncoder(), llm),
    ):
        out = create_conflict_analyzer_node("better_judge")(state)

    assert out["bypass_decision"] in {"ALL_A", "ALL_B", "MIX"}
    assert get_degradations()
    assert "Choose ONE verdict" in llm.prompts[0] or "Prefer selecting" in llm.prompts[0]
