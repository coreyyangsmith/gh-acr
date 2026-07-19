"""Tests for structure-aware evidence fitting."""

from __future__ import annotations

import json

from src.agents.prompt_budget import (
    EvidenceBlock,
    fit_global_ab_prompt,
    fit_variable_blocks,
)
from src.utils.degradation import clear_degradations, get_degradations
from tests.helpers import FakeEncoder


JUDGE_TEMPLATE = """Instructions stay forever.

Choose A B or Mix.

Summaries Parent A:
{a_summary}

Summaries Parent B:
{b_summary}

Diffs A:
{a_diff}

Diffs B:
{b_diff}

Return exactly one string:
A
B
Mix
"""

PLAN_TEMPLATE = """You are Conflict Resolution Planner.

## Schema Specification
Return JSON with file paths.

a_diff:
{{ a_diff }}

a_summary:
{{ a_summary }}

b_diff:
{{ b_diff }}

b_summary:
{{ b_summary }}
"""


def test_fit_preserves_instructions_and_clips_evidence(monkeypatch):
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    clear_degradations()
    enc = FakeEncoder()
    # Force a tiny budget via monkeypatch of allowed_prompt_tokens.
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 80)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    paths = ["a.py", "b.py"]
    summaries = {
        "a.py": {"summary_a": " ".join(f"sa{i}" for i in range(40)), "summary_b": " ".join(f"sb{i}" for i in range(40))},
        "b.py": {"summary_a": " ".join(f"ta{i}" for i in range(40)), "summary_b": " ".join(f"tb{i}" for i in range(40))},
    }
    diffs_a = {
        "a.py": " ".join(f"da{i}" for i in range(200)),
        "b.py": " ".join(f"db{i}" for i in range(200)),
    }
    diffs_b = {
        "a.py": " ".join(f"ea{i}" for i in range(200)),
        "b.py": " ".join(f"eb{i}" for i in range(200)),
    }

    report = fit_global_ab_prompt(
        template=JUDGE_TEMPLATE,
        render="format",
        paths=paths,
        summaries=summaries,
        diffs_a=diffs_a,
        diffs_b=diffs_b,
        encoder=enc,
        model_name="openrouter/qwen/qwen3-32b",
        node="conflict_analyzer",
    )
    assert "Instructions stay forever" in report.prompt
    assert "Return exactly one string" in report.prompt
    assert report.was_clipped
    assert report.tokens_after <= 80
    # File labels survive
    assert "a.py:" in report.prompt
    events = get_degradations()
    assert events
    detail = json.loads(events[0]["detail"])
    assert detail["truncation_mode"] == "structured"
    assert detail["was_clipped"] is True


def test_ab_symmetry_equal_caps(monkeypatch):
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    clear_degradations()
    enc = FakeEncoder()
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 100)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    report = fit_global_ab_prompt(
        template=JUDGE_TEMPLATE,
        render="format",
        paths=["f.py"],
        summaries={
            "f.py": {
                "summary_a": " ".join(f"a{i}" for i in range(10)),
                "summary_b": " ".join(f"b{i}" for i in range(10)),
            }
        },
        diffs_a={"f.py": " ".join(f"da{i}" for i in range(80))},
        diffs_b={"f.py": " ".join(f"db{i}" for i in range(80))},
        encoder=enc,
        model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
        node="conflict_analyzer",
        record=False,
    )
    a_tokens = sum(
        a.tokens_after for a in report.actions if a.side == "A"
    )
    b_tokens = sum(
        a.tokens_after for a in report.actions if a.side == "B"
    )
    # Equal side caps within one token of rounding.
    assert abs(a_tokens - b_tokens) <= 1


def test_fit_determinism(monkeypatch):
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    enc = FakeEncoder()
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 90)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    kwargs = dict(
        template=PLAN_TEMPLATE,
        render="mustache",
        paths=["z.py", "a.py"],
        summaries={
            "a.py": {"summary_a": "asum", "summary_b": "bsum"},
            "z.py": {"summary_a": " ".join(f"s{i}" for i in range(50)), "summary_b": " ".join(f"t{i}" for i in range(50))},
        },
        diffs_a={
            "a.py": " ".join(f"x{i}" for i in range(40)),
            "z.py": " ".join(f"y{i}" for i in range(40)),
        },
        diffs_b={
            "a.py": " ".join(f"u{i}" for i in range(40)),
            "z.py": " ".join(f"v{i}" for i in range(40)),
        },
        encoder=enc,
        model_name="openrouter/qwen/qwen3-32b",
        node="conflict_agent",
        record=False,
    )
    r1 = fit_global_ab_prompt(**kwargs)
    r2 = fit_global_ab_prompt(**kwargs)
    assert r1.prompt == r2.prompt
    assert r1.to_dict()["actions"] == r2.to_dict()["actions"]


def test_fit_local_qwen32_uses_real_budget_and_clips(monkeypatch):
    """End-to-end fit against local:Qwen/Qwen3-32B MODEL_COSTS (no budget mock)."""
    monkeypatch.delenv("PROMPT_TRUNCATION_BUFFER", raising=False)
    clear_degradations()
    enc = FakeEncoder()
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    # Build evidence far larger than the ~30.6k local budget.
    huge = " ".join(f"x{i}" for i in range(80_000))
    report = fit_global_ab_prompt(
        template=JUDGE_TEMPLATE,
        render="format",
        paths=["big.py"],
        summaries={"big.py": {"summary_a": huge, "summary_b": huge}},
        diffs_a={"big.py": huge},
        diffs_b={"big.py": huge},
        encoder=enc,
        model_name="local:Qwen/Qwen3-32B",
        node="conflict_analyzer",
    )
    assert report.was_clipped
    assert report.tokens_after <= 30_656
    assert report.budget_tokens == 30_656
    assert "Instructions stay forever" in report.prompt
    assert "Return exactly one string" in report.prompt
    assert not any(
        e.get("category") == "prompt_truncation" and "wrapper_fallback" in str(e.get("detail", ""))
        for e in get_degradations()
    )


def test_resolver_omits_feedback_before_patches(monkeypatch):
    monkeypatch.setenv("PROMPT_TRUNCATION_BUFFER", "0")
    clear_degradations()
    enc = FakeEncoder()
    import src.agents.prompt_budget.fit as fit_mod

    monkeypatch.setattr(fit_mod, "allowed_prompt_tokens", lambda *a, **k: 60)
    monkeypatch.setattr(fit_mod, "REPAIR_HEADROOM_TOKENS", 0)

    template = (
        "Resolver instructions and schema stay.\n"
        "plan:\n{{ plan }}\n"
        "original_code:\n{{ original_code }}\n"
        "patch_a:\n{{ patch_a }}\n"
        "patch_b:\n{{ patch_b }}\n"
        "review_feedback:\n{{ review_feedback }}\n"
    )
    report = fit_variable_blocks(
        template=template,
        render="mustache",
        fixed_variables={"plan": '{"f.py":"merge"}'},
        blocks=[
            EvidenceBlock(block_id="original_code", text="orig", kind="context", priority=20),
            EvidenceBlock(
                block_id="patch_a",
                text=" ".join(f"pa{i}" for i in range(40)),
                side="A",
                kind="diff",
                priority=30,
            ),
            EvidenceBlock(
                block_id="patch_b",
                text=" ".join(f"pb{i}" for i in range(40)),
                side="B",
                kind="diff",
                priority=30,
            ),
            EvidenceBlock(
                block_id="review_feedback",
                text=" ".join(f"fb{i}" for i in range(40)),
                kind="secondary",
                priority=40,
            ),
        ],
        encoder=enc,
        model_name="openrouter/qwen/qwen3-32b",
        node="resolution_agent",
        file_path="f.py",
    )
    assert "Resolver instructions and schema stay" in report.prompt
    assert '{"f.py":"merge"}' in report.prompt
    by_id = {a.block_id: a for a in report.actions}
    # Secondary feedback should be reduced at least as aggressively as patches.
    assert by_id["review_feedback"].tokens_after <= by_id["patch_a"].tokens_after + 1
    assert report.was_clipped
    assert get_degradations()
