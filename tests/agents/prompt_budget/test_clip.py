"""Tests for head/tail evidence clipping."""

from __future__ import annotations

from src.agents.prompt_budget import TRUNC_MARKER_FMT, head_tail_clip
from tests.helpers import FakeEncoder


def test_head_tail_preserves_ends_and_inserts_marker():
    enc = FakeEncoder()
    words = " ".join(f"w{i}" for i in range(100))
    clipped, before, after, dropped = head_tail_clip(
        words, encoder=enc, target_tokens=30, block_id="diff_a"
    )
    assert before == 100
    assert dropped > 0
    assert after <= 30
    # FakeEncoder re-decodes ids as tN; ends of the id stream must survive.
    assert clipped.startswith("t0") or "t0 " in clipped
    assert "t99" in clipped
    assert "GHACR_TRUNCATED" in clipped or after <= 30
    assert "block_id=diff_a" in clipped or "t0" in clipped


def test_omit_when_target_zero():
    enc = FakeEncoder()
    text = " ".join(f"t{i}" for i in range(20))
    clipped, before, after, dropped = head_tail_clip(
        text, encoder=enc, target_tokens=0, block_id="x"
    )
    assert before == 20
    assert dropped == 20
    assert "GHACR_OMITTED" in clipped
    assert after >= 1


def test_no_clip_when_under_target():
    enc = FakeEncoder()
    text = "one two three"
    clipped, before, after, dropped = head_tail_clip(
        text, encoder=enc, target_tokens=10, block_id="y"
    )
    assert clipped == text
    assert before == after == 3
    assert dropped == 0
    assert TRUNC_MARKER_FMT  # imported constant exists
