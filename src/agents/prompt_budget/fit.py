"""Deterministic structure-aware evidence fitting for merge prompts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

from ...utils.degradation import record_degradation
from ..token_utils import count_tokens
from ..utils import render_template
from .budget import allowed_prompt_tokens
from .clip import head_tail_clip

logger = logging.getLogger(__name__)

# Leave room for invoke_and_parse repair suffixes appended after the first attempt.
REPAIR_HEADROOM_TOKENS = 256


@dataclass(frozen=True)
class EvidenceBlock:
    """One variable evidence region that may be clipped or omitted."""

    block_id: str
    text: str
    side: Literal["A", "B", ""] = ""
    kind: Literal["summary", "diff", "context", "primary", "secondary"] = "context"
    file_path: str | None = None
    # Lower priority value = keep longer (clip/omit later).
    priority: int = 50


@dataclass
class BlockFitAction:
    block_id: str
    action: Literal["kept", "head_tail_clipped", "omitted"]
    tokens_before: int
    tokens_after: int
    omitted_tokens: int
    side: str = ""
    kind: str = ""
    file_path: str | None = None
    text: str = ""


@dataclass
class FitReport:
    prompt: str
    budget_tokens: int
    immutable_tokens: int
    evidence_budget: int
    tokens_before: int
    tokens_after: int
    was_clipped: bool
    truncation_mode: str = "structured"
    node: str = ""
    model_name: str = ""
    file_path: str | None = None
    actions: list[BlockFitAction] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "truncation_mode": self.truncation_mode,
            "node": self.node,
            "model_name": self.model_name,
            "file_path": self.file_path,
            "budget_tokens": self.budget_tokens,
            "immutable_tokens": self.immutable_tokens,
            "evidence_budget": self.evidence_budget,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "was_clipped": self.was_clipped,
            "actions": [
                {
                    "block_id": a.block_id,
                    "action": a.action,
                    "tokens_before": a.tokens_before,
                    "tokens_after": a.tokens_after,
                    "omitted_tokens": a.omitted_tokens,
                    "side": a.side,
                    "kind": a.kind,
                    "file_path": a.file_path,
                }
                for a in self.actions
            ],
        }

    def artifact_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _kind_rank(kind: str) -> int:
    # Lower = allocate first / clip last.
    order = {
        "primary": 0,
        "summary": 1,
        "context": 2,
        "diff": 3,
        "secondary": 4,
    }
    return order.get(kind, 50)


def _allocate_targets(
    blocks: Sequence[EvidenceBlock],
    *,
    encoder: Any,
    evidence_budget: int,
) -> dict[str, int]:
    """Return per-block token targets summing to <= evidence_budget."""
    sizes = {b.block_id: count_tokens(encoder, b.text or "") for b in blocks}
    total = sum(sizes.values())
    if evidence_budget <= 0:
        return {b.block_id: 0 for b in blocks}
    if total <= evidence_budget:
        return dict(sizes)

    targets = {b.block_id: 0 for b in blocks}
    sides = sorted({b.side for b in blocks if b.side})
    if len(sides) >= 2:
        # Equal A/B caps; leftover (odd token) goes to first side lexicographically.
        base = evidence_budget // len(sides)
        rem = evidence_budget - base * len(sides)
        side_caps = {s: base for s in sides}
        for s in sides:
            if rem <= 0:
                break
            side_caps[s] += 1
            rem -= 1
    else:
        only = sides[0] if sides else ""
        side_caps = {only: evidence_budget}

    for side, cap in side_caps.items():
        side_blocks = [b for b in blocks if (b.side or "") == side]
        if not side_blocks:
            continue
        remaining = cap
        # Allocate high-priority kinds first.
        ordered = sorted(
            side_blocks,
            key=lambda b: (_kind_rank(b.kind), b.priority, b.file_path or "", b.block_id),
        )
        # First pass: proportional within each kind group while budget remains.
        by_kind: dict[str, list[EvidenceBlock]] = {}
        for b in ordered:
            by_kind.setdefault(b.kind, []).append(b)
        for kind in sorted(by_kind.keys(), key=_kind_rank):
            group = by_kind[kind]
            group_need = sum(sizes[b.block_id] for b in group)
            if group_need <= remaining:
                for b in group:
                    targets[b.block_id] = sizes[b.block_id]
                remaining -= group_need
                continue
            # Proportional split of remaining across this kind; deterministic by path.
            group_sorted = sorted(group, key=lambda b: (b.file_path or "", b.block_id))
            weights = [max(1, sizes[b.block_id]) for b in group_sorted]
            weight_sum = sum(weights)
            assigned = 0
            for i, b in enumerate(group_sorted):
                if i == len(group_sorted) - 1:
                    share = max(0, remaining - assigned)
                else:
                    share = (remaining * weights[i]) // weight_sum
                take = min(sizes[b.block_id], share)
                targets[b.block_id] = take
                assigned += take
            remaining = max(0, remaining - assigned)

        # If still over (rounding), shrink largest targets first.
        used = sum(targets[b.block_id] for b in side_blocks)
        overflow = used - cap
        if overflow > 0:
            shrinkable = sorted(
                side_blocks,
                key=lambda b: (
                    -_kind_rank(b.kind),
                    -targets[b.block_id],
                    b.file_path or "",
                    b.block_id,
                ),
            )
            for b in shrinkable:
                if overflow <= 0:
                    break
                reducible = targets[b.block_id]
                cut = min(reducible, overflow)
                targets[b.block_id] -= cut
                overflow -= cut

    return targets


def _apply_targets(
    blocks: Sequence[EvidenceBlock],
    targets: Mapping[str, int],
    *,
    encoder: Any,
) -> list[BlockFitAction]:
    actions: list[BlockFitAction] = []
    for b in blocks:
        target = int(targets.get(b.block_id, 0))
        before = count_tokens(encoder, b.text or "")
        if before <= target:
            actions.append(
                BlockFitAction(
                    block_id=b.block_id,
                    action="kept",
                    tokens_before=before,
                    tokens_after=before,
                    omitted_tokens=0,
                    side=b.side,
                    kind=b.kind,
                    file_path=b.file_path,
                    text=b.text or "",
                )
            )
            continue
        clipped, tokens_before, tokens_after, omitted = head_tail_clip(
            b.text or "",
            encoder=encoder,
            target_tokens=target,
            block_id=b.block_id,
        )
        action: Literal["kept", "head_tail_clipped", "omitted"]
        if target <= 0 or omitted >= tokens_before:
            action = "omitted"
        else:
            action = "head_tail_clipped"
        actions.append(
            BlockFitAction(
                block_id=b.block_id,
                action=action,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                omitted_tokens=omitted,
                side=b.side,
                kind=b.kind,
                file_path=b.file_path,
                text=clipped,
            )
        )
    return actions


def _render_vars(
    template: str,
    render: Literal["mustache", "format"],
    variables: Mapping[str, str],
) -> str:
    if render == "format":
        return template.format(**variables)
    return render_template(template, dict(variables))


def _shrink_to_budget(
    *,
    template: str,
    render: Literal["mustache", "format"],
    fixed_variables: Mapping[str, str],
    blocks: Sequence[EvidenceBlock],
    encoder: Any,
    budget: int,
    evidence_budget: int,
) -> tuple[list[BlockFitAction], dict[str, str], str, int]:
    """Allocate/clip, then reduce evidence_budget until rendered prompt fits."""
    working_budget = evidence_budget
    actions: list[BlockFitAction] = []
    fitted_vars: dict[str, str] = {}
    prompt = ""
    tokens_after = 0
    for _ in range(8):
        targets = _allocate_targets(blocks, encoder=encoder, evidence_budget=working_budget)
        actions = _apply_targets(blocks, targets, encoder=encoder)
        fitted_vars = {**dict(fixed_variables), **{a.block_id: a.text for a in actions}}
        prompt = _render_vars(template, render, fitted_vars)
        tokens_after = count_tokens(encoder, prompt)
        if tokens_after <= budget or working_budget <= 0:
            break
        # Reduce working evidence budget by the overrun (+small margin).
        working_budget = max(0, working_budget - (tokens_after - budget) - 4)
    return actions, fitted_vars, prompt, tokens_after


def _record_fit_degradation(report: FitReport) -> None:
    if not report.was_clipped:
        return
    record_degradation(
        "prompt_truncation",
        "structure-aware prompt fit applied",
        detail=json.dumps(report.to_dict(), ensure_ascii=False),
        node=report.node or None,
        file=report.file_path,
    )


def fit_variable_blocks(
    *,
    template: str,
    render: Literal["mustache", "format"],
    fixed_variables: Mapping[str, str],
    blocks: Sequence[EvidenceBlock],
    encoder: Any,
    model_name: str,
    node: str,
    file_path: str | None = None,
    record: bool = True,
    repair_headroom: int = REPAIR_HEADROOM_TOKENS,
) -> FitReport:
    """Fit evidence blocks into a prompt template under the model budget.

    ``fixed_variables`` are never clipped (plans, labels). ``blocks`` are the
    expandable evidence regions. The immutable token cost is measured by
    rendering the template with empty strings for every block id.
    """
    budget = allowed_prompt_tokens(model_name, encoder=encoder)
    empty_vars = {**dict(fixed_variables), **{b.block_id: "" for b in blocks}}
    immutable_prompt = _render_vars(template, render, empty_vars)
    immutable_tokens = count_tokens(encoder, immutable_prompt)

    evidence_budget = max(0, budget - immutable_tokens - max(0, repair_headroom))
    tokens_before_evidence = sum(count_tokens(encoder, b.text or "") for b in blocks)

    actions, fitted_vars, prompt, tokens_after = _shrink_to_budget(
        template=template,
        render=render,
        fixed_variables=fixed_variables,
        blocks=blocks,
        encoder=encoder,
        budget=budget,
        evidence_budget=evidence_budget,
    )

    tokens_before = immutable_tokens + tokens_before_evidence
    was_clipped = any(a.action != "kept" for a in actions) or tokens_after < tokens_before

    report = FitReport(
        prompt=prompt,
        budget_tokens=budget,
        immutable_tokens=immutable_tokens,
        evidence_budget=evidence_budget,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        was_clipped=was_clipped,
        node=node,
        model_name=model_name,
        file_path=file_path,
        actions=actions,
        variables={k: fitted_vars[k] for k in fitted_vars},
    )
    if record:
        _record_fit_degradation(report)
    if was_clipped:
        logger.warning(
            "[prompt_budget] Structured fit node=%s model=%s before=%d after=%d "
            "budget=%d evidence_budget=%d clipped_blocks=%d",
            node,
            model_name,
            tokens_before,
            tokens_after,
            budget,
            evidence_budget,
            sum(1 for a in actions if a.action != "kept"),
        )
    return report


def fit_global_ab_prompt(
    *,
    template: str,
    render: Literal["mustache", "format"],
    paths: Sequence[str],
    summaries: Mapping[str, Mapping[str, str]],
    diffs_a: Mapping[str, str],
    diffs_b: Mapping[str, str],
    encoder: Any,
    model_name: str,
    node: str,
    record: bool = True,
) -> FitReport:
    """Fit analyzer/planner prompts with per-file A/B summary and diff blocks.

    Summaries are preferred over diffs; A and B receive equal evidence caps.
    File labels (``path:``) are preserved on every retained/clipped block.
    """
    ordered_paths = sorted(
        set(paths) | set(summaries.keys()) | set(diffs_a.keys()) | set(diffs_b.keys())
    )
    blocks: list[EvidenceBlock] = []
    for path in ordered_paths:
        sa = str((summaries.get(path) or {}).get("summary_a", "") or "")
        sb = str((summaries.get(path) or {}).get("summary_b", "") or "")
        da = str(diffs_a.get(path, "") or "")
        db = str(diffs_b.get(path, "") or "")
        # Keep path labels outside clipped bodies so they survive encode/decode.
        blocks.append(
            EvidenceBlock(
                block_id=f"summary_a::{path}",
                text=sa,
                side="A",
                kind="summary",
                file_path=path,
                priority=10,
            )
        )
        blocks.append(
            EvidenceBlock(
                block_id=f"summary_b::{path}",
                text=sb,
                side="B",
                kind="summary",
                file_path=path,
                priority=10,
            )
        )
        blocks.append(
            EvidenceBlock(
                block_id=f"diff_a::{path}",
                text=da,
                side="A",
                kind="diff",
                file_path=path,
                priority=30,
            )
        )
        blocks.append(
            EvidenceBlock(
                block_id=f"diff_b::{path}",
                text=db,
                side="B",
                kind="diff",
                file_path=path,
                priority=30,
            )
        )

    budget = allowed_prompt_tokens(model_name, encoder=encoder)
    empty_vars = {"a_summary": "", "b_summary": "", "a_diff": "", "b_diff": ""}
    immutable_prompt = _render_vars(template, render, empty_vars)
    immutable_tokens = count_tokens(encoder, immutable_prompt)
    evidence_budget = max(0, budget - immutable_tokens - REPAIR_HEADROOM_TOKENS)
    tokens_before_evidence = sum(count_tokens(encoder, b.text or "") for b in blocks)

    working_budget = evidence_budget
    actions: list[BlockFitAction] = []
    variables = dict(empty_vars)
    prompt = immutable_prompt
    tokens_after = immutable_tokens
    for _ in range(8):
        targets = _allocate_targets(blocks, encoder=encoder, evidence_budget=working_budget)
        actions = _apply_targets(blocks, targets, encoder=encoder)
        by_id = {a.block_id: a.text for a in actions}

        def _join(side_kind: str) -> str:
            parts: list[str] = []
            for path in ordered_paths:
                body = by_id.get(f"{side_kind}::{path}", "")
                parts.append(f"{path}: {body}" if body else f"{path}:")
            return "\n\n".join(parts)

        variables = {
            "a_summary": _join("summary_a"),
            "b_summary": _join("summary_b"),
            "a_diff": _join("diff_a"),
            "b_diff": _join("diff_b"),
        }
        prompt = _render_vars(template, render, variables)
        tokens_after = count_tokens(encoder, prompt)
        if tokens_after <= budget or working_budget <= 0:
            break
        working_budget = max(0, working_budget - (tokens_after - budget) - 4)

    tokens_before = immutable_tokens + tokens_before_evidence
    was_clipped = any(a.action != "kept" for a in actions) or tokens_after < tokens_before
    report = FitReport(
        prompt=prompt,
        budget_tokens=budget,
        immutable_tokens=immutable_tokens,
        evidence_budget=evidence_budget,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        was_clipped=was_clipped,
        node=node,
        model_name=model_name,
        actions=actions,
        variables=variables,
    )
    if record:
        _record_fit_degradation(report)
    if was_clipped:
        logger.warning(
            "[prompt_budget] Structured global fit node=%s model=%s before=%d "
            "after=%d budget=%d files=%d",
            node,
            model_name,
            tokens_before,
            tokens_after,
            budget,
            len(ordered_paths),
        )
    return report


__all__ = [
    "BlockFitAction",
    "EvidenceBlock",
    "FitReport",
    "REPAIR_HEADROOM_TOKENS",
    "fit_global_ab_prompt",
    "fit_variable_blocks",
]
