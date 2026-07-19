"""Conditional trace replay for better_judge ablations.

Reuses a canonical ``better_judge`` run (same-run or existing on-disk artifacts)
so ablation methods only execute the stages that the ablation actually changes.

Snapshot layout (under ``<scenario>/better_judge/``)::

    replay_snapshot.json   # versioned, structured state for hydrators
    analyzer/, summarizer/, …  # legacy per-agent artifacts (fallback source)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import (
    agent_call_dir,
    file_path_to_slug,
    write_agent_call,
)
from .parse_utils import extract_analyzer_verdict, parse_plan_json

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1
SNAPSHOT_FILENAME = "replay_snapshot.json"
METHOD_REPLAY_META_FILENAME = "replay_metadata.json"

BJ_ABLATION_METHODS = frozenset(
    {"bj_no_summary", "bj_no_judge", "bj_no_plan", "bj_no_review"}
)
CANONICAL_SOURCE_METHOD = "better_judge"

# Node keys used for skip markers (match graph node / agent roles).
NODE_SUMMARISE = "summarise"
NODE_ANALYZE = "analyze"
NODE_PLAN = "plan"
NODE_PATCH = "patch"
NODE_REVIEW = "review"


@dataclass
class ReplayPlan:
    """Describes how an ablation should consume a canonical trace."""

    strategy: str
    reused_nodes: list[str] = field(default_factory=list)
    executed_nodes: list[str] = field(default_factory=list)
    skip_nodes: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    source_compatible: bool = True
    hydrate_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayProvenance:
    """Provenance recorded on artifacts / ledger / CSV rows."""

    enabled: bool = False
    source_method: str = CANONICAL_SOURCE_METHOD
    source_path: str | None = None
    source_model: str | None = None
    source_timestamp: str | None = None
    source_version: int | None = None
    strategy: str | None = None
    reused_nodes: list[str] = field(default_factory=list)
    executed_nodes: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    source_compatible: bool = True
    legacy_adapted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_ablation_method(eval_method: str | None) -> bool:
    return str(eval_method or "") in BJ_ABLATION_METHODS


def snapshot_path(source_root: Path | str) -> Path:
    return Path(source_root) / SNAPSHOT_FILENAME


def resolve_source_root(
    scenario_dir: Path | str,
    *,
    source_method: str = CANONICAL_SOURCE_METHOD,
) -> Path:
    return Path(scenario_dir) / source_method


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_decision(raw: str | None) -> str:
    text = str(raw or "MIX").upper().strip()
    if text in ("ALL_A", "A"):
        return "ALL_A"
    if text in ("ALL_B", "B"):
        return "ALL_B"
    return "MIX"


def _bypass_method_label(decision: str) -> str:
    if decision == "ALL_A":
        return "A"
    if decision == "ALL_B":
        return "B"
    return "MIX"


def build_snapshot_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build a versioned replay snapshot from a completed better_judge state."""
    decision = _normalize_decision(
        state.get("bypass_decision") or state.get("bypass_method")
    )
    summaries = state.get("summaries") or {}
    conflict_plan = state.get("conflict_plan") or {}
    resolved = state.get("resolved_contents") or {}
    final_diffs = state.get("final_diffs") or {}
    resolution_history = state.get("resolution_history") or {}

    attempt_1: dict[str, str] = {}
    for path, history in resolution_history.items():
        if isinstance(history, list) and history:
            attempt_1[str(path)] = str(history[0])
        elif path in resolved:
            # A/B bypass or single-pass: final is attempt 1
            attempt_1[str(path)] = str(resolved[path])
    if not attempt_1 and resolved:
        attempt_1 = {str(k): str(v) for k, v in resolved.items()}

    files = list(
        (state.get("sample_row") or {})
        .get("scenario_json", {})
        .get("files_in_merge_conflict")
        or list(summaries.keys())
        or list(resolved.keys())
    )

    review_iters = int(state.get("_review_iter", 0) or 0)
    # If review loop ran, attempt count is review_iters+1 for final patch;
    # attempt_1 is always the first resolver pass.
    max_attempts = 1
    for hist in resolution_history.values():
        if isinstance(hist, list):
            max_attempts = max(max_attempts, len(hist))

    return {
        "version": SNAPSHOT_VERSION,
        "source_method": CANONICAL_SOURCE_METHOD,
        "scenario_id": state.get("scenario_id"),
        "model_name": state.get("model_name"),
        "timestamp": _utc_now(),
        "bypass_decision": decision,
        "bypass_method": _bypass_method_label(decision),
        "bypass_analyzer_output": state.get("bypass_analyzer_output") or "",
        "summaries": {
            str(p): {
                "summary_a": str((s or {}).get("summary_a", "")),
                "summary_b": str((s or {}).get("summary_b", "")),
            }
            for p, s in summaries.items()
        },
        "conflict_plan": {str(k): str(v) for k, v in conflict_plan.items()}
        if conflict_plan
        else {},
        "resolver_attempt_1": attempt_1,
        "resolved_contents": {str(k): str(v) for k, v in resolved.items()},
        "final_diffs": {str(k): str(v) for k, v in final_diffs.items()},
        "review_iters": review_iters,
        "resolver_attempts": max_attempts,
        "files": [str(f) for f in files],
    }


def write_snapshot(source_root: Path | str, snapshot: Mapping[str, Any]) -> Path:
    """Persist ``replay_snapshot.json`` under the canonical method root."""
    root = Path(source_root)
    root.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(root)
    path.write_text(
        json.dumps(dict(snapshot), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "[trace_replay] wrote snapshot scenario=%s decision=%s → %s",
        snapshot.get("scenario_id"),
        snapshot.get("bypass_decision"),
        path,
    )
    return path


def save_snapshot_from_state(
    state: Mapping[str, Any],
    *,
    source_root: Path | str | None = None,
) -> Path | None:
    """Write a snapshot after a successful better_judge run."""
    root = source_root
    if root is None:
        raw = state.get("artifact_root")
        if not raw:
            return None
        root = Path(str(raw))
    snap = build_snapshot_from_state(state)
    return write_snapshot(root, snap)


def _read_text(path: Path) -> str | None:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("[trace_replay] failed reading %s: %s", path, exc)
    return None


def _list_attempt_dirs(parent: Path) -> list[int]:
    if not parent.is_dir():
        return []
    attempts: list[int] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("attempt_"):
            try:
                attempts.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(attempts)


def adapt_legacy_artifacts(
    source_root: Path | str,
    *,
    files: list[str] | None = None,
    model_name: str | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a snapshot dict from existing per-agent artifact directories.

    Returns None when the tree is too incomplete to be useful.
    """
    root = Path(source_root)
    if not root.is_dir():
        return None

    # Discover files from summarizer / final / resolver dirs if not provided
    discovered: list[str] = list(files or [])
    if not discovered:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name in {"analyzer", "planner"}:
                continue
            # file_slug directories contain summarizer/resolver/final
            if (child / "summarizer").is_dir() or (child / "final").is_dir():
                # We only have slugs; prefer metadata file_path when present
                meta_path = child / "summarizer" / "a" / "metadata.json"
                fp = None
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        fp = meta.get("file_path")
                    except (OSError, json.JSONDecodeError):
                        fp = None
                discovered.append(str(fp) if fp else child.name)

    summaries: dict[str, dict[str, str]] = {}
    for path in discovered:
        slug = file_path_to_slug(path)
        a_txt = _read_text(root / slug / "summarizer" / "a" / "output.txt")
        b_txt = _read_text(root / slug / "summarizer" / "b" / "output.txt")
        if a_txt is None and b_txt is None:
            continue
        summaries[path] = {
            "summary_a": a_txt or "",
            "summary_b": b_txt or "",
        }

    analyzer_raw = _read_text(root / "analyzer" / "output.txt") or ""
    decision = "MIX"
    if analyzer_raw.strip():
        parsed, _ = extract_analyzer_verdict(analyzer_raw)
        if parsed:
            decision = parsed
    else:
        # Infer A/B bypass from bypass/ artifacts without MIX planner
        bypass_dirs = [
            d
            for d in root.iterdir()
            if d.is_dir() and (d / "bypass").is_dir()
        ]
        planner_out = _read_text(root / "planner" / "output.txt")
        if bypass_dirs and not planner_out:
            # Ambiguous A vs B — try decision.txt
            for d in bypass_dirs:
                dec = _read_text(d / "bypass" / "artifacts" / "decision.txt")
                if dec:
                    decision = _normalize_decision(dec)
                    break

    conflict_plan: dict[str, str] = {}
    planner_raw = _read_text(root / "planner" / "output.txt")
    if planner_raw and planner_raw.strip():
        try:
            conflict_plan = parse_plan_json(
                planner_raw,
                expected_paths=set(discovered) if discovered else None,
            )
        except Exception:
            try:
                conflict_plan = {
                    str(k): str(v)
                    for k, v in json.loads(planner_raw).items()
                }
            except Exception:
                conflict_plan = {}

    attempt_1: dict[str, str] = {}
    resolved: dict[str, str] = {}
    final_diffs: dict[str, str] = {}
    max_attempts = 1
    for path in discovered or list(summaries.keys()):
        slug = file_path_to_slug(path)
        attempts = _list_attempt_dirs(root / slug / "resolver")
        if attempts:
            max_attempts = max(max_attempts, max(attempts))
            a1 = _read_text(root / slug / "resolver" / "attempt_1" / "output.txt")
            if a1 is not None:
                attempt_1[path] = a1
        final_txt = _read_text(root / slug / "final" / "resolved.txt")
        if final_txt is not None:
            resolved[path] = final_txt
        else:
            bypass_txt = _read_text(root / slug / "bypass" / "output.txt")
            if bypass_txt is not None:
                resolved[path] = bypass_txt
                attempt_1.setdefault(path, bypass_txt)
        fd = _read_text(root / slug / "final" / "final_diff.txt")
        if fd is not None:
            final_diffs[path] = fd

    # Require at least analyzer or summaries or finals to be useful
    if not summaries and not analyzer_raw and not resolved:
        return None

    file_list = discovered or list(
        set(summaries) | set(resolved) | set(attempt_1)
    )
    return {
        "version": SNAPSHOT_VERSION,
        "source_method": CANONICAL_SOURCE_METHOD,
        "scenario_id": scenario_id,
        "model_name": model_name,
        "timestamp": _utc_now(),
        "legacy_adapted": True,
        "bypass_decision": decision,
        "bypass_method": _bypass_method_label(decision),
        "bypass_analyzer_output": analyzer_raw,
        "summaries": summaries,
        "conflict_plan": conflict_plan,
        "resolver_attempt_1": attempt_1,
        "resolved_contents": resolved,
        "final_diffs": final_diffs,
        "review_iters": max(0, max_attempts - 1),
        "resolver_attempts": max_attempts,
        "files": file_list,
    }


def load_snapshot(
    source_root: Path | str,
    *,
    files: list[str] | None = None,
    model_name: str | None = None,
    scenario_id: str | None = None,
    expected_model: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Load a snapshot or adapt legacy artifacts.

    Returns ``(snapshot, error_reason, legacy_adapted)``.
    """
    root = Path(source_root)
    if not root.is_dir():
        return None, "missing_source_root", False

    path = snapshot_path(root)
    if path.is_file():
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"corrupt_snapshot:{exc}", False
        if int(snap.get("version") or 0) != SNAPSHOT_VERSION:
            return None, f"unsupported_snapshot_version:{snap.get('version')}", False
        if expected_model and snap.get("model_name") and expected_model != snap.get(
            "model_name"
        ):
            return None, "model_mismatch", False
        return snap, None, bool(snap.get("legacy_adapted", False))

    adapted = adapt_legacy_artifacts(
        root,
        files=files,
        model_name=model_name or expected_model,
        scenario_id=scenario_id,
    )
    if adapted is None:
        return None, "incomplete_legacy_artifacts", False
    if expected_model and adapted.get("model_name") and expected_model != adapted.get(
        "model_name"
    ):
        # Legacy often lacks model; only fail when both present and differ
        return None, "model_mismatch", True
    return adapted, None, True


def plan_ablation_replay(
    eval_method: str,
    snapshot: Mapping[str, Any] | None,
    *,
    load_error: str | None = None,
) -> ReplayPlan:
    """Decide reuse vs live execution for one ablation method."""
    if not is_ablation_method(eval_method):
        return ReplayPlan(
            strategy="not_applicable",
            executed_nodes=["full_graph"],
            source_compatible=False,
            fallback_reason="not_an_ablation",
        )

    if snapshot is None:
        return ReplayPlan(
            strategy="live_fallback",
            executed_nodes=["full_graph"],
            source_compatible=False,
            fallback_reason=load_error or "missing_snapshot",
        )

    decision = _normalize_decision(snapshot.get("bypass_decision"))
    has_summaries = bool(snapshot.get("summaries"))
    has_attempt_1 = bool(snapshot.get("resolver_attempt_1"))
    has_resolved = bool(snapshot.get("resolved_contents"))
    has_plan = bool(snapshot.get("conflict_plan"))

    if eval_method == "bj_no_summary":
        # Raw-diff summaries change analyzer inputs; never reuse downstream.
        return ReplayPlan(
            strategy="no_reuse",
            reused_nodes=[],
            executed_nodes=[
                "seed_raw_summaries",
                NODE_ANALYZE,
                NODE_PLAN,
                NODE_PATCH,
                NODE_REVIEW,
                "finalize",
            ],
            skip_nodes=[],
            hydrate_keys=[],
            source_compatible=True,
            fallback_reason=None,
        )

    if eval_method == "bj_no_judge":
        if decision == "MIX" and has_resolved and has_summaries:
            return ReplayPlan(
                strategy="full_suffix_reuse",
                reused_nodes=[
                    NODE_SUMMARISE,
                    NODE_ANALYZE,
                    NODE_PLAN,
                    NODE_PATCH,
                    NODE_REVIEW,
                ],
                executed_nodes=["finalize"],
                skip_nodes=[
                    NODE_SUMMARISE,
                    "force_mix_marker",
                    NODE_PLAN,
                    NODE_PATCH,
                    NODE_REVIEW,
                    "feedback",
                ],
                hydrate_keys=[
                    "summaries",
                    "bypass",
                    "conflict_plan",
                    "resolved_contents",
                    "final_diffs",
                ],
            )
        if not has_summaries:
            return ReplayPlan(
                strategy="live_fallback",
                executed_nodes=["full_graph"],
                source_compatible=False,
                fallback_reason="missing_summaries_for_forced_mix",
            )
        return ReplayPlan(
            strategy="reuse_summaries_run_mix",
            reused_nodes=[NODE_SUMMARISE],
            executed_nodes=[
                "force_mix_marker",
                NODE_PLAN,
                NODE_PATCH,
                NODE_REVIEW,
                "finalize",
            ],
            skip_nodes=[NODE_SUMMARISE],
            hydrate_keys=["summaries"],
        )

    if eval_method == "bj_no_plan":
        if decision in ("ALL_A", "ALL_B") and has_resolved:
            return ReplayPlan(
                strategy="reuse_bypass",
                reused_nodes=[NODE_SUMMARISE, NODE_ANALYZE, "bypass"],
                executed_nodes=["finalize"],
                skip_nodes=[
                    NODE_SUMMARISE,
                    NODE_ANALYZE,
                    "all_merge_plan",
                    NODE_PATCH,
                ],
                hydrate_keys=[
                    "summaries",
                    "bypass",
                    "resolved_contents",
                    "final_diffs",
                ],
            )
        if not has_summaries:
            return ReplayPlan(
                strategy="live_fallback",
                executed_nodes=["full_graph"],
                source_compatible=False,
                fallback_reason="missing_summaries_for_no_plan",
            )
        return ReplayPlan(
            strategy="reuse_prefix_run_resolver",
            reused_nodes=[NODE_SUMMARISE, NODE_ANALYZE],
            executed_nodes=["all_merge_plan", NODE_PATCH, "finalize"],
            skip_nodes=[NODE_SUMMARISE, NODE_ANALYZE],
            hydrate_keys=["summaries", "bypass"],
        )

    if eval_method == "bj_no_review":
        if decision in ("ALL_A", "ALL_B") and has_resolved:
            return ReplayPlan(
                strategy="reuse_bypass",
                reused_nodes=[NODE_SUMMARISE, NODE_ANALYZE, "bypass"],
                executed_nodes=["finalize"],
                skip_nodes=[
                    NODE_SUMMARISE,
                    NODE_ANALYZE,
                    NODE_PLAN,
                    NODE_PATCH,
                ],
                hydrate_keys=[
                    "summaries",
                    "bypass",
                    "resolved_contents",
                    "final_diffs",
                ],
            )
        # MIX: reuse first resolver attempt only (never post-review final)
        if not (has_summaries and has_plan and has_attempt_1):
            missing = []
            if not has_summaries:
                missing.append("summaries")
            if not has_plan:
                missing.append("conflict_plan")
            if not has_attempt_1:
                missing.append("resolver_attempt_1")
            return ReplayPlan(
                strategy="live_fallback",
                executed_nodes=["full_graph"],
                source_compatible=False,
                fallback_reason="missing_" + ",".join(missing),
            )
        return ReplayPlan(
            strategy="reuse_first_resolver",
            reused_nodes=[NODE_SUMMARISE, NODE_ANALYZE, NODE_PLAN, NODE_PATCH],
            executed_nodes=["finalize"],
            skip_nodes=[NODE_SUMMARISE, NODE_ANALYZE, NODE_PLAN, NODE_PATCH],
            hydrate_keys=[
                "summaries",
                "bypass",
                "conflict_plan",
                "resolver_attempt_1",
            ],
        )

    return ReplayPlan(
        strategy="live_fallback",
        executed_nodes=["full_graph"],
        source_compatible=False,
        fallback_reason=f"unknown_method:{eval_method}",
    )


def hydrate_state_from_snapshot(
    state: dict[str, Any],
    snapshot: Mapping[str, Any],
    plan: ReplayPlan,
) -> dict[str, Any]:
    """Inject snapshot fields listed in ``plan.hydrate_keys`` into state."""
    keys = set(plan.hydrate_keys)
    if "summaries" in keys and snapshot.get("summaries"):
        state["summaries"] = {
            str(p): {
                "summary_a": str((s or {}).get("summary_a", "")),
                "summary_b": str((s or {}).get("summary_b", "")),
            }
            for p, s in (snapshot.get("summaries") or {}).items()
        }
        state["status"] = "summarised"

    if "bypass" in keys:
        decision = _normalize_decision(snapshot.get("bypass_decision"))
        state["bypass_decision"] = decision
        state["bypass_method"] = _bypass_method_label(decision)
        state["bypass_analyzer_output"] = snapshot.get("bypass_analyzer_output") or ""
        state["status"] = "analyzed"

    if "conflict_plan" in keys and snapshot.get("conflict_plan"):
        state["conflict_plan"] = {
            str(k): str(v) for k, v in (snapshot.get("conflict_plan") or {}).items()
        }
        state["status"] = "planned"

    if "resolver_attempt_1" in keys:
        attempt_1 = snapshot.get("resolver_attempt_1") or {}
        state["resolved_contents"] = {str(k): str(v) for k, v in attempt_1.items()}
        state["resolution_history"] = {
            str(k): [str(v)] for k, v in attempt_1.items()
        }
        state["_review_iter"] = 0
        state["status"] = "resolved_multi"
        # Clear any review feedback so finalize uses attempt_1 as-is
        state["review_feedback"] = {}
        state["review_results"] = {}

    if "resolved_contents" in keys and snapshot.get("resolved_contents"):
        state["resolved_contents"] = {
            str(k): str(v)
            for k, v in (snapshot.get("resolved_contents") or {}).items()
        }
        state["status"] = "resolved_multi"

    if "final_diffs" in keys and snapshot.get("final_diffs"):
        state["final_diffs"] = {
            str(k): str(v) for k, v in (snapshot.get("final_diffs") or {}).items()
        }

    state["_replay_skip_nodes"] = list(plan.skip_nodes)
    return state


def apply_trace_replay(state: dict[str, Any]) -> tuple[dict[str, Any], ReplayProvenance]:
    """Load snapshot, plan, and hydrate state for an ablation run.

    When replay is disabled or the method is not an ablation, returns the
    state unchanged with ``enabled=False`` provenance.
    """
    cfg = state.get("trace_replay") or {}
    enabled = bool(cfg.get("enabled"))
    eval_method = str(state.get("eval_method") or "")
    provenance = ReplayProvenance(enabled=enabled and is_ablation_method(eval_method))

    if not enabled or not is_ablation_method(eval_method):
        return state, provenance

    source_method = str(cfg.get("source_method") or CANONICAL_SOURCE_METHOD)
    source_root = cfg.get("source_root")
    if not source_root:
        # Derive from artifact_root sibling
        art = state.get("artifact_root")
        if art:
            source_root = str(Path(str(art)).parent / source_method)
    provenance.source_method = source_method
    provenance.source_path = str(source_root) if source_root else None

    files = list(
        (state.get("sample_row") or {})
        .get("scenario_json", {})
        .get("files_in_merge_conflict")
        or []
    )
    expected_model = state.get("model_name")
    snap, err, legacy = load_snapshot(
        source_root or "",
        files=files,
        model_name=expected_model,
        scenario_id=state.get("scenario_id"),
        expected_model=expected_model,
    )
    provenance.legacy_adapted = legacy
    if snap:
        provenance.source_model = snap.get("model_name")
        provenance.source_timestamp = snap.get("timestamp")
        provenance.source_version = int(snap.get("version") or SNAPSHOT_VERSION)

    plan = plan_ablation_replay(eval_method, snap, load_error=err)
    provenance.strategy = plan.strategy
    provenance.reused_nodes = list(plan.reused_nodes)
    provenance.executed_nodes = list(plan.executed_nodes)
    provenance.fallback_reason = plan.fallback_reason
    provenance.source_compatible = plan.source_compatible

    if plan.strategy in ("live_fallback", "no_reuse", "not_applicable"):
        # no_reuse still runs the ablation graph normally (seed_raw_summaries etc.)
        state["trace_replay_provenance"] = provenance.to_dict()
        state["_replay_skip_nodes"] = []
        logger.info(
            "[trace_replay] scenario=%s method=%s strategy=%s fallback=%s",
            state.get("scenario_id"),
            eval_method,
            plan.strategy,
            plan.fallback_reason,
        )
        return state, provenance

    assert snap is not None
    hydrate_state_from_snapshot(state, snap, plan)
    state["trace_replay_provenance"] = provenance.to_dict()
    logger.info(
        "[trace_replay] scenario=%s method=%s strategy=%s reused=%s executed=%s",
        state.get("scenario_id"),
        eval_method,
        plan.strategy,
        plan.reused_nodes,
        plan.executed_nodes,
    )
    return state, provenance


def write_method_replay_metadata(
    artifact_root: Path | str | None,
    provenance: ReplayProvenance | Mapping[str, Any],
) -> Path | None:
    """Write ``replay_metadata.json`` under the ablation method artifact root."""
    if artifact_root is None:
        return None
    try:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        payload = (
            provenance.to_dict()
            if isinstance(provenance, ReplayProvenance)
            else dict(provenance)
        )
        payload.setdefault("timestamp", _utc_now())
        path = root / METHOD_REPLAY_META_FILENAME
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
    except OSError as exc:
        logger.warning(
            "[trace_replay] failed writing method replay metadata under %s: %s",
            artifact_root,
            exc,
        )
        return None


def should_skip_node(state: Mapping[str, Any], node_key: str) -> bool:
    """Return True when a graph node should be skipped (already hydrated)."""
    skip = state.get("_replay_skip_nodes") or set()
    if isinstance(skip, (list, tuple)):
        skip = set(skip)
    return node_key in skip


def wrap_node_with_replay_skip(node_fn, node_key: str):
    """Wrap a LangGraph node so hydrated stages are no-ops under replay."""

    def _wrapped(state: dict[str, Any]) -> dict[str, Any]:
        if should_skip_node(state, node_key):
            logger.info(
                "[trace_replay] skip node=%s scenario=%s method=%s",
                node_key,
                state.get("scenario_id"),
                state.get("eval_method"),
            )
            # Mirror stub status labels so downstream routing still works
            if node_key == NODE_SUMMARISE and state.get("summaries") is not None:
                state["status"] = "summarised"
            elif node_key == NODE_ANALYZE and state.get("bypass_decision"):
                state["status"] = "analyzed"
            elif node_key == NODE_PLAN and state.get("conflict_plan") is not None:
                state["status"] = "planned"
            elif node_key == NODE_PATCH and state.get("resolved_contents") is not None:
                state["status"] = "resolved_multi"
            elif node_key == NODE_REVIEW:
                # Ensure _route_after_review returns "finish" (not "retry").
                resolved = state.get("resolved_contents") or {}
                accept = {
                    path: {"outcome": "ACCEPT", "rationale": "trace_replay_skip"}
                    for path in resolved
                }
                state["review_results"] = accept
                state.setdefault("reviews", {})
                state["_review_iter"] = max(int(state.get("_review_iter", 0) or 0), 2)
                state["status"] = "reviewed"
            return state
        return node_fn(state)

    _wrapped.__name__ = getattr(node_fn, "__name__", f"replay_skip_{node_key}")
    return _wrapped


def audit_replayed_call(
    state: Mapping[str, Any],
    *,
    agent: str,
    node: str,
    file_path: str | None = None,
    call_id: str | None = None,
    output_text: str = "",
    note: str = "hydrated from canonical better_judge trace",
) -> None:
    """Optionally write a non-LLM audit artifact for a reused stage."""
    from .artifact_io import base_metadata, get_artifact_root

    root = get_artifact_root(state)
    if root is None:
        return
    slug = file_path_to_slug(file_path) if file_path else None
    call_dir = agent_call_dir(root, agent=agent, file_slug=slug, call_id=call_id)
    prov = state.get("trace_replay_provenance") or {}
    write_agent_call(
        call_dir,
        input_text="",
        output_text=output_text,
        artifacts={"note.txt": note},
        metadata=base_metadata(
            agent=agent,
            node=node,
            state=state,
            file_path=file_path,
            call_id=call_id,
            llm_used=False,
            extra={
                "reason": "trace_replay",
                "replayed_from": prov.get("source_method") or CANONICAL_SOURCE_METHOD,
                "trace_replay_strategy": prov.get("strategy"),
            },
        ),
    )


__all__ = [
    "BJ_ABLATION_METHODS",
    "CANONICAL_SOURCE_METHOD",
    "METHOD_REPLAY_META_FILENAME",
    "NODE_ANALYZE",
    "NODE_PATCH",
    "NODE_PLAN",
    "NODE_REVIEW",
    "NODE_SUMMARISE",
    "ReplayPlan",
    "ReplayProvenance",
    "SNAPSHOT_FILENAME",
    "SNAPSHOT_VERSION",
    "adapt_legacy_artifacts",
    "apply_trace_replay",
    "audit_replayed_call",
    "build_snapshot_from_state",
    "hydrate_state_from_snapshot",
    "is_ablation_method",
    "load_snapshot",
    "plan_ablation_replay",
    "resolve_source_root",
    "save_snapshot_from_state",
    "should_skip_node",
    "snapshot_path",
    "wrap_node_with_replay_skip",
    "write_method_replay_metadata",
    "write_snapshot",
]
