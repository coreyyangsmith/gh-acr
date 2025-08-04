# src/agents/state.py
from typing import List, Dict, Optional, Literal, TypedDict
from dataclasses import dataclass

Status = Literal["start","planning","resolving","validate","success","fail","done"]

class ConflictFile(TypedDict):
    path: str
    base: str         # blob text
    left: str
    right: str
    proposal: Optional[str]  # resolver output
    exact_match: Optional[bool]
    similarity: Optional[float]
    compile_ok: Optional[bool]
    tests_passed: Optional[bool]

class ScenarioState(TypedDict, total=False):
    # Immutable inputs
    scenario_id: int
    repo_name: str
    p1: str
    p2: str
    merge_commit: str
    language: str
    conflict_paths: List[str]

    # Working data
    repo_dir: str
    conflicts: List[ConflictFile]
    planned_resolvers: List[str]
    resolver_index: int
    current_resolver: Optional[str]

    # Validation
    tree_match: Optional[bool]
    file_exact_rate: Optional[float]
    hunk_f1: Optional[float]
    build_ok: Optional[bool]

    # Control
    status: Status
    error: Optional[str]
    logs: List[str]
