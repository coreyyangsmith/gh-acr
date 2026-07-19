# Agent raw_model_outputs verification

Benchmark: `data\git_good_bench_merge_commits_all.csv` (1909 scenarios).
Scope: `eval_method=agent` only (bypass7 ignored).

## Per-model summary

### gpt-5-nano (`openai_gpt-5-nano`)

- Scenario dirs: **1898** / 1909
- Agent dirs: **1898** / 1909
- Fully OK (artifacts + results): **1865**
- Gap rows: **44** (needs reprocess: **42**)
- File-level: ok=3442, missing=1106, empty=54, invalid=0 (checked=3496)
- Avg OK output size: 22711.2 bytes
- Results CSV unique agent IDs: 1896
- Categories: `{"complete_ok": 1865, "empty_output": 31, "not_processed": 11, "missing_results": 2}`
- Outputs: `data\agent_coverage_gaps\gpt-5-nano_agent_missing_or_failed.csv`, `data\agent_coverage_gaps\gpt-5-nano_agent_needs_reprocess.csv`

### llama-3.1-8b (`llama-3.1-8b`)

- Scenario dirs: **1842** / 1909
- Agent dirs: **1795** / 1909
- Fully OK (artifacts + results): **1788**
- Gap rows: **121** (needs reprocess: **114**)
- File-level: ok=3215, missing=1387, empty=0, invalid=0 (checked=3215)
- Avg OK output size: 12145.1 bytes
- Results CSV unique agent IDs: 1897
- Categories: `{"complete_ok": 1788, "not_processed": 114, "missing_results": 7}`
- Outputs: `data\agent_coverage_gaps\llama-3.1-8b_agent_missing_or_failed.csv`, `data\agent_coverage_gaps\llama-3.1-8b_agent_needs_reprocess.csv`

### qwen3-32b (`groq_qwen_qwen3-32b`)

- Scenario dirs: **1905** / 1909
- Agent dirs: **1905** / 1909
- Fully OK (artifacts + results): **1899**
- Gap rows: **10** (needs reprocess: **4**)
- File-level: ok=3517, missing=1085, empty=0, invalid=0 (checked=3517)
- Avg OK output size: 30592.5 bytes
- Results CSV unique agent IDs: 1899
- Categories: `{"complete_ok": 1899, "missing_results": 6, "not_processed": 4}`
- Outputs: `data\agent_coverage_gaps\qwen3-32b_agent_missing_or_failed.csv`, `data\agent_coverage_gaps\qwen3-32b_agent_needs_reprocess.csv`

## Legitimacy checks applied

1. Every benchmark scenario ID should have `raw_model_outputs/<model>/<id>/agent/`.
2. Each `files_in_merge_conflict` entry should map to a non-empty agent output (`<slug>.txt` or `<slug>/final/resolved.txt`).
3. Reject 0-byte / tiny outputs and whole-file API/traceback payloads.
4. Do **not** flag ordinary source that merely contains words like `failed to` or `Exception:` (common false positives).
5. Cross-check presence in the corresponding combined results CSV (`eval_method=agent`).
