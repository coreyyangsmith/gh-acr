# FM3 Role-Boundary Ambiguity Analysis

## Construct Mapping

The repository does not contain a direct `FM3` or `role-boundary ambiguity` label. This analysis treats RQ3 labels as pre-specified proxies: a narrow ambiguity/incompleteness proxy, a medium proxy that adds simplicity-vs-complexity tension, and a broad proxy that also includes structural/modification/orientation tension.

Prompt evidence supports the construct definition: the force-MIX planner asks for the simplest strategy and minimal change sets, the resolver is instructed to keep edits minimal and implement exactly the plan, while the reviewer checks partial merges and plan compliance.

## MIX Coverage

- Labeled RQ3 instances analyzed: 1078.
- Bypass7 MIX rows in result data: 6 unique merge instances and 9 file-level MIX traces.
- RQ3-labeled Bypass7 MIX overlap: 3 unique merge instances and 6 file-level MIX traces.
- Force-MIX evaluated instances available for secondary comparison: 90.

## Proxy Prevalence In MIX

- Narrow proxy prevalence: MIX 33.3% vs non-MIX 10.1%; Fisher p=0.2762.
- Medium proxy prevalence: MIX 66.7% vs non-MIX 28.5%; Fisher p=0.198.
- Broad proxy prevalence: MIX 100.0% vs non-MIX 63.0%; Fisher p=0.3007.

## Trace Evidence

No persisted `resolution*.txt`, `review*.txt`, `agent_plan.txt`, or `review_feedback_history.txt` artifacts for the MIX cases were found in this checkout.
The iteration/oscillation part of FM3 therefore remains a trace-level claim until archived outputs are located or the runs are regenerated with existing artifact persistence.

## Paper-Ready Interpretation

The RQ3 labels can empirically substantiate FM3 as a proxy-based, correlational claim: MIX cases can be audited for enrichment in ambiguity/incompleteness labels and simplicity-complexity tension, then linked to final correctness metrics. The strongest wording is that the labels are consistent with role-boundary ambiguity, while direct evidence for oscillation across all three iterations requires the saved role traces.

## Bypass7 MIX Case Audit

|           id | file_name                                                   |   exact_match |   similarity | difficulty   | project_size   | label_dataset   | fm3_narrow_proxy   | fm3_medium_proxy   | fm3_broad_proxy   | dominant_preference        |
|-------------:|:------------------------------------------------------------|--------------:|-------------:|:-------------|:---------------|:----------------|:-------------------|:-------------------|:------------------|:---------------------------|
|  15863986603 | torchrl/data/map/tree.py                                    |             0 |     0.857056 | easy         | large          | unlabeled       | NA                 | NA                 | NA                | NA                         |
| 112937595172 | src/crewai/llm.py                                           |             0 |     0.989009 | easy         | large          | unlabeled       | NA                 | NA                 | NA                | NA                         |
| 810893186987 | src/crewai/__init__.py                                      |             1 |     0.998392 | easy         | large          | rq3             | True               | True               | True              | favored complexity;neither |
| 492143690495 | llmfoundry/command_utils/data_prep/convert_delta_to_json.py |             0 |     0.994207 | medium       | large          | rq3_fail_only   | False              | False              | True              | NA                         |
| 442137388430 | openhands/agenthub/__init__.py                              |             0 |     0.942396 | medium       | medium         | unlabeled       | NA                 | NA                 | NA                | NA                         |
| 226595983323 | tests/runtime/conftest.py                                   |             0 |     0.819616 | hard         | medium         | rq3_fail_only   | False              | True               | True              | NA                         |
| 226595983323 | tests/runtime/test_bash.py                                  |             0 |     0.477185 | hard         | medium         | rq3_fail_only   | False              | True               | True              | NA                         |
| 226595983323 | tests/runtime/test_browsing.py                              |             0 |     0        | hard         | medium         | rq3_fail_only   | False              | True               | True              | NA                         |
| 226595983323 | tests/runtime/test_mcp_action.py                            |             0 |     0.936619 | hard         | medium         | rq3_fail_only   | False              | True               | True              | NA                         |

## Output Files

- `fm3_proxy_mapping.csv`
- `fm3_mix_label_table.csv`
- `fm3_proxy_prevalence.csv`
- `fm3_proxy_outcomes.csv`
- `fm3_mix_case_audit.csv`
- `fm3_trace_evidence.csv`
