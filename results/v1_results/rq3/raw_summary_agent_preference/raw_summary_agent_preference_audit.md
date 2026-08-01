# Raw Summary Agent Preference Audit

## Coverage

- Raw file-level summary pairs parsed: 9,866.
- Raw instance-level rows after aggregation: 5,625.
- Failures-only file-level rows with joined EM failure metadata: 9,106.
- Failures-only instance-level rows with joined EM failure metadata: 5,052.
- Parse failure file rows: 1,735.
- Models: groq_qwen_qwen3-32b, llama-3.1-8b, openai_gpt-5-nano.
- Methods: bypass7.

## Metadata Coverage

| grain          |   rows |   rows_with_selected_parent |   selected_parent_coverage_pct |   rows_with_exact_match_metadata |   exact_match_metadata_coverage_pct |   failure_rows |   non_failure_rows_with_metadata |
|:---------------|-------:|----------------------------:|-------------------------------:|---------------------------------:|------------------------------------:|---------------:|---------------------------------:|
| file_level     |   9866 |                        9854 |                        99.8784 |                             9863 |                             99.9696 |           9106 |                              757 |
| instance_level |   5625 |                        5618 |                        99.8756 |                             5624 |                             99.9822 |           5052 |                              572 |

## Derived Preference Labels: All Raw Instances

- `favored_simplicity`: 2,027 (36.0%)
- `favored_complexity`: 1,552 (27.6%)
- `tie_ambiguous`: 1,265 (22.5%)
- `unparseable`: 774 (13.8%)
- `mix_or_missing_selection`: 7 (0.1%)

## Derived Preference Labels: Failures Only

- `favored_simplicity`: 1,938 (38.4%)
- `favored_complexity`: 1,299 (25.7%)
- `tie_ambiguous`: 1,101 (21.8%)
- `unparseable`: 709 (14.0%)
- `mix_or_missing_selection`: 5 (0.1%)

## All vs Failures Comparison

| scope         | grain          |   rows |   parse_ok_rows |   parse_ok_pct |   selected_parent_A |   selected_parent_B |   selected_parent_missing_or_other |   favored_simplicity |   favored_complexity |   tie_ambiguous |   unparseable |   mix_or_missing_selection |   mean_a_total_changes |   mean_b_total_changes |   mean_selected_minus_rejected_changes |
|:--------------|:---------------|-------:|----------------:|---------------:|--------------------:|--------------------:|-----------------------------------:|---------------------:|---------------------:|----------------:|--------------:|---------------------------:|-----------------------:|-----------------------:|---------------------------------------:|
| all           | file_level     |   9866 |            8131 |        82.4144 |                7072 |                2782 |                                 12 |                 3628 |                 2905 |            1586 |          1735 |                         12 |                7.71285 |                6.11585 |                              0.0456667 |
| failures_only | file_level     |   9106 |            7487 |        82.2205 |                6538 |                2560 |                                  8 |                 3503 |                 2572 |            1404 |          1619 |                          8 |                7.66857 |                6.31485 |                             -0.380413  |
| all           | instance_level |   5625 |            4840 |        86.0444 |                3931 |                1687 |                                  7 |                 2027 |                 1552 |            1265 |           774 |                          7 |               13.528   |               10.7269  |                              0.0800997 |
| failures_only | instance_level |   5052 |            4333 |        85.768  |                3516 |                1531 |                                  5 |                 1938 |                 1299 |            1101 |           709 |                          5 |               13.8222  |               11.3822  |                             -0.685754  |

## Model-Level Breakdown

| scope         | raw_model_dir       |   rows |   parse_ok_rows |   parse_ok_pct |   selected_parent_A |   selected_parent_B |   favored_simplicity |   favored_complexity |   tie_ambiguous |   unparseable |   mix_or_missing_selection |   mean_selected_minus_rejected_changes |
|:--------------|:--------------------|-------:|----------------:|---------------:|--------------------:|--------------------:|---------------------:|---------------------:|----------------:|--------------:|---------------------------:|---------------------------------------:|
| all           | groq_qwen_qwen3-32b |   1899 |            1880 |        98.9995 |                1895 |                   4 |                  699 |                  722 |             470 |             8 |                          0 |                              0.179042  |
| all           | llama-3.1-8b        |   1830 |            1064 |        58.1421 |                1107 |                 722 |                  400 |                  457 |             206 |           766 |                          1 |                              3.16949   |
| all           | openai_gpt-5-nano   |   1896 |            1896 |       100      |                 929 |                 961 |                  928 |                  373 |             589 |             0 |                          6 |                             -3.00899   |
| failures_only | groq_qwen_qwen3-32b |   1684 |            1666 |        98.9311 |                1681 |                   3 |                  669 |                  594 |             413 |             8 |                          0 |                             -0.0682898 |
| failures_only | llama-3.1-8b        |   1604 |             903 |        56.2968 |                 973 |                 631 |                  364 |                  369 |             170 |           701 |                          0 |                              1.51122   |
| failures_only | openai_gpt-5-nano   |   1764 |            1764 |       100      |                 862 |                 897 |                  905 |                  336 |             518 |             0 |                          5 |                             -3.28027   |

## Selected vs Rejected Change Types

Negative values mean the pipeline-selected parent had fewer summarized entries of that type than the rejected parent; positive values mean the selected parent had more.

### Most Under-Represented In Selected Parent (All Instances)

| scope   | grain          | change_type   |   selected_parent_count |   rejected_parent_count |   selected_minus_rejected |
|:--------|:---------------|:--------------|------------------------:|------------------------:|--------------------------:|
| all     | instance_level | modification  |                   38163 |                   44331 |                     -6168 |
| all     | instance_level | removal       |                    2633 |                    2714 |                       -81 |
| all     | instance_level | likely intent |                      44 |                      75 |                       -31 |
| all     | instance_level | deprecation   |                       0 |                       7 |                        -7 |
| all     | instance_level | renaming      |                       2 |                       8 |                        -6 |
| all     | instance_level | relocation    |                       0 |                       6 |                        -6 |
| all     | instance_level | change        |                       0 |                       5 |                        -5 |
| all     | instance_level | reordering    |                       1 |                       6 |                        -5 |

### Most Over-Represented In Selected Parent (All Instances)

| scope   | grain          | change_type    |   selected_parent_count |   rejected_parent_count |   selected_minus_rejected |
|:--------|:---------------|:---------------|------------------------:|------------------------:|--------------------------:|
| all     | instance_level | addition       |                   25602 |                   18827 |                      6775 |
| all     | instance_level | unknown        |                    1896 |                    1875 |                        21 |
| all     | instance_level | replacement    |                       9 |                       5 |                         4 |
| all     | instance_level | fix            |                       3 |                       0 |                         3 |
| all     | instance_level | exception      |                       1 |                       0 |                         1 |
| all     | instance_level | function       |                       1 |                       0 |                         1 |
| all     | instance_level | reorganization |                       1 |                       0 |                         1 |
| all     | instance_level | summary        |                       2 |                       1 |                         1 |

### Most Under-Represented In Selected Parent (Failures Only)

| scope         | grain          | change_type   |   selected_parent_count |   rejected_parent_count |   selected_minus_rejected |
|:--------------|:---------------|:--------------|------------------------:|------------------------:|--------------------------:|
| failures_only | instance_level | modification  |                   34570 |                   43089 |                     -8519 |
| failures_only | instance_level | removal       |                    2348 |                    2492 |                      -144 |
| failures_only | instance_level | likely intent |                      39 |                      69 |                       -30 |
| failures_only | instance_level | deprecation   |                       0 |                       7 |                        -7 |
| failures_only | instance_level | renaming      |                       2 |                       8 |                        -6 |
| failures_only | instance_level | relocation    |                       0 |                       6 |                        -6 |
| failures_only | instance_level | change        |                       0 |                       5 |                        -5 |
| failures_only | instance_level | reordering    |                       1 |                       6 |                        -5 |

### Most Over-Represented In Selected Parent (Failures Only)

| scope         | grain          | change_type    |   selected_parent_count |   rejected_parent_count |   selected_minus_rejected |
|:--------------|:---------------|:---------------|------------------------:|------------------------:|--------------------------:|
| failures_only | instance_level | addition       |                   23137 |                   17852 |                      5285 |
| failures_only | instance_level | unknown        |                    1761 |                    1747 |                        14 |
| failures_only | instance_level | replacement    |                       9 |                       5 |                         4 |
| failures_only | instance_level | fix            |                       3 |                       0 |                         3 |
| failures_only | instance_level | exception      |                       1 |                       0 |                         1 |
| failures_only | instance_level | function       |                       1 |                       0 |                         1 |
| failures_only | instance_level | reorganization |                       1 |                       0 |                         1 |
| failures_only | instance_level | summary        |                       2 |                       1 |                         1 |

## Raw A/B Change-Type Distribution

| scope   | grain      | change_type          |   a_parent_count |   b_parent_count |   a_minus_b |   total_count |
|:--------|:-----------|:---------------------|-----------------:|-----------------:|------------:|--------------:|
| all     | file_level | modification         |            48039 |            34544 |       13495 |         82583 |
| all     | file_level | addition             |            23396 |            21049 |        2347 |         44445 |
| all     | file_level | removal              |             2598 |             2754 |        -156 |          5352 |
| all     | file_level | unknown              |             1918 |             1853 |          65 |          3771 |
| all     | file_level | likely intent        |               61 |               58 |           3 |           119 |
| all     | file_level | refactor             |               12 |                9 |           3 |            21 |
| all     | file_level | replacement          |               11 |                3 |           8 |            14 |
| all     | file_level | renaming             |                2 |                8 |          -6 |            10 |
| all     | file_level | import               |                5 |                4 |           1 |             9 |
| all     | file_level | deprecation          |                0 |                7 |          -7 |             7 |
| all     | file_level | intent               |                6 |                1 |           5 |             7 |
| all     | file_level | note                 |                3 |                4 |          -1 |             7 |
| all     | file_level | reordering           |                1 |                6 |          -5 |             7 |
| all     | file_level | documentation        |                3 |                3 |           0 |             6 |
| all     | file_level | relocation           |                0 |                6 |          -6 |             6 |
| all     | file_level | change               |                5 |                0 |           5 |             5 |
| all     | file_level | import change        |                4 |                1 |           3 |             5 |
| all     | file_level | analysis             |                1 |                3 |          -2 |             4 |
| all     | file_level | reformatting         |                1 |                3 |          -2 |             4 |
| all     | file_level | risk                 |                3 |                1 |           2 |             4 |
| all     | file_level | class rename         |                0 |                3 |          -3 |             3 |
| all     | file_level | fix                  |                3 |                0 |           3 |             3 |
| all     | file_level | intention            |                3 |                0 |           3 |             3 |
| all     | file_level | json object          |                2 |                1 |           1 |             3 |
| all     | file_level | summary              |                1 |                2 |          -1 |             3 |
| all     | file_level | comment              |                0 |                2 |          -2 |             2 |
| all     | file_level | dataclass            |                2 |                0 |           2 |             2 |
| all     | file_level | logic                |                2 |                0 |           2 |             2 |
| all     | file_level | overall              |                1 |                1 |           0 |             2 |
| all     | file_level | additional           |                1 |                0 |           1 |             1 |
| all     | file_level | assertion            |                1 |                0 |           1 |             1 |
| all     | file_level | boolean              |                0 |                1 |          -1 |             1 |
| all     | file_level | cleanup              |                1 |                0 |           1 |             1 |
| all     | file_level | commentary           |                0 |                1 |          -1 |             1 |
| all     | file_level | commit message       |                0 |                1 |          -1 |             1 |
| all     | file_level | documentation change |                0 |                1 |          -1 |             1 |
| all     | file_level | exception            |                1 |                0 |           1 |             1 |
| all     | file_level | formatting           |                0 |                1 |          -1 |             1 |
| all     | file_level | function             |                1 |                0 |           1 |             1 |
| all     | file_level | informational        |                0 |                1 |          -1 |             1 |

## Interpretation

This analysis expands the Summary Agent artifact audit from curated labeled cases to all paired summaries available under `data/raw_model_outputs`. The all-instance rows describe the raw handoff artifacts regardless of outcome. The failures-only rows are limited to artifacts that could be joined to exact-match metadata and had at least one failed file in the corresponding result rows.

The selected-vs-rejected statistics remain proxy evidence: they compare the compact Summary Agent records for the parent selected by the bypass decision against the non-selected parent. They show whether asymmetry is already visible in the upstream summaries, but they do not prove that every missing summarized change was omitted by the Summarizer rather than absent from the underlying parent diff.

## Parse Error Samples

| raw_model_dir       |    sample_id | eval_method   | artifact_file_slug                                                        | file_path                                                                 | a_parse_ok   | b_parse_ok   | a_parse_error                                                                      | b_parse_error                                                                      | a_summary_path                                                                                                                                          | b_summary_path                                                                                                                                          |
|:--------------------|-------------:|:--------------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------|:-------------|:-------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| groq_qwen_qwen3-32b | 143142204719 | bypass7       | open_instruct_dataset_transformation.py                                   | open_instruct/dataset_transformation.py                                   | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\143142204719\bypass7\open_instruct_dataset_transformation.py\a_summary.txt                                   | data\raw_model_outputs\groq_qwen_qwen3-32b\143142204719\bypass7\open_instruct_dataset_transformation.py\b_summary.txt                                   |
| groq_qwen_qwen3-32b | 153728985754 | bypass7       | scripts_eval_constraints_if_functions.py                                  | scripts/eval_constraints/if_functions.py                                  | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\153728985754\bypass7\scripts_eval_constraints_if_functions.py\a_summary.txt                                  | data\raw_model_outputs\groq_qwen_qwen3-32b\153728985754\bypass7\scripts_eval_constraints_if_functions.py\b_summary.txt                                  |
| groq_qwen_qwen3-32b | 155467931302 | bypass7       | sdks_opik_optimizer_src_opik_optimizer_base_optimizer.py                  | sdks/opik_optimizer/src/opik_optimizer/base_optimizer.py                  | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\155467931302\bypass7\sdks_opik_optimizer_src_opik_optimizer_base_optimizer.py\a_summary.txt                  | data\raw_model_outputs\groq_qwen_qwen3-32b\155467931302\bypass7\sdks_opik_optimizer_src_opik_optimizer_base_optimizer.py\b_summary.txt                  |
| groq_qwen_qwen3-32b | 218091418381 | bypass7       | src_axolotl_cli_utils.py                                                  | src/axolotl/cli/utils.py                                                  | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\218091418381\bypass7\src_axolotl_cli_utils.py\a_summary.txt                                                  | data\raw_model_outputs\groq_qwen_qwen3-32b\218091418381\bypass7\src_axolotl_cli_utils.py\b_summary.txt                                                  |
| groq_qwen_qwen3-32b | 314867664798 | bypass7       | openhands_core_schema_action.py                                           | openhands/core/schema/action.py                                           | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\314867664798\bypass7\openhands_core_schema_action.py\a_summary.txt                                           | data\raw_model_outputs\groq_qwen_qwen3-32b\314867664798\bypass7\openhands_core_schema_action.py\b_summary.txt                                           |
| groq_qwen_qwen3-32b | 437689644084 | bypass7       | mm_agents_anthropic_main.py                                               | mm_agents/anthropic/main.py                                               | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\437689644084\bypass7\mm_agents_anthropic_main.py\a_summary.txt                                               | data\raw_model_outputs\groq_qwen_qwen3-32b\437689644084\bypass7\mm_agents_anthropic_main.py\b_summary.txt                                               |
| groq_qwen_qwen3-32b | 473036096332 | bypass7       | tests_smoke_tests_test_managed_job.py                                     | tests/smoke_tests/test_managed_job.py                                     | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\473036096332\bypass7\tests_smoke_tests_test_managed_job.py\a_summary.txt                                     | data\raw_model_outputs\groq_qwen_qwen3-32b\473036096332\bypass7\tests_smoke_tests_test_managed_job.py\b_summary.txt                                     |
| groq_qwen_qwen3-32b | 492978147465 | bypass7       | sky_client_cli_command.py                                                 | sky/client/cli/command.py                                                 | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\492978147465\bypass7\sky_client_cli_command.py\a_summary.txt                                                 | data\raw_model_outputs\groq_qwen_qwen3-32b\492978147465\bypass7\sky_client_cli_command.py\b_summary.txt                                                 |
| groq_qwen_qwen3-32b | 547094754328 | bypass7       | src_transformers_modeling_utils.py                                        | src/transformers/modeling_utils.py                                        | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\547094754328\bypass7\src_transformers_modeling_utils.py\a_summary.txt                                        | data\raw_model_outputs\groq_qwen_qwen3-32b\547094754328\bypass7\src_transformers_modeling_utils.py\b_summary.txt                                        |
| groq_qwen_qwen3-32b | 563597886573 | bypass7       | src_llmcompressor_transformers_sparsification_compressed_tensors_utils.py | src/llmcompressor/transformers/sparsification/compressed_tensors_utils.py | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\563597886573\bypass7\src_llmcompressor_transformers_sparsification_compressed_tensors_utils.py\a_summary.txt | data\raw_model_outputs\groq_qwen_qwen3-32b\563597886573\bypass7\src_llmcompressor_transformers_sparsification_compressed_tensors_utils.py\b_summary.txt |
| groq_qwen_qwen3-32b | 587240115231 | bypass7       | netmiko_ssh_dispatcher.py                                                 | netmiko/ssh_dispatcher.py                                                 | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\587240115231\bypass7\netmiko_ssh_dispatcher.py\a_summary.txt                                                 | data\raw_model_outputs\groq_qwen_qwen3-32b\587240115231\bypass7\netmiko_ssh_dispatcher.py\b_summary.txt                                                 |
| groq_qwen_qwen3-32b | 651031498269 | bypass7       | gef.py                                                                    | gef.py                                                                    | False        | False        | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\651031498269\bypass7\gef.py\a_summary.txt                                                                    | data\raw_model_outputs\groq_qwen_qwen3-32b\651031498269\bypass7\gef.py\b_summary.txt                                                                    |
| groq_qwen_qwen3-32b | 727878654500 | bypass7       | tests_conftest.py                                                         | tests/conftest.py                                                         | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\727878654500\bypass7\tests_conftest.py\a_summary.txt                                                         | data\raw_model_outputs\groq_qwen_qwen3-32b\727878654500\bypass7\tests_conftest.py\b_summary.txt                                                         |
| groq_qwen_qwen3-32b | 750908101200 | bypass7       | dspy_utils_parallelizer.py                                                | dspy/utils/parallelizer.py                                                | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\750908101200\bypass7\dspy_utils_parallelizer.py\a_summary.txt                                                | data\raw_model_outputs\groq_qwen_qwen3-32b\750908101200\bypass7\dspy_utils_parallelizer.py\b_summary.txt                                                |
| groq_qwen_qwen3-32b | 795115689640 | bypass7       | app_api_router.py                                                         | app/api/router.py                                                         | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\795115689640\bypass7\app_api_router.py\a_summary.txt                                                         | data\raw_model_outputs\groq_qwen_qwen3-32b\795115689640\bypass7\app_api_router.py\b_summary.txt                                                         |
| groq_qwen_qwen3-32b | 849853816180 | bypass7       | src_transformers_modeling_utils.py                                        | src/transformers/modeling_utils.py                                        | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\849853816180\bypass7\src_transformers_modeling_utils.py\a_summary.txt                                        | data\raw_model_outputs\groq_qwen_qwen3-32b\849853816180\bypass7\src_transformers_modeling_utils.py\b_summary.txt                                        |
| groq_qwen_qwen3-32b |  90998825390 | bypass7       | lotus_models_rm.py                                                        | lotus/models/rm.py                                                        | True         | False        | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | data\raw_model_outputs\groq_qwen_qwen3-32b\90998825390\bypass7\lotus_models_rm.py\a_summary.txt                                                         | data\raw_model_outputs\groq_qwen_qwen3-32b\90998825390\bypass7\lotus_models_rm.py\b_summary.txt                                                         |
| groq_qwen_qwen3-32b |  93075495244 | bypass7       | thunder_core_interpreter.py                                               | thunder/core/interpreter.py                                               | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\93075495244\bypass7\thunder_core_interpreter.py\a_summary.txt                                                | data\raw_model_outputs\groq_qwen_qwen3-32b\93075495244\bypass7\thunder_core_interpreter.py\b_summary.txt                                                |
| groq_qwen_qwen3-32b | 973399694447 | bypass7       | ray-curator_ray_curator_stages_filters_heuristic_filter.py                | ray-curator/ray_curator/stages/filters/heuristic_filter.py                | False        | True         | nested_summary_json: Expecting value: line 1 column 1 (char 0)                     | nested_summary_json_fallback_type_regex: Expecting value: line 1 column 1 (char 0) | data\raw_model_outputs\groq_qwen_qwen3-32b\973399694447\bypass7\ray-curator_ray_curator_stages_filters_heuristic_filter.py\a_summary.txt                | data\raw_model_outputs\groq_qwen_qwen3-32b\973399694447\bypass7\ray-curator_ray_curator_stages_filters_heuristic_filter.py\b_summary.txt                |
| llama-3.1-8b        | 102813106132 | bypass7       | benchmarks_benchmark_runner.py                                            |                                                                           | False        | False        | outer_json: Extra data: line 29 column 2 (char 1398)                               | outer_json: Extra data: line 23 column 2 (char 1451)                               | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\benchmarks_benchmark_runner.py\a_summary.txt                                                   | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\benchmarks_benchmark_runner.py\b_summary.txt                                                   |
| llama-3.1-8b        | 102813106132 | bypass7       | benchmarks_maxtext_trillium_model_configs.py                              |                                                                           | False        | False        | outer_json: Expecting ',' delimiter: line 12 column 46 (char 325)                  | outer_json: Unterminated string starting at: line 191 column 19 (char 7645)        | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\benchmarks_maxtext_trillium_model_configs.py\a_summary.txt                                     | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\benchmarks_maxtext_trillium_model_configs.py\b_summary.txt                                     |
| llama-3.1-8b        | 102813106132 | bypass7       | benchmarks_maxtext_v5p_model_configs.py                                   |                                                                           | False        | False        | outer_json: Extra data: line 26 column 2 (char 1314)                               | outer_json: Extra data: line 53 column 2 (char 1789)                               | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\benchmarks_maxtext_v5p_model_configs.py\a_summary.txt                                          | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\benchmarks_maxtext_v5p_model_configs.py\b_summary.txt                                          |
| llama-3.1-8b        | 102813106132 | bypass7       | MaxText_layers_models.py                                                  |                                                                           | False        | False        | outer_json: Extra data: line 17 column 2 (char 763)                                | outer_json: Extra data: line 65 column 2 (char 6876)                               | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\MaxText_layers_models.py\a_summary.txt                                                         | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\MaxText_layers_models.py\b_summary.txt                                                         |
| llama-3.1-8b        | 102813106132 | bypass7       | MaxText_layers_quantizations.py                                           |                                                                           | False        | False        | outer_json: Extra data: line 47 column 2 (char 3104)                               | outer_json: Extra data: line 179 column 2 (char 7360)                              | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\MaxText_layers_quantizations.py\a_summary.txt                                                  | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\MaxText_layers_quantizations.py\b_summary.txt                                                  |
| llama-3.1-8b        | 102813106132 | bypass7       | MaxText_pyconfig.py                                                       |                                                                           | False        | False        | outer_json: Extra data: line 50 column 2 (char 1696)                               | outer_json: Expecting ',' delimiter: line 112 column 283 (char 4806)               | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\MaxText_pyconfig.py\a_summary.txt                                                              | data\raw_model_outputs\llama-3.1-8b\102813106132\bypass7\MaxText_pyconfig.py\b_summary.txt                                                              |
