# Summary-Agent Complexity Preference Audit

## Coverage

- File-level summary pairs parsed: 2,228.
- Instance-level rows after aggregation: 1,128.
- Instances with selected parent metadata A/B: 1,128.
- Instances with at least one exact-match failure row: 1,128.
- Parse failure file rows: 464.
- Fail-only paired IDs without parsed summary artifacts: 1.

## Analyzer Decisions

- `A`: 803 (71.2%)
- `B`: 325 (28.8%)

## Derived Preference Labels

- `favored_simplicity`: 405 (35.9%)
- `favored_complexity`: 302 (26.8%)
- `tie_ambiguous`: 252 (22.3%)
- `unparseable`: 169 (15.0%)

## Manual Label Agreement

- Comparable instances: 823.
- Agreement with existing manual dominant preference: 571 (69.4%).

## Selected vs Rejected Type Counts

| scope          | change_type            |   selected_parent_count |   rejected_parent_count |   selected_minus_rejected |
|:---------------|:-----------------------|------------------------:|------------------------:|--------------------------:|
| instance_level | addition               |                    5993 |                    7952 |                     -1959 |
| instance_level | additional             |                       1 |                       0 |                         1 |
| instance_level | analysis               |                       1 |                       2 |                        -1 |
| instance_level | assertion              |                       0 |                       1 |                        -1 |
| instance_level | boolean                |                       0 |                       1 |                        -1 |
| instance_level | class rename           |                       0 |                       3 |                        -3 |
| instance_level | comment                |                       0 |                       1 |                        -1 |
| instance_level | commentary             |                       0 |                       1 |                        -1 |
| instance_level | deprecation            |                       0 |                       3 |                        -3 |
| instance_level | import                 |                       2 |                       1 |                         1 |
| instance_level | intent                 |                       0 |                       1 |                        -1 |
| instance_level | json object            |                       1 |                       0 |                         1 |
| instance_level | likely intent          |                       5 |                      17 |                       -12 |
| instance_level | metaclass introduction |                       0 |                       1 |                        -1 |
| instance_level | modification           |                   10309 |                    9357 |                       952 |
| instance_level | note                   |                       1 |                       2 |                        -1 |
| instance_level | refactor               |                       1 |                       1 |                         0 |
| instance_level | reformatting           |                       1 |                       0 |                         1 |
| instance_level | removal                |                     524 |                     565 |                       -41 |
| instance_level | renaming               |                       1 |                       6 |                        -5 |
| instance_level | reordering             |                       0 |                       2 |                        -2 |
| instance_level | reorganization         |                       1 |                       0 |                         1 |
| instance_level | replacement            |                       0 |                       3 |                        -3 |
| instance_level | summary                |                       1 |                       0 |                         1 |
| instance_level | unknown                |                     491 |                     482 |                         9 |
