# Quantitative Change Metrics — Methodology

## Concrete Data Inputs Used

### Case Folders (6 folders, 2,594 total sample subfolders)

| # | Case Folder | Model | Type | Subfolders |
|---|-------------|-------|------|------------|
| 1 | `data/labeled/2025-11-09-gpt5nano-failure-cases/` | GPT-5-nano | fail | 500 |
| 2 | `data/labeled/2025-11-11-gpt5nano-pass-cases/` | GPT-5-nano | pass | 315 |
| 3 | `data/labeled/2025-11-23-qwen3-failure-cases/` | Qwen3-32B | fail | 461 |
| 4 | `data/labeled/2025-12-03-qwen3-pass-cases/` | Qwen3-32B | pass | 415 |
| 5 | `data/labeled/2026-01-23-llama-pass-cases/` | LLaMA-3.1-8B | pass | 422 |
| 6 | `data/labeled/2026-02-03-llama-fail-cases/` | LLaMA-3.1-8B | fail | 481 |

Each subfolder contains: `default/` (ancestor, parents, GT, diffs, commit messages), `agent/` (single-agent output), `bypass/` (multi-agent output).

### CSV Inputs

| Input | Path | Description |
|-------|------|-------------|
| Performance results | `data/2026_01_results_final.csv` | 29,436 rows (1,909 unique IDs); columns: `id`, `model_name`, `eval_method`, `exact_match`, `similarity`, `bleu3`, `rouge_l`, `difficulty`, `project_size`, etc. Models: GPT-5-nano, Qwen3-32B, LLaMA-3.1-8B |
| Scenario metadata | `data/git_good_bench_merge_commits_all.csv` | 1,909 rows; used for scenario enrichment (n_conflict_files, n_total_conflicts, repo_commits, repo_code_lines, repo_contributors) |
| RQ3 paired labels | `results/rq3/paired_data.csv` | 872 rows; columns: `id`, binary label columns (`favored_simplicity`, `structural_change_bias`, etc.), per-method performance, deltas |
| RQ3 complexity | `results/rq3/complexity_metrics.csv` | 5,500 rows; columns: `sample_id`, `method`, `sloc`, `lloc`, `cc_total`, `cc_avg`, `mi_score`, `h_difficulty`, `h_bugs`, etc. |

### Classification JSONs (6 files)

All under `data/labeled/`: `2025-11-09-gpt5nano-failure-classifications.json`, `2025-11-11-gpt5nano-pass-classifications.json`, `2025-11-23-qwen3-failure-classifications.json`, `2025-12-03-qwen3-pass-classifications.json`, `2026-01-23-llama-pass-classifications.json`, `2026-02-03-llama-fail-classifications.json`

### Output Directory

All outputs written to `results/rq_quantitative/`.

---

## High-Level Pipeline Flow

```mermaid
flowchart TB
    subgraph Inputs["Data Inputs"]
        CF["Case Folders\n(sample subfolders with\ncode, diffs, commit msgs)"]
        RES["RQ2 Results CSV\n(models_combined.csv)"]
        RQ3P["RQ3 Paired Data CSV\n(paired_data.csv)"]
        RQ3C["RQ3 Complexity CSV\n(complexity_metrics.csv)"]
        CJ["Classification JSONs\n(manual labels)"]
        DS["GitGoodBench CSV\n(scenario metadata)"]
    end

    subgraph Step1["Step 1: ID Gathering"]
        ID["Unique Sample IDs"]
    end

    subgraph Step2["Step 2: Per-Sample Metric Computation"]
        direction TB
        LOAD["Load sample folder:\noriginal.txt, a.txt, b.txt,\nground_truth.txt,\nagent/*.txt, bypass/*.txt"]
        LOAD --> SIZE["Size Metrics (radon)\nLOC, SLOC, blank, comment"]
        LOAD --> DIFF["Diff Metrics\nParse .diff or generate via difflib"]
        LOAD --> COMMIT["Commit Metrics\nParse commit_message.txt"]
        SIZE --> VM["VersionMetrics\n(per sample x per version)"]
        DIFF --> VM
        COMMIT --> VM
    end

    subgraph Step3["Step 3: Aggregation"]
        AGG["Summary by Version\nmean / std / median / min / max"]
    end

    subgraph Step4["Step 4: Pairwise Deltas"]
        DELTA["Deltas: agent vs GT,\nbypass vs GT,\nagent vs bypass"]
    end

    subgraph Step5["Step 5: Enrichment"]
        ENRICH["Join with scenario metadata:\nn_conflict_files,\nn_total_conflicts,\nrepo_commits"]
    end

    subgraph Step7["Step 7: Correlation Analysis"]
        direction TB
        PERF["7a: Quantitative vs Performance\nSpearman + Pearson\n(delta_exact_match,\ndelta_similarity, etc.)"]
        LABEL["7b: Quantitative vs Labels\nMann-Whitney U\n(favored_simplicity,\nstructural_change_bias, etc.)"]
        CROSS["7c: Quantitative vs Complexity\nSpearman + Pearson\n(cc_total, mi_score,\nh_bugs, etc.)"]
    end

    subgraph Step8["Step 8: Visualization"]
        direction TB
        P1["Size by Version\n(boxplots)"]
        P2["Change Magnitude\n(boxplots)"]
        P3["Commit Distribution\n(histograms)"]
        P4["By Difficulty\n(boxplots)"]
        P5["Correlation Heatmap"]
        P6["Metric vs Performance\n(scatter + trend)"]
        P7["Metrics by Label\n(grouped bar)"]
        P8["Summary Table"]
    end

    CF --> Step1
    CJ --> Step1
    Step1 --> Step2
    Step2 --> Step3
    Step2 --> Step4
    DS --> Step5
    Step4 --> Step5
    RES --> Step7
    Step5 --> PERF
    Step5 --> LABEL
    RQ3P --> LABEL
    Step5 --> CROSS
    RQ3C --> CROSS
    Step7 --> Step8
    Step3 --> Step8
    Step5 --> Step8
    RES --> Step8
    RQ3P --> Step8

    subgraph Outputs["CSV Outputs"]
        O1["quantitative_metrics.csv\n(~1211 rows: sample x version)"]
        O2["quantitative_summary.csv\n(6 rows: per version)"]
        O3["quantitative_deltas.csv\n(204 rows: per sample)"]
        O4["quantitative_performance_correlation.csv"]
        O5["quantitative_label_correlation.csv"]
        O6["quantitative_complexity_cross.csv"]
    end

    Step2 --> O1
    Step3 --> O2
    Step4 --> O3
    PERF --> O4
    LABEL --> O5
    CROSS --> O6
```

## Detailed Metric Computation

### For Each Sample Folder

```mermaid
flowchart LR
    subgraph folder["Sample Folder (e.g. 334950146121-1/)"]
        direction TB
        DEF["default/filename/"]
        AGT["agent/filename.txt"]
        BYP["bypass/filename/bypass_filename.txt"]
    end

    subgraph default_files["Default Subfolder"]
        ORI["original.txt\n(ancestor)"]
        ATXT["a.txt\n(parent A)"]
        BTXT["b.txt\n(parent B)"]
        GTTXT["ground_truth.txt"]
        ADIFF["a.diff"]
        BDIFF["b.diff"]
        GTDIFF["ground_truth.diff"]
        ACM["a_commit_message.txt"]
        BCM["b_commit_message.txt"]
    end

    DEF --> default_files

    subgraph compute["Per-Version Computation"]
        direction TB
        V_PREV["previous: size_metrics(original.txt)\ndiff = 0 (baseline)"]
        V_A["a: size_metrics(a.txt)\ndiff = parse(a.diff)\ndelta = a - ancestor"]
        V_B["b: size_metrics(b.txt)\ndiff = parse(b.diff)\ndelta = b - ancestor"]
        V_GT["ground_truth: size_metrics(gt.txt)\ndiff = parse(gt.diff)\ndelta = gt - ancestor"]
        V_AGT["agent: size_metrics(agent.txt)\ndiff = generate_diff(ancestor, agent)\ndelta = agent - ancestor"]
        V_BYP["bypass: size_metrics(bypass.txt)\ndiff = generate_diff(ancestor, bypass)\ndelta = bypass - ancestor"]
    end

    ORI --> V_PREV
    ATXT --> V_A
    ADIFF --> V_A
    BTXT --> V_B
    BDIFF --> V_B
    GTTXT --> V_GT
    GTDIFF --> V_GT
    AGT --> V_AGT
    ORI --> V_AGT
    BYP --> V_BYP
    ORI --> V_BYP

    subgraph commits["Commit Counting"]
        CA["n_commits_a = count_sections(a_commit_message.txt)"]
        CB["n_commits_b = count_sections(b_commit_message.txt)"]
        CT["n_commits_total = a + b"]
    end

    ACM --> CA
    BCM --> CB
    CA --> CT
    CB --> CT

    compute --> ROW["6 rows per sample\n(one per version)\neach with all metrics +\ncommit counts"]
    commits --> ROW
```

### Correlation Analysis Detail

```mermaid
flowchart TD
    DELTAS["quantitative_deltas.csv\n(per-sample metrics)"] --> STRIP["Strip sample_id suffix\n(e.g. 12345-1 → 12345)\nfor join compatibility"]

    subgraph perf_corr["7a: Performance Correlation"]
        PAIR["Compute performance pairs:\nbypass_metric - agent_metric\n→ delta_exact_match, delta_similarity, etc."]
        JOIN1["Inner join on base_id"]
        CORR1["For each (quant_metric, perf_delta):\n• Spearman rank correlation\n• Pearson linear correlation\n• Sort by |Spearman r|"]
    end

    subgraph label_corr["7b: Label Correlation"]
        JOIN2["Inner join with paired_data\non base_id"]
        SPLIT["Split samples:\nwith_label (=1) vs without_label (=0)"]
        MW["For each (label, quant_metric):\n• Mean with vs without\n• Mann-Whitney U test\n• Sort by p-value"]
    end

    subgraph cross_corr["7c: Complexity Cross-Correlation"]
        JOIN3["Inner join with complexity_metrics\non base_id × method"]
        CORR3["For each (quant_metric, complexity_metric, method):\n• Spearman rank correlation\n• Pearson linear correlation\n• Sort by |Spearman r|"]
    end

    STRIP --> JOIN1
    STRIP --> JOIN2
    STRIP --> JOIN3
    RES2["results_csv"] --> PAIR
    PAIR --> JOIN1
    JOIN1 --> CORR1
    PAIRED["paired_data.csv"] --> JOIN2
    JOIN2 --> SPLIT --> MW
    COMPLEX["complexity_metrics.csv"] --> JOIN3
    JOIN3 --> CORR3
```

## Source Code Architecture

```
src/analysis/quantitative/
├── __init__.py           # Exports: generate_all_quantitative, QuantFlags
├── config.py             # Constants, QuantConfig dataclass
│   ├── VERSIONS, VERSION_DISPLAY_NAMES, VERSION_COLORS
│   ├── SIZE_METRICS, CHANGE_METRICS, DIFF_METRICS, COMMIT_METRICS
│   ├── PERFORMANCE_METRICS, METRIC_DISPLAY_NAMES
│   ├── Bucket definitions (commit count, change magnitude, LOC delta)
│   └── QuantConfig (n_bootstrap, ci_level, figsize, colors, ...)
├── metrics.py            # Core computation (stateless, no I/O)
│   ├── DiffMetrics, parse_unified_diff()
│   ├── SizeMetrics, compute_size_metrics()  [uses radon]
│   ├── count_commits()
│   └── VersionMetrics, compute_version_metrics()
├── loader.py             # Data loading + batch processing
│   ├── SampleData (container for all raw artifacts)
│   ├── load_sample_data() → SampleData
│   ├── compute_sample_quantitative_metrics() → list[dict]
│   ├── process_all_samples() → pd.DataFrame
│   ├── aggregate_metrics_by_version() → pd.DataFrame
│   └── compute_quantitative_deltas() → pd.DataFrame
├── correlations.py       # Statistical analysis
│   ├── _bootstrap_ci(), _coerce_exact_match()
│   ├── _prepare_performance_pairs()
│   ├── compute_performance_correlations()   [Spearman, Pearson]
│   ├── compute_label_correlations()         [Mann-Whitney U]
│   └── compute_complexity_cross_correlations()
├── plots.py              # All visualization functions
│   ├── plot_size_by_version()               [4-panel boxplot]
│   ├── plot_change_magnitude_by_version()   [3-panel boxplot]
│   ├── plot_commit_count_distribution()     [3-panel histogram]
│   ├── plot_change_by_difficulty()          [3-panel by easy/med/hard]
│   ├── plot_correlation_heatmap()           [annotated heatmap]
│   ├── plot_metric_vs_performance_scatter() [2×2 scatter + trend]
│   ├── plot_metrics_by_label()              [grouped horizontal bar]
│   └── plot_summary_table()                 [rendered table]
└── main.py               # 8-step orchestrator + CLI (tyro)
    ├── QuantFlags (CLI dataclass)
    ├── _load_scenario_metadata()
    ├── _extract_sample_ids_from_case_folder()
    ├── _extract_sample_ids_from_json()
    └── generate_all_quantitative()
```
