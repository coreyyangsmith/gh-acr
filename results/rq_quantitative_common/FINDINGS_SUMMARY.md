# Common-Set Quantitative Analysis: Findings Summary

**Date:** 2026-02-07
**Dataset:** 1,888 merge scenarios with outputs from ALL 3 models (GPT-5-nano, Qwen3-32B, LLaMA-3.1-8B)
**Focus:** Non-label statistics/metrics computed across the full common dataset

## Data Inputs

| Input | Path | Description |
|-------|------|-------------|
| Results CSV | `data/2026_01_results_final.csv` | 29,436 rows, all models + methods |
| Dataset CSV | `data/git_good_bench_merge_commits_all.csv` | 1,909 scenarios with metadata |
| **Common Set** | 1,888 IDs | Intersection where all 3 models have agent + bypass results |

## Common Set Composition

| Category | Breakdown |
|----------|-----------|
| **Difficulty** | easy: 877 (46.4%), medium: 378 (20.0%), hard: 633 (33.5%) |
| **Project Size** | small: 103 (5.5%), medium: 1,088 (57.6%), large: 600 (31.8%), huge: 97 (5.1%) |
| **Conflict Files** | median=1, mean=1.7, range=[1, 15] |
| **Total Conflicts** | median=2, mean=3.4, range=[1, 74] |
| **Repo Commits** | median=1,711, mean=2,934, range=[55, 22,973] |
| **Repo Code Lines** | median=70,350, mean=199,219, range=[1,036, 2,500,524] |
| **Repo Contributors** | median=81, mean=117, range=[5, 472] |

**Supporting files:** `common_scenario_metadata.csv`, `common_scenario_distribution.csv`, `common_difficulty_distribution.png`, `common_scenario_distributions.png`

---

## Key Finding 1: Model Performance Comparison on Common Set

| Model | Method | Exact Match | Similarity | BLEU-3 | ROUGE-L |
|-------|--------|------------|------------|--------|---------|
| **GPT-5-nano** | Agent | **0.052** | **0.872** | **0.894** | **0.912** |
| GPT-5-nano | Bypass | 0.069 | 0.859 | 0.865 | 0.897 |
| Qwen3-32B | Agent | 0.000 | 0.587 | 0.616 | 0.654 |
| **Qwen3-32B** | **Bypass** | **0.113** | **0.905** | **0.911** | **0.933** |
| LLaMA-3.1-8B | Agent | 0.000 | 0.517 | 0.460 | 0.614 |
| **LLaMA-3.1-8B** | **Bypass** | **0.120** | **0.903** | **0.908** | **0.930** |

**Key Insight:** GPT-5-nano is the only model where the Agent method produces meaningful results (5.2% EM, 0.87 similarity). For Qwen3 and LLaMA, the Agent method achieves 0% exact match and substantially lower similarity/BLEU/ROUGE scores. The Bypass (multi-agent) method dramatically closes this gap - Qwen3 and LLaMA Bypass performance (11-12% EM, ~0.90 similarity) surpasses GPT-5-nano Bypass (6.9% EM, 0.86 similarity).

**Supporting files:** `common_model_comparison.csv`, `common_model_comparison.png`

---

## Key Finding 2: Conflict Complexity is the Strongest Predictor of Performance

The **number of total conflicts** (`n_total_conflicts`) is the single strongest scenario-level predictor of performance, with consistently negative correlations across all models and methods:

### Top Aggregated Correlations (All Models, averaged)

| Scenario Metric | Performance Metric | Method | Spearman r | p-value |
|----------------|-------------------|--------|-----------|---------|
| n_total_conflicts | similarity | Bypass | **-0.229*** | <0.001 |
| n_total_conflicts | rouge_l | Bypass | **-0.227*** | <0.001 |
| n_total_conflicts | bleu3 | Bypass | **-0.222*** | <0.001 |
| n_conflict_files | rouge_l | Bypass | **-0.197*** | <0.001 |
| n_conflict_files | bleu3 | Bypass | **-0.196*** | <0.001 |
| n_conflict_files | similarity | Bypass | **-0.195*** | <0.001 |
| n_total_conflicts | exact_match | Agent | **-0.186*** | <0.001 |
| n_conflict_files | exact_match | Bypass | **-0.155*** | <0.001 |
| n_total_conflicts | similarity | Agent | **-0.155*** | <0.001 |
| n_total_conflicts | exact_match | Bypass | **-0.150*** | <0.001 |

### Per-Model Strongest Correlations

| Model | Scenario Metric | Performance Metric | Method | r |
|-------|----------------|-------------------|--------|---|
| **GPT-5-nano** | n_total_conflicts | bleu3 | Agent | **-0.344*** |
| GPT-5-nano | n_total_conflicts | rouge_l | Agent | -0.328*** |
| GPT-5-nano | n_total_conflicts | similarity | Agent | -0.324*** |
| **Qwen3-32B** | n_total_conflicts | bleu3 | Bypass | **-0.237*** |
| Qwen3-32B | n_total_conflicts | rouge_l | Bypass | -0.236*** |
| **LLaMA-3.1-8B** | n_total_conflicts | similarity | Bypass | **-0.224*** |
| LLaMA-3.1-8B | n_total_conflicts | rouge_l | Bypass | -0.217*** |

**Interpretation:** More merge conflicts in a scenario reliably predicts worse LLM resolution quality. This effect is strongest for GPT-5-nano's Agent method (r=-0.34) and is consistent across all models for the Bypass method (r ~ -0.22 to -0.24). The number of conflicted files shows a similar but weaker pattern.

**Supporting files:** `common_scenario_perf_correlation.csv`, `common_correlation_heatmap_Agent_Single.png`, `common_correlation_heatmap_Bypass_Multi.png`, `common_scenario_vs_perf_Agent_Single.png`, `common_scenario_vs_perf_Bypass_Multi.png`

---

## Key Finding 3: Repo-Level Metrics Have Weak but Significant Effects on Agent Performance

Repository-level characteristics (commits, code lines, contributors) show small but statistically significant negative correlations with Agent-method performance only:

| Scenario Metric | Performance Metric | Method | r |
|----------------|-------------------|--------|---|
| repo_contributors | bleu3 | Agent | -0.138*** |
| repo_commits | bleu3 | Agent | -0.126*** |
| repo_code_lines | bleu3 | Agent | -0.125*** |
| repo_contributors | rouge_l | Agent | -0.117*** |
| repo_commits | rouge_l | Agent | -0.108*** |
| repo_contributors | similarity | Agent | -0.107*** |

These correlations are **not significant for the Bypass method**, suggesting that the multi-agent approach is more robust to repository complexity than the single-agent approach.

**Supporting files:** `common_scenario_perf_correlation.csv`, `common_correlation_heatmap_Agent_Single.png`

---

## Key Finding 4: Difficulty Strongly Modulates Performance

Performance degrades from easy to hard scenarios, but the pattern varies by model and method:

### Bypass Method (Primary Results)

| Model | Easy EM | Medium EM | Hard EM | Easy Sim | Medium Sim | Hard Sim |
|-------|---------|-----------|---------|----------|------------|----------|
| Qwen3-32B | 0.152 | 0.122 | 0.055 | 0.922 | 0.902 | 0.884 |
| LLaMA-3.1-8B | 0.164 | 0.108 | 0.065 | 0.918 | 0.891 | 0.889 |
| GPT-5-nano | 0.088 | 0.071 | 0.043 | 0.875 | 0.844 | 0.847 |

The **easy-to-hard drop** in exact match rate:
- Qwen3-32B: 15.2% -> 5.5% (64% relative decrease)
- LLaMA-3.1-8B: 16.4% -> 6.5% (60% relative decrease)
- GPT-5-nano: 8.8% -> 4.3% (51% relative decrease)

### Agent Method

For GPT-5-nano (the only model with non-zero Agent EM): 9.7% easy -> 0.9% hard (91% decrease). Similarity also drops from 0.902 to 0.853.

**Supporting files:** `common_perf_by_difficulty.csv`, `common_perf_by_difficulty_Agent_Single.png`, `common_perf_by_difficulty_Bypass_Multi.png`

---

## Key Finding 5: Project Size has a Non-Linear Effect

| Model (Bypass) | Small EM | Medium EM | Large EM | Huge EM* |
|----------------|----------|-----------|----------|----------|
| Qwen3-32B | 0.175 | 0.121 | 0.097 | (see CSV) |
| LLaMA-3.1-8B | 0.184 | 0.124 | 0.102 | (see CSV) |
| GPT-5-nano | 0.068 | 0.064 | 0.080 | (see CSV) |

Small projects yield the highest exact match rates for Qwen3 and LLaMA Bypass methods, suggesting simpler codebases are easier to resolve correctly. However, GPT-5-nano shows a different pattern with large projects slightly outperforming medium ones.

For the Agent method, LLaMA-3.1-8B shows a striking pattern: similarity of 0.706 on small projects vs. 0.492 on large ones, a much steeper decline than the other models.

**Supporting files:** `common_perf_by_project_size.csv`

---

## Methodology

This analysis uses the **common-set approach**: only scenarios where ALL 3 models produced outputs for both agent and bypass methods are included. This ensures:

1. **Fair comparison** - all models are evaluated on identical scenarios
2. **Maximum statistical power** - 1,888 scenarios (vs. ~1,078 in the case-folder-based analysis)
3. **No label dependency** - all statistics computed purely from CSV data (results + dataset metadata)

The scenario metadata (conflict counts, repo stats) is extracted from the GitGoodBench dataset CSV. Performance is aggregated at the instance level (min for exact_match across multi-file scenarios, mean for continuous metrics).

### Pipeline

```
Results CSV (29,436 rows)
    |
    v
[Find Common IDs] --> 1,888 IDs (intersection across 3 models, agent+bypass)
    |
    v
[Load Scenario Metadata] --> difficulty, project_size, conflict counts, repo stats
    |
    v
[Build Performance DF] --> 11,328 rows (1,888 x 3 models x 2 methods)
    |
    v
[Correlations] --> Spearman/Pearson: scenario metrics vs performance (per model + aggregated)
    |
    v
[Category Breakdowns] --> Performance by difficulty, project_size
    |
    v
[Model Comparison] --> Head-to-head on identical set
    |
    v
[Figures & CSVs] --> 9 figures, 7 CSVs
```

### Script

```bash
python -m src.results.quantitative.common_set_analysis \
    --results-csv data/2026_01_results_final.csv \
    --dataset-csv data/git_good_bench_merge_commits_all.csv \
    --output-dir results/rq_quantitative_common
```

---

## Output File Index

### CSVs
| File | Description |
|------|-------------|
| `common_ids.txt` | List of 1,888 common IDs |
| `common_scenario_metadata.csv` | Scenario metadata for 1,888 IDs |
| `common_performance.csv` | Per-ID, per-model, per-method performance |
| `common_scenario_perf_correlation.csv` | 160 scenario-performance Spearman/Pearson correlations |
| `common_perf_by_difficulty.csv` | Performance breakdown by difficulty |
| `common_perf_by_project_size.csv` | Performance breakdown by project size |
| `common_model_comparison.csv` | Per-model summary statistics |
| `common_scenario_distribution.csv` | Scenario metadata descriptive stats |

### Figures
| File | Description |
|------|-------------|
| `common_correlation_heatmap_Agent_Single.png` | Heatmap: scenario metrics vs Agent performance |
| `common_correlation_heatmap_Bypass_Multi.png` | Heatmap: scenario metrics vs Bypass performance |
| `common_perf_by_difficulty_Agent_Single.png` | Performance by difficulty (Agent) |
| `common_perf_by_difficulty_Bypass_Multi.png` | Performance by difficulty (Bypass) |
| `common_model_comparison.png` | Model comparison bar chart |
| `common_scenario_distributions.png` | Scenario metadata histograms |
| `common_scenario_vs_perf_Agent_Single.png` | Scatter: scenario metrics vs exact match (Agent) |
| `common_scenario_vs_perf_Bypass_Multi.png` | Scatter: scenario metrics vs exact match (Bypass) |
| `common_difficulty_distribution.png` | Dataset composition (difficulty + project_size) |
