# RQ1: Does Multi-Agent Improve Merge Conflict Resolution Quality?

## Summary

We evaluate whether a multi-agent approach—where the system can select Parent A, Parent B, or generate a custom merge—improves resolution quality compared to single-agent LLM generation across three coding models: Qwen3-32B, Llama-3.1-8B-Instruct, and GPT-5-nano.

**Experimental Design:** We align all methods to the same set of **1,888 merge conflict instances** (comprising 3,272 files across 342 repositories) present across all evaluation conditions. We report **per-instance metrics** as our primary measure: a merge conflict is considered resolved only when *all* constituent files are correctly resolved.

**Key Finding:** Multi-agent consistently improves exact match rates over single-agent across all models (+1.7 to +11.9 percentage points). However, the improvement is entirely driven by *parent selection* rather than novel merge generation. Models exhibit strong biases toward selecting Parent A, and only Llama-3.1-8B demonstrates any positive intelligent selection capability.

---

## 1. Overall Performance Comparison

All results reported at the **per-instance level** (n = 1,888 merge conflicts). For exact match, an instance is correct only if ALL files within that conflict are resolved correctly. Soft metrics are averaged across files within each instance.

### 1.1 Exact Match Results

| Method | Exact Matches | Exact Match Rate |
|--------|---------------|------------------|
| **Base A** | 213 | 11.3% |
| **Base B** | 192 | 10.2% |

| Model | Single-Agent EM | Multi-Agent EM | Δ EM |
|-------|-----------------|----------------|------|
| Qwen3-32B | 0.0% | 11.3% | +11.3pp |
| Llama-3.1-8B | 0.0% | 11.9% | +11.9pp |
| GPT-5-nano | 5.2% | 6.9% | +1.7pp |

**Key Observations:**

1. **Qwen's multi-agent exactly matches Base A**: At 11.3% EM with 99.85% A-selection, Qwen's multi-agent performance equals the Base A rate, confirming it functions purely as a Base A selector.
2. **Llama achieves the highest EM**: At 11.9%, Llama slightly exceeds both baselines (11.3% Base A, 10.2% Base B), suggesting modest discriminative capability.
3. **GPT-5-nano underperforms both baselines**: Despite near-balanced A/B selection, GPT-5-nano achieves only 6.9% EM—well below both baselines. This indicates actively suboptimal parent selection.

### 1.2 Similarity and Soft Metrics

| Model | Single Sim | Multi Sim | Δ Sim | Single BLEU-3 | Multi BLEU-3 | Single ROUGE-L | Multi ROUGE-L |
|-------|------------|-----------|-------|---------------|--------------|----------------|---------------|
| Qwen3-32B | 0.587 | 0.905 | +54.2% | 0.616 | 0.911 | 0.654 | 0.933 |
| Llama-3.1-8B | 0.516 | 0.901 | +74.6% | 0.460 | 0.906 | 0.614 | 0.928 |
| GPT-5-nano | 0.872 | 0.859 | **−1.5%** | 0.894 | 0.865 | 0.912 | 0.897 |

**Critical Observation:** While open-weight models (Qwen, Llama) show substantial similarity improvements, GPT-5-nano exhibits a *decrease* in similarity under multi-agent (−1.5%). This suggests that when GPT-5-nano generates its own merge (single-agent), it produces outputs closer to ground truth than when selecting parents via bypass.

### 1.3 Comparison with Baselines

| Method | EM Rate | Similarity | BLEU-3 | ROUGE-L |
|--------|---------|------------|--------|---------|
| Base A | 11.3% | 0.905 | 0.911 | 0.932 |
| Base B | 10.2% | 0.888 | 0.891 | 0.919 |
| Qwen3-32B (multi) | 11.3% | 0.905 | 0.911 | 0.933 |
| Llama-3.1-8B (multi) | 11.9% | 0.901 | 0.906 | 0.928 |
| GPT-5-nano (multi) | 6.9% | 0.859 | 0.865 | 0.897 |

---

## 2. Bypass Method Distribution: The Parent Selection Bias

| Model | Choose A | Choose B | MIX (Custom) | Total Files |
|-------|----------|----------|--------------|-------------|
| Qwen3-32B | 3,267 (99.85%) | 5 (0.15%) | 0 (0.0%) | 3,272 |
| Llama-3.1-8B | 2,116 (64.7%) | 1,156 (35.3%) | 0 (0.0%) | 3,272 |
| GPT-5-nano | 1,582 (48.3%) | 1,681 (51.4%) | 9 (0.28%) | 3,272 |

### 2.1 Intelligent Selection Value

We compute the "intelligent selection value" (ISV) to measure whether models select parents better than random chance:

$$\text{ISV} = \text{Actual EM} - \text{Expected EM}$$

where Expected EM = (% choosing A × Base A EM) + (% choosing B × Base B EM)

| Model | A Selection | B Selection | Expected EM | Actual EM | ISV |
|-------|-------------|-------------|-------------|-----------|-----|
| Qwen3-32B | 99.85% | 0.15% | 11.3% | 11.3% | **0.0pp** |
| Llama-3.1-8B | 64.7% | 35.3% | 10.9% | 11.9% | **+1.0pp** |
| GPT-5-nano | 48.3% | 51.4% | 10.7% | 6.9% | **−3.8pp** |

**Interpretation:**
- **Qwen (ISV=0)**: Pure baseline reproduction—no added value from intelligent selection
- **Llama (ISV=+1.0pp)**: Positive value suggests modest discriminative capability
- **GPT-5-nano (ISV=−3.8pp)**: Negative value indicates the model actively selects the *wrong* parent more often than random chance

### 2.2 The MIX Gap

The near-absence of MIX resolutions (<1% across all models) reveals that multi-agent "improvement" is entirely attributable to the bypass mechanism rather than agentic merge generation:

1. **Exact match ceiling**: Maximum achievable EM is bounded by the union of correct answers in Parent A and Parent B
2. **Classification not generation**: The multi-agent functions as a parent classifier, not a merge generator
3. **Novel merge capability absent**: No model meaningfully attempts custom conflict resolution

---

## 3. Win/Tie/Loss Analysis

We classify each instance as a win (multi > single), tie, or loss (multi < single).

### 3.1 Exact Match Win/Tie/Loss (Per-Instance)

| Model | Wins | Ties | Losses | Win Rate |
|-------|------|------|--------|----------|
| Qwen3-32B | 214 (11.3%) | 1,674 (88.7%) | 0 (0.0%) | 11.3% |
| Llama-3.1-8B | 225 (11.9%) | 1,663 (88.1%) | 0 (0.0%) | 11.9% |
| GPT-5-nano | 105 (5.6%) | 1,711 (90.6%) | 72 (3.8%) | 5.6% |

**Observations:**
- Open-weight models (Qwen, Llama) show **zero regressions** because their single-agent produces 0% exact matches
- GPT-5-nano shows **3.8% regressions**—instances where single-agent achieved exact match but multi-agent did not

### 3.2 Similarity Win/Tie/Loss (Per-Instance)

| Model | Wins | Ties | Losses |
|-------|------|------|--------|
| Qwen3-32B | 1,805 (95.6%) | 0 (0.0%) | 83 (4.4%) |
| Llama-3.1-8B | 1,708 (90.5%) | 0 (0.0%) | 180 (9.5%) |
| GPT-5-nano | 966 (51.2%) | 14 (0.7%) | 908 (48.1%) |

**The similarity win/loss pattern reveals the bypass trade-off:** Multi-agent improves similarity for Qwen (95.6% wins) and Llama (90.5% wins), but GPT-5-nano sees near-parity (51.2% wins vs 48.1% losses). For capable models, bypassing to parents can be *worse* than the model's own generation.

---

## 4. Discussion

### 4.1 The Baseline Ceiling Problem

Multi-agent exact match rates are bounded by baseline performance. With Base A at 11.3% and Base B at 10.2%, no model can exceed ~11-12% EM through parent selection alone. The only path to higher performance is through successful MIX generation, which models essentially never attempt.

### 4.2 Qwen's Complete Collapse to Base A

Qwen's 99.85% A-selection with ISV=0 demonstrates complete failure of discriminative reasoning. Its multi-agent performance precisely equals Base A (11.3% = 11.3%), confirming that the system provides no value beyond simply selecting Parent A by default.

### 4.3 GPT-5-nano's Selection Paradox

Despite being the most capable model (highest single-agent EM at 5.2%), GPT-5-nano performs *worst* in multi-agent mode:
- Achieves only 6.9% EM vs 11.3% (Base A) or 10.2% (Base B)
- Negative ISV (−3.8pp) indicates worse-than-random parent selection
- The bypass mechanism actively harms performance for this model

This suggests that GPT-5-nano's own merge generation is often superior to either parent, but the multi-agent system overrides this capability with suboptimal parent selection.

### 4.4 Recommendations

1. **Randomize parent presentation order** to control for position bias
2. **Incentivize MIX generation** when neither parent is optimal
3. **Model-specific bypass thresholds**: stronger models may benefit from confident generation over conservative bypass
4. **Evaluate on oracle upper bound**: compute Union(Base A correct, Base B correct) to understand ceiling

---

## 5. Conclusion

Multi-agent approaches improve exact match rates over single-agent across all models. Analyzing 1,888 merge conflict instances across 342 repositories, we find:

1. **Improvement is entirely bypass-driven**: Models select parents rather than generate novel merges (<0.3% MIX usage)
2. **Qwen completely collapses to Base A**: 99.85% A-selection with no intelligent selection value
3. **Llama shows modest discrimination**: +1.0pp intelligent selection value, highest EM rate (11.9%)
4. **GPT-5-nano makes anti-optimal selections**: −3.8pp intelligent selection value, underperforming both baselines

The multi-agent system's value lies purely in parent selection, but only Llama demonstrates any positive selection intelligence. Future work should address position bias, encourage MIX generation, and consider model-specific bypass strategies.

---

## Appendix A: Per-File Supplementary Analysis

We also report per-file metrics (n = 3,272 files) to isolate file-level behavior. Per-file observations are clustered within instances, so these results are presented descriptively; formal claims are anchored to the per-instance analysis above.

**Note on interpretation:** 66.5% of instances are single-file, so per-file and per-instance metrics align for a majority of cases. The remaining 33.5% create a multiplicative penalty at the instance level that explains the ~3pp drop in exact match rates.

### A.1 Per-File Exact Match

| Method | Exact Matches | Rate |
|--------|---------------|------|
| Base A | 483 | 14.8% |
| Base B | 525 | 16.0% |

| Model | Single-Agent EM | Multi-Agent EM | Δ EM |
|-------|-----------------|----------------|------|
| Qwen3-32B | 0.0% | 14.8% | +14.8pp |
| Llama-3.1-8B | 0.0% | 15.8% | +15.8pp |
| GPT-5-nano | 6.8% | 11.3% | +4.5pp |

### A.2 Per-File Similarity Metrics

| Model | Single Sim | Multi Sim | Δ Sim |
|-------|------------|-----------|-------|
| Qwen3-32B | 0.569 | 0.896 | +57.4% |
| Llama-3.1-8B | 0.537 | 0.893 | +66.4% |
| GPT-5-nano | 0.864 | 0.853 | −1.3% |

### A.3 Per-File Win/Tie/Loss (Exact Match)

| Model | Wins | Ties | Losses |
|-------|------|------|--------|
| Qwen3-32B | 484 (14.8%) | 2,788 (85.2%) | 0 (0.0%) |
| Llama-3.1-8B | 517 (15.8%) | 2,755 (84.2%) | 0 (0.0%) |
| GPT-5-nano | 242 (7.4%) | 2,870 (87.7%) | 160 (4.9%) |

### A.4 Multi-File Penalty Effect

The difference between per-file and per-instance exact match rates arises from multi-file conflicts:

| Files per Instance | Count | % of Instances |
|-------------------|-------|----------------|
| 1 file | 1,255 | 66.5% |
| 2 files | 338 | 17.9% |
| 3 files | 129 | 6.8% |
| 4+ files | 166 | 8.8% |

For multi-file conflicts, perfect resolution requires all files correct. With per-file EM ≈ 15% and average 1.73 files/instance, the expected per-instance EM ≈ 0.15^1.73 × adjustment ≈ 11-12%, consistent with observed rates.

---

## Appendix B: Dataset and Methodology

### B.1 Dataset Composition

| Metric | Count | Description |
|--------|-------|-------------|
| **Unique repositories** | 342 | Distinct GitHub repositories contributing conflicts |
| **Unique merge conflicts (instances)** | 1,888 | Distinct merge conflict instances (by ID)—primary unit |
| **Unique file names** | 2,189 | Distinct file paths across all conflicts |
| **Total files** | 3,272 | Files across all conflicts—supplementary unit |

### B.2 Granularity Definitions

| Granularity | Unit | Exact Match Rule | Soft Metric Rule | N | Status |
|-------------|------|------------------|------------------|---|--------|
| **Per-Instance** | Merge conflict ID | ALL files match | Average across files | 1,888 | **Primary** |
| **Per-File** | Individual file | File matches | File-level metric | 3,272 | Supplementary |

**Rationale:** Per-instance is the primary metric because a merge conflict is only truly resolved when all constituent files are correct. Per-file metrics are useful for diagnosing file-level behavior and stratified analyses (e.g., by language, file size), but per-file observations are clustered within instances and should not be treated as independent.

### B.3 Generated Output Files

| File | Description |
|------|-------------|
| `rq1_model_summary_per_instance.csv` | Per-instance model metrics with 95% CI (primary) |
| `rq1_win_tie_loss_per_instance.csv` | Per-instance win/tie/loss analysis (primary) |
| `rq1_all_methods_per_instance.csv` | All methods comparison at instance level |
| `rq1_model_summary.csv` | Per-file model metrics with 95% CI (supplementary) |
| `rq1_win_tie_loss.csv` | Per-file win/tie/loss analysis (supplementary) |
| `rq1_bypass_distribution.csv` | Parent selection distribution |

*Note: All metrics computed on the aligned dataset of 1,888 instances (3,272 files) present across all evaluation conditions.*
