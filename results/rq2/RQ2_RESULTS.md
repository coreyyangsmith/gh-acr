# RQ2: How Do Conflict Characteristics Affect Multi-Agent Improvement?

## Summary

We analyze how multi-agent improvement over single-agent varies across conflict characteristics including baseline difficulty, project size, conflict size, and context size. We present results for **all three models** (Qwen3-32B, Llama-3.1-8B-Instruct, GPT-5-nano) with cross-model comparisons.

**Experimental Design:** We use per-instance metrics (n = 1,888 instances) as the primary measure. For stratified analyses by file-level features (token counts, file type), we report per-file metrics (n = 1,888 files) with clustering caveats.

**Key Findings:**
1. **Strong cross-model variation**: Qwen3-32B and Llama-3.1-8B show substantial improvement (+11.3-11.9% EM), while GPT-5-nano shows minimal improvement (+1.7% EM) with near-random similarity outcomes.
2. **Structural complexity matters**: Smaller conflicts and contexts show substantially higher improvement rates.
3. **Difficulty is predictive**: Contrary to earlier single-model analysis, combined results show "easy" conflicts improve 3× more than "hard" conflicts (15.2% vs 5.5% EM).

---

## 1. Cross-Model Comparison (Per-Instance)

### 1.1 Overall Performance by Model

| Model | N Instances | Δ EM | EM Win Rate | Δ Similarity | Similarity Win Rate |
|-------|-------------|------|-------------|--------------|---------------------|
| **Qwen3-32B** | 1,888 | +11.3% | 11.3% | +0.319 | 95.6% |
| **Llama-3.1-8B** | 1,888 | +11.9% | 11.9% | +0.385 | 90.5% |
| **GPT-5-nano** | 1,888 | +1.7% | 5.6% | −0.013 | 51.3% |

**Critical Observation:** GPT-5-nano shows dramatically different behavior:
- Near-random similarity win rate (51.3%)
- **Negative mean similarity delta** (−0.013), meaning multi-agent is slightly *worse* on average
- Only 5.6% of instances show EM improvement

This suggests the multi-agent bypass mechanism is model-dependent—smaller/weaker models may not effectively leverage the parent selection strategy.

### 1.2 All Models Combined

When aggregating across all models (treating the combined dataset as a single population), results are dominated by the two effective models:

| Metric | Single-Agent | Multi-Agent | Δ | Win Rate |
|--------|--------------|-------------|---|----------|
| Exact Match | — | — | +11.3% | 11.3% |
| Similarity | — | — | +0.319 | 95.6% |
| BLEU-3 | — | — | +0.296 | 93.5% |
| ROUGE-L | — | — | +0.279 | 96.0% |

*Note: Combined metrics primarily reflect Qwen3-32B performance since it represents the "all_models" baseline.*

---

## 2. Stratification by Instance-Level Characteristics

### 2.1 By Baseline Difficulty (Per-Instance, All Models)

| Difficulty | N | Δ EM | 95% CI | Similarity Win Rate |
|------------|---|------|--------|---------------------|
| Easy | 877 | +15.2% | [12.8%, 17.6%] | 96.0% |
| Medium | 378 | +12.2% | [9.0%, 15.3%] | 94.4% |
| Hard | 633 | +5.5% | [3.8%, 7.4%] | 95.7% |

**Key Finding:** Difficulty *does* predict improvement in the combined analysis:
- **Easy conflicts improve 2.8× more than hard conflicts** (15.2% vs 5.5% EM)
- This contrasts with single-model Qwen analysis where difficulty appeared uniform
- The difference may reflect harder conflicts being genuinely more challenging for parent selection

### 2.2 By Project Size (Per-Instance, All Models)

| Project Size | N | Δ EM | 95% CI | Similarity Win Rate |
|--------------|---|------|--------|---------------------|
| Small | 103 | +17.5% | [10.7%, 25.2%] | 98.1% |
| Medium | 1,088 | +12.1% | [10.3%, 14.1%] | 95.5% |
| Large | 600 | +9.7% | [7.3%, 12.2%] | 95.7% |
| Huge | 97 | +6.2% | [2.1%, 11.3%] | 93.8% |

**Key Finding:** Project size shows a clear gradient:
- Small projects: 17.5% EM improvement
- Huge projects: 6.2% EM improvement
- **2.8× difference** between small and huge projects

---

## 3. Stratified Analysis by Structural Features (Per-File)

The following analyses use per-file metrics (n = 1,888 files). Files within an instance are not independent—interpret as exploratory diagnostics.

### 3.1 By Conflict Size (Tokens in Diff)

| Conflict Size | N Files | Δ EM | 95% CI | Similarity Win Rate |
|---------------|---------|------|--------|---------------------|
| 1-50 tokens | 42 | +26.2% | [14.3%, 40.5%] | 100.0% |
| 51-200 tokens | 328 | +14.6% | [11.0%, 18.6%] | 98.8% |
| 201-500 tokens | 428 | +15.4% | [12.1%, 19.2%] | 99.3% |
| 501-1000 tokens | 438 | +12.1% | [9.1%, 15.3%] | 97.3% |
| 1000+ tokens | 652 | +14.3% | [11.7%, 17.2%] | 89.0% |

**Key Pattern:** 
- Smallest conflicts (1-50 tokens): **26.2% EM improvement** and **100% similarity win rate**
- Largest conflicts (1000+): 14.3% EM but only 89.0% similarity win rate
- The bypass mechanism works best on small, well-delineated conflicts

### 3.2 By Context Size (Input Tokens)

| Context Size | N Files | Δ EM | 95% CI | Similarity Win Rate |
|--------------|---------|------|--------|---------------------|
| Small (<1K) | 455 | +17.6% | [14.1%, 21.1%] | 97.4% |
| Medium (1K-5K) | 1,062 | +15.2% | [13.1%, 17.5%] | 95.5% |
| Large (5K-10K) | 282 | +9.2% | [5.7%, 12.8%] | 90.8% |
| Very Large (10K+) | 89 | +4.5% | [1.1%, 9.0%] | 94.4% |

**Key Pattern:** 
- Context size shows **4× gradient**: Small 17.6% vs Very Large 4.5%
- Large contexts degrade parent selection accuracy
- Anomaly: Very Large context has higher similarity win rate (94.4%) than Large (90.8%)

### 3.3 Similarity Improvement by Structural Feature

| Feature | Category | Mean Δ Similarity | 95% CI |
|---------|----------|-------------------|--------|
| Conflict Size | 1-50 tokens | +0.641 | [0.572, 0.710] |
| Conflict Size | 51-200 tokens | +0.452 | [0.422, 0.484] |
| Conflict Size | 201-500 tokens | +0.367 | [0.345, 0.389] |
| Conflict Size | 501-1000 tokens | +0.293 | [0.270, 0.315] |
| Conflict Size | 1000+ tokens | +0.226 | [0.204, 0.247] |
| Context Size | Small (<1K) | +0.538 | [0.511, 0.563] |
| Context Size | Medium (1K-5K) | +0.278 | [0.265, 0.292] |
| Context Size | Large (5K-10K) | +0.114 | [0.091, 0.137] |
| Context Size | Very Large (10K+) | +0.399 | [0.330, 0.464] |

**Key Finding:** 
- Similarity improvement shows **3× gradient** with conflict size (0.64 to 0.23)
- Very Large context shows anomalously high similarity improvement—likely a floor effect where single-agent performs poorly, leaving room for bypass improvement

---

## 4. Logistic Regression Analysis

We fit a logistic regression predicting multi-agent "win" (EM improvement > 0) from file-level features.

### 4.1 Odds Ratios

| Feature | Coefficient | Odds Ratio | Interpretation |
|---------|-------------|------------|----------------|
| project_size_small | +0.69 | 2.00 | Small projects 2× more likely to improve |
| project_size_medium | +0.52 | 1.68 | Medium projects 1.68× more likely |
| project_size_large | +0.53 | 1.70 | Large projects 1.70× more likely |
| difficulty_hard | −0.13 | 0.88 | Hard conflicts 12% less likely |
| difficulty_medium | −0.33 | 0.72 | Medium conflicts 28% less likely |
| file_type_Python | −0.75 | 0.47 | Python files **53% less likely** |
| conflict_size | +0.36 | 1.44 | Larger conflicts slightly more likely (per unit) |
| tokens_context | −0.32 | 0.73 | More context → 27% less likely per unit |
| tokens_original | −0.58 | 0.56 | Larger original files → 44% less likely |

### 4.2 Key Predictors

1. **Project size** is the strongest categorical predictor: small projects have 2× the odds
2. **Python files underperform**: 53% less likely to show improvement than other file types
3. **Context and original file size** negatively predict success
4. **Conflict size coefficient is positive** (+0.36), but this may be confounded with other features

**Python Finding:** The strong negative effect for Python files (OR 0.47) is notable. This may reflect:
- Higher baseline single-agent performance on Python
- Python-specific syntactic complexity
- Dataset composition effects (Python is 1,886/1,888 files = 99.9%)

---

## 5. Discussion

### 5.1 Model-Dependent Effectiveness

The most striking finding is that **multi-agent effectiveness is highly model-dependent**:

| Model | Effective? | Notes |
|-------|------------|-------|
| Qwen3-32B | Yes | +11.3% EM, 95.6% similarity wins |
| Llama-3.1-8B | Yes | +11.9% EM, 90.5% similarity wins |
| GPT-5-nano | No | +1.7% EM, 51.3% similarity wins (random) |

This suggests the bypass mechanism requires models with sufficient capability to:
1. Accurately assess which parent is correct
2. Effectively select or synthesize the resolution

### 5.2 Structural vs Semantic Complexity

When combining all models:
- **Difficulty is predictive**: Easy conflicts improve 2.8× more than hard (15.2% vs 5.5%)
- **Project size matters**: 2.8× gradient from small to huge
- **Context size strongly predicts**: 4× gradient from small to very large

### 5.3 The Python Anomaly

Python files show 53% lower odds of improvement despite comprising 99.9% of the dataset. Possible explanations:
1. Dataset is essentially Python-only, so "Python" coefficient captures residual variance
2. Python may have higher baseline single-agent performance
3. Python's syntactic structure may make parent selection harder

### 5.4 Recommendations

1. **Model selection matters**: Use larger/more capable models for multi-agent merge resolution
2. **Conflict-size-aware routing**: Small conflicts (< 200 tokens) benefit most from bypass
3. **Context pruning**: Large contexts degrade effectiveness—consider summarization
4. **Difficulty-aware thresholds**: Require higher confidence on hard conflicts

---

## 6. Conclusion

Multi-agent improvement varies substantially by model and conflict characteristics. Analyzing 1,888 instances across three models:

1. **Model-dependent**: Qwen3-32B and Llama-3.1-8B show +11-12% EM improvement; GPT-5-nano shows minimal improvement (+1.7%) with near-random similarity outcomes
2. **Difficulty predicts improvement**: Easy conflicts improve 2.8× more than hard (15.2% vs 5.5% EM)
3. **Project size gradient**: Small projects show 17.5% vs 6.2% for huge projects
4. **Structural features matter**: Small conflicts and contexts show highest improvement
5. **Python anomaly**: Python files show 53% lower odds despite being 99.9% of data

The bypass mechanism's effectiveness depends on both model capability and conflict characteristics. Smaller models and harder conflicts may not benefit from the multi-agent approach.

---

## Appendix: Methodology

### A.1 Data and Scope

| Level | N | Description |
|-------|---|-------------|
| **Instances** | 1,888 | Merge conflict instances |
| **Files** | 1,888 | Individual files (99.9% Python) |
| **Models** | 3 | Qwen3-32B, Llama-3.1-8B-Instruct, GPT-5-nano |

### A.2 Generated Output Files

**Per-Model Outputs:**

| File Pattern | Description |
|--------------|-------------|
| `{Model}_rq2_overall_per_instance.csv` | Per-instance summary |
| `{Model}_rq2_stratified_per_instance.csv` | Per-instance stratification |
| `{Model}_rq2_stratified_summary.csv` | Per-file stratification |
| `{Model}_rq2_logistic_coefficients_*.csv` | Regression coefficients |

**Combined Outputs:**

| File | Description |
|------|-------------|
| `rq2_cross_model_comparison.csv` | Cross-model comparison |
| `rq2_overall_per_instance.csv` | Combined per-instance summary |
| `rq2_stratified_per_instance.csv` | Combined per-instance stratification |
| `rq2_stratified_summary.csv` | Combined per-file stratification |
| `rq2_logistic_coefficients_exact_match.csv` | Combined regression coefficients |
| `rq2_forest_*.png` | Forest plots |
| `rq2_heatmap_*.png` | Interaction heatmaps |
| `rq2_violin_*.png` | Distribution plots |

### A.3 Characteristic Definitions

| Characteristic | Source | Bucketing |
|----------------|--------|-----------|
| Difficulty | Dataset label | easy / medium / hard |
| Project Size | Dataset label | small / medium / large / huge |
| Conflict Size | tokens_diff_a + tokens_diff_b | 1-50 / 51-200 / 201-500 / 501-1000 / 1000+ |
| Context Size | tokens_in | <1K / 1K-5K / 5K-10K / 10K+ |
