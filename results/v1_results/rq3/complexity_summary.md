## Code Complexity Analysis

This section analyzes code complexity metrics across different methods 
and their relationship to merge resolution performance.

### Complexity by Method

Average complexity metrics across all analyzed samples:

| Method | SLOC | CC Avg | MI Score | Samples |
|--------|------|--------|----------|---------|
| a_only | 439 | 4.11 | 49.4 | 1086 |
| b_only | 438 | 4.13 | 50.1 | 1086 |
| ground_truth | 444 | 4.03 | 48.4 | 1100 |
| agent | 366 | 3.22 | 41.2 | 1094 |
| bypass | 438 | 4.10 | 49.9 | 1087 |

**Key Findings:**

- **Agent produces simpler code** (CC: 3.22) than Bypass (4.10)
- **Bypass has higher maintainability** (MI: 49.9) than Agent (41.2)
- **Bypass complexity is closer to ground truth**

### Complexity vs Performance Correlation

How does ground truth complexity predict which method performs better?

**Statistically Significant Correlations (p < 0.05):**

| Complexity Metric | Performance Metric | Spearman r | Interpretation |
|-------------------|-------------------|------------|----------------|
| sloc | delta_bleu3 | 0.790 | Higher complexity -> Bypass advantage |
| sloc | delta_similarity | 0.778 | Higher complexity -> Bypass advantage |
| lloc | delta_bleu3 | 0.772 | Higher complexity -> Bypass advantage |
| lloc | delta_similarity | 0.759 | Higher complexity -> Bypass advantage |
| cc_total | delta_bleu3 | 0.705 | Higher complexity -> Bypass advantage |
| h_bugs | delta_bleu3 | 0.700 | Higher complexity -> Bypass advantage |
| cc_total | delta_similarity | 0.692 | Higher complexity -> Bypass advantage |
| h_bugs | delta_similarity | 0.690 | Higher complexity -> Bypass advantage |
| mi_score | delta_bleu3 | -0.674 | Higher complexity -> Agent advantage |
| mi_score | delta_similarity | -0.668 | Higher complexity -> Agent advantage |

**Overall Finding:** More complex code (higher SLOC, CC) strongly correlates with 
Bypass advantage, suggesting multi-agent approaches excel at handling complex merges.

### Agent vs Bypass Complexity Differences

**Cyclomatic Complexity alignment with ground truth:**
- Agent closer: 185/863 (21.4%)
- Bypass closer: 678/863 (78.6%)

**Maintainability Index alignment with ground truth:**
- Agent closer: 257/863 (29.8%)
- Bypass closer: 606/863 (70.2%)


### Complexity by Model and Outcome

Comparison of code complexity across different LLM models and pass/failure outcomes:

**Agent Output Complexity:**

| Model | Outcome | SLOC | CC Avg | MI Score | Samples |
|-------|---------|------|--------|----------|---------|
| gpt5nano | failure | 478 | 3.82 | 43.1 | 311 |
| gpt5nano | pass | 314 | 3.42 | 57.1 | 110 |
| llama | pass | 159 | 1.55 | 30.2 | 183 |
| qwen3 | failure | 454 | 3.57 | 36.5 | 309 |
| qwen3 | pass | 262 | 3.17 | 47.1 | 181 |

**Pass vs Failure Findings:**

- **gpt5nano**: Failure cases have higher complexity (CC +0.40) than pass cases
  - Pass cases have better maintainability (MI +13.9)
- **qwen3**: Failure cases have higher complexity (CC +0.39) than pass cases
  - Pass cases have better maintainability (MI +10.6)

**Bypass Output Complexity:**

| Model | Outcome | SLOC | CC Avg | MI Score | Samples |
|-------|---------|------|--------|----------|---------|
| gpt5nano | failure | 513 | 4.37 | 47.2 | 305 |
| gpt5nano | pass | 361 | 3.53 | 56.3 | 112 |
| llama | pass | 272 | 3.43 | 57.5 | 183 |
| qwen3 | failure | 571 | 4.61 | 42.8 | 307 |
| qwen3 | pass | 300 | 3.80 | 54.6 | 180 |
