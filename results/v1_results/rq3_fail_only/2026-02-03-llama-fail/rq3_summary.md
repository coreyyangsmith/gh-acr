# RQ3 Classification Analysis Summary

## How to Read This Report

This report compares the performance of two code conflict resolution methods:
- **Agent**: Single-agent approach
- **Bypass (Multi-Agent)**: Multi-agent approach with bypass mechanism

**Key Metrics:**
- **Delta EM (Exact Match)**: Bypass score minus Agent score. *Positive = Bypass is better*
- **Delta Similarity**: Same interpretation. Range: -1 to +1
- **Win Rate**: Percentage of samples where Bypass outperformed Agent
- **P-value**: Statistical significance. *p < 0.05 = statistically significant difference*

---

## Label Distribution

Total labeled samples analyzed: **306**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Simplicity | 187 | 61.1% |
| Favored Complexity | 134 | 43.8% |
| Simple Commit Message | 97 | 31.7% |
| Modification Bias | 92 | 30.1% |
| Structural Change Bias | 82 | 26.8% |
| Vague Commit Message | 67 | 21.9% |
| Detailed Commit Message | 59 | 19.3% |
| Fix-Oriented | 18 | 5.9% |
| Feature-Oriented | 16 | 5.2% |
| Refactor-Oriented | 8 | 2.6% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Performance is similar** between Agent and Bypass on average.
- Average Delta EM: **0.000**

### No Strong Method Preference

No labels showed a strong preference (|Delta EM| > 0.05) for either method.

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Favored Simplicity | similarity | 0.0259 | Significant (p < 0.05) |
| Favored Simplicity | bleu3 | 0.0386 | Significant (p < 0.05) |
| Favored Simplicity | rouge_l | 0.0448 | Significant (p < 0.05) |

**Interpretation:** These labels show statistically significant differences in how
the two methods (Agent vs Bypass) perform. This suggests these labels are
predictive of which method will work better for a given sample.

---

## Stratified Analysis

Performance breakdown by task characteristics:

### By Difficulty Level

How does method performance vary across difficulty levels for each label?

See `stratified_difficulty.csv` for the full breakdown. Key columns:
- `delta_exact_match`: Bypass - Agent score (positive = Bypass better)
- `bypass_win_rate_exact_match`: % of samples where Bypass won

### By Project Size

How does method performance vary across project sizes for each label?

See `stratified_project_size.csv` for the full breakdown.

---

## Label Distribution by Winner

Which labels are more common when Bypass wins vs when methods tie (or Agent wins)?
This reveals what characteristics predict Bypass success.

**Sample breakdown:** Bypass wins: 0 | Agent wins: 0 | Ties: 306

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Simplicity | 0.0% | 61.1% | -61.1pp | No |
| Favored Complexity | 0.0% | 43.8% | -43.8pp | No |
| Simple Commit Message | 0.0% | 31.7% | -31.7pp | No |
| Modification Bias | 0.0% | 30.1% | -30.1pp | No |
| Structural Change Bias | 0.0% | 26.8% | -26.8pp | No |
| Vague Commit Message | 0.0% | 21.9% | -21.9pp | No |
| Detailed Commit Message | 0.0% | 19.3% | -19.3pp | No |
| Fix-Oriented | 0.0% | 5.9% | -5.9pp | No |
| Feature-Oriented | 0.0% | 5.2% | -5.2pp | No |

See `label_winner_correlation.csv` for the full breakdown including odds ratios and p-values.

---

## Glossary

| Term | Definition |
|------|------------|
| Delta | Difference between Bypass and Agent scores (Bypass - Agent) |
| Exact Match (EM) | Binary metric: 1 if output exactly matches ground truth, 0 otherwise |
| Similarity | Continuous metric (0-1) measuring how similar output is to ground truth |
| Win Rate | Percentage of samples where one method outperformed the other |
| P-value | Probability of observing this difference by chance (lower = more significant) |
| T-test | Statistical test comparing means of two groups |
