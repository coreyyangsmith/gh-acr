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

Total labeled samples analyzed: **926**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Simplicity | 539 | 58.1% |
| Favored Complexity | 379 | 40.9% |
| Structural Change Bias | 267 | 28.8% |
| Simple Commit Message | 237 | 25.6% |
| Modification Bias | 202 | 21.8% |
| Vague Commit Message | 179 | 19.3% |
| Detailed Commit Message | 162 | 17.5% |
| Fix-Oriented | 70 | 7.6% |
| Unclear | 69 | 7.4% |
| Feature-Oriented | 52 | 5.6% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.068** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Misprioritization | +0.222 | 22.2% | 18 | Moderate Bypass advantage |
| Unclear | +0.127 | 12.7% | 63 | Slight Bypass advantage |
| Feature-Oriented | +0.089 | 8.9% | 45 | Slight Bypass advantage |
| Vague Commit Message | +0.079 | 7.9% | 151 | Slight Bypass advantage |
| Fix-Oriented | +0.077 | 7.7% | 65 | Slight Bypass advantage |
| Favored Simplicity | +0.060 | 6.0% | 466 | Slight Bypass advantage |
| Detailed Commit Message | +0.060 | 6.0% | 134 | Slight Bypass advantage |
| Refactor-Oriented | +0.059 | 5.9% | 17 | Slight Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Test-Oriented | exact_match | 0.0000 | Highly significant (p < 0.001) |
| Favored Complexity | exact_match | 0.0000 | Highly significant (p < 0.001) |
| Structural Change Bias | exact_match | 0.0153 | Significant (p < 0.05) |
| Unclear | rouge_l | 0.0216 | Significant (p < 0.05) |
| Unclear | similarity | 0.0283 | Significant (p < 0.05) |
| Modification Bias | exact_match | 0.0297 | Significant (p < 0.05) |
| Structural Change Bias | similarity | 0.0363 | Significant (p < 0.05) |
| Unclear | bleu3 | 0.0412 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 41 | Agent wins: 0 | Ties: 737

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Bypass Wins (vs Tie)

These labels appear more frequently when Bypass outperforms Agent than when they tie:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Unclear | 19.5% | 7.5% | +12.0pp | Yes (p<0.05) |
| Vague Commit Message | 29.3% | 18.9% | +10.4pp | No |
| Favored Simplicity | 68.3% | 59.4% | +8.9pp | No |
| Misprioritization | 9.8% | 1.9% | +7.9pp | Yes (p<0.05) |

**Interpretation:** Labels like *Unclear, Misprioritization* are significantly more common
when Bypass wins, suggesting these characteristics predict Bypass success.

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Complexity | 9.8% | 40.8% | -31.1pp | Yes (p<0.05) |
| Structural Change Bias | 14.6% | 29.6% | -14.9pp | No |
| Modification Bias | 9.8% | 20.9% | -11.1pp | No |

**Interpretation:** Labels like *Favored Complexity* are significantly more common
when methods tie, suggesting Bypass provides no advantage for these cases.

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
