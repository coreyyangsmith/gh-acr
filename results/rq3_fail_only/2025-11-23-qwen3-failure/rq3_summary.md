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

Total labeled samples analyzed: **309**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Complexity | 153 | 49.5% |
| Favored Simplicity | 148 | 47.9% |
| Structural Change Bias | 115 | 37.2% |
| Simple Commit Message | 89 | 28.8% |
| Modification Bias | 73 | 23.6% |
| Vague Commit Message | 63 | 20.4% |
| Detailed Commit Message | 28 | 9.1% |
| Fix-Oriented | 26 | 8.4% |
| Unclear | 22 | 7.1% |
| Feature-Oriented | 13 | 4.2% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.052** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Fix-Oriented | +0.115 | 11.5% | 26 | Slight Bypass advantage |
| Feature-Oriented | +0.077 | 7.7% | 13 | Slight Bypass advantage |
| Favored Simplicity | +0.074 | 7.4% | 148 | Slight Bypass advantage |
| Vague Commit Message | +0.063 | 6.3% | 63 | Slight Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Detailed Commit Message | exact_match | 0.0001 | Highly significant (p < 0.001) |
| Favored Complexity | exact_match | 0.0186 | Significant (p < 0.05) |
| Favored Simplicity | exact_match | 0.0480 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 15 | Agent wins: 0 | Ties: 294

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Bypass Wins (vs Tie)

These labels appear more frequently when Bypass outperforms Agent than when they tie:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Simplicity | 73.3% | 46.6% | +26.7pp | No |
| Fix-Oriented | 20.0% | 7.8% | +12.2pp | No |
| Vague Commit Message | 26.7% | 20.1% | +6.6pp | No |

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Complexity | 20.0% | 51.0% | -31.0pp | Yes (p<0.05) |
| Structural Change Bias | 26.7% | 37.8% | -11.1pp | No |
| Detailed Commit Message | 0.0% | 9.5% | -9.5pp | No |

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
