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

Total labeled samples analyzed: **180**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Complexity | 120 | 66.3% |
| Structural Change Bias | 69 | 38.1% |
| Favored Simplicity | 68 | 37.6% |
| Modification Bias | 58 | 32.0% |
| Vague Commit Message | 51 | 28.2% |
| Simple Commit Message | 43 | 23.8% |
| Unclear | 22 | 12.2% |
| Detailed Commit Message | 15 | 8.3% |
| Fix-Oriented | 9 | 5.0% |
| Feature-Oriented | 5 | 2.8% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.531** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Unclear | +0.636 | 63.6% | 22 | Strong Bypass advantage |
| Simple Commit Message | +0.605 | 60.5% | 43 | Strong Bypass advantage |
| Favored Complexity | +0.567 | 56.7% | 120 | Strong Bypass advantage |
| Modification Bias | +0.534 | 53.4% | 58 | Strong Bypass advantage |
| Detailed Commit Message | +0.533 | 53.3% | 15 | Strong Bypass advantage |
| Structural Change Bias | +0.493 | 49.3% | 69 | Strong Bypass advantage |
| Vague Commit Message | +0.471 | 47.1% | 51 | Strong Bypass advantage |
| Favored Simplicity | +0.412 | 41.2% | 68 | Strong Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Favored Simplicity | exact_match | 0.0002 | Highly significant (p < 0.001) |
| Structural Change Bias | exact_match | 0.0373 | Significant (p < 0.05) |
| Vague Commit Message | exact_match | 0.0444 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 107 | Agent wins: 0 | Ties: 74

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Simplicity | 26.2% | 54.1% | -27.9pp | Yes (p<0.05) |
| Structural Change Bias | 31.8% | 47.3% | -15.5pp | No |
| Vague Commit Message | 22.4% | 36.5% | -14.1pp | No |
| Modification Bias | 29.0% | 36.5% | -7.5pp | No |
| Favored Complexity | 63.6% | 70.3% | -6.7pp | No |

**Interpretation:** Labels like *Favored Simplicity* are significantly more common
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
