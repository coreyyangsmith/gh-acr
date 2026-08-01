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

Total labeled samples analyzed: **111**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Complexity | 64 | 57.1% |
| Favored Simplicity | 46 | 41.1% |
| Structural Change Bias | 44 | 39.3% |
| Detailed Commit Message | 41 | 36.6% |
| Simple Commit Message | 26 | 23.2% |
| Modification Bias | 21 | 18.8% |
| Unclear | 18 | 16.1% |
| Vague Commit Message | 13 | 11.6% |
| Fix-Oriented | 13 | 11.6% |
| Feature-Oriented | 12 | 10.7% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.358** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Vague Commit Message | +0.462 | 46.2% | 13 | Strong Bypass advantage |
| Simple Commit Message | +0.462 | 46.2% | 26 | Strong Bypass advantage |
| Structural Change Bias | +0.409 | 40.9% | 44 | Strong Bypass advantage |
| Fix-Oriented | +0.385 | 38.5% | 13 | Strong Bypass advantage |
| Detailed Commit Message | +0.341 | 34.1% | 41 | Strong Bypass advantage |
| Feature-Oriented | +0.333 | 33.3% | 12 | Strong Bypass advantage |
| Favored Simplicity | +0.326 | 32.6% | 46 | Strong Bypass advantage |
| Favored Complexity | +0.297 | 29.7% | 64 | Moderate Bypass advantage |
| Modification Bias | +0.286 | 28.6% | 21 | Moderate Bypass advantage |
| Unclear | +0.278 | 27.8% | 18 | Moderate Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Favored Complexity | exact_match | 0.0178 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 44 | Agent wins: 0 | Ties: 68

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Bypass Wins (vs Tie)

These labels appear more frequently when Bypass outperforms Agent than when they tie:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Simple Commit Message | 27.3% | 20.6% | +6.7pp | No |

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Complexity | 43.2% | 66.2% | -23.0pp | Yes (p<0.05) |
| Favored Simplicity | 34.1% | 45.6% | -11.5pp | No |
| Modification Bias | 13.6% | 22.1% | -8.4pp | No |
| Detailed Commit Message | 31.8% | 39.7% | -7.9pp | No |
| Unclear | 11.4% | 19.1% | -7.8pp | No |

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
