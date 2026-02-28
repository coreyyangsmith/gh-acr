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

Total labeled samples analyzed: **186**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Complexity | 138 | 74.2% |
| Favored Simplicity | 77 | 41.4% |
| Structural Change Bias | 77 | 41.4% |
| Detailed Commit Message | 66 | 35.5% |
| Modification Bias | 58 | 31.2% |
| Simple Commit Message | 42 | 22.6% |
| Vague Commit Message | 34 | 18.3% |
| Fix-Oriented | 18 | 9.7% |
| Unclear | 8 | 4.3% |
| Feature-Oriented | 6 | 3.2% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.730** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Favored Complexity | +0.775 | 77.5% | 138 | Strong Bypass advantage |
| Detailed Commit Message | +0.773 | 77.3% | 66 | Strong Bypass advantage |
| Vague Commit Message | +0.765 | 76.5% | 34 | Strong Bypass advantage |
| Structural Change Bias | +0.740 | 74.0% | 77 | Strong Bypass advantage |
| Fix-Oriented | +0.722 | 72.2% | 18 | Strong Bypass advantage |
| Favored Simplicity | +0.701 | 70.1% | 77 | Strong Bypass advantage |
| Simple Commit Message | +0.690 | 69.0% | 42 | Strong Bypass advantage |
| Modification Bias | +0.672 | 67.2% | 58 | Strong Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Modification Bias | exact_match | 0.0164 | Significant (p < 0.05) |
| Favored Simplicity | exact_match | 0.0165 | Significant (p < 0.05) |
| Favored Complexity | bleu3 | 0.0292 | Significant (p < 0.05) |
| Favored Complexity | similarity | 0.0347 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 147 | Agent wins: 0 | Ties: 39

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Simplicity | 36.7% | 59.0% | -22.2pp | Yes (p<0.05) |
| Modification Bias | 26.5% | 48.7% | -22.2pp | Yes (p<0.05) |
| Simple Commit Message | 19.7% | 33.3% | -13.6pp | No |
| Structural Change Bias | 38.8% | 51.3% | -12.5pp | No |
| Favored Complexity | 72.8% | 79.5% | -6.7pp | No |

**Interpretation:** Labels like *Favored Simplicity, Modification Bias* are significantly more common
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
