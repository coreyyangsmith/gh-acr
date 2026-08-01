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

Total labeled samples analyzed: **312**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Simplicity | 204 | 65.4% |
| Favored Complexity | 92 | 29.5% |
| Detailed Commit Message | 75 | 24.0% |
| Structural Change Bias | 70 | 22.4% |
| Simple Commit Message | 51 | 16.3% |
| Vague Commit Message | 49 | 15.7% |
| Unclear | 44 | 14.1% |
| Modification Bias | 37 | 11.9% |
| Fix-Oriented | 26 | 8.3% |
| Feature-Oriented | 23 | 7.4% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.109** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Misprioritization | +0.222 | 22.2% | 18 | Moderate Bypass advantage |
| Vague Commit Message | +0.184 | 18.4% | 49 | Moderate Bypass advantage |
| Unclear | +0.159 | 15.9% | 44 | Moderate Bypass advantage |
| Feature-Oriented | +0.130 | 13.0% | 23 | Slight Bypass advantage |
| Simple Commit Message | +0.118 | 11.8% | 51 | Slight Bypass advantage |
| Detailed Commit Message | +0.107 | 10.7% | 75 | Slight Bypass advantage |
| Favored Simplicity | +0.093 | 9.3% | 204 | Slight Bypass advantage |
| Fix-Oriented | +0.077 | 7.7% | 26 | Slight Bypass advantage |
| Structural Change Bias | +0.057 | 5.7% | 70 | Slight Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Favored Complexity | exact_match | 0.0002 | Highly significant (p < 0.001) |
| Modification Bias | exact_match | 0.0247 | Significant (p < 0.05) |
| Unclear | rouge_l | 0.0438 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 29 | Agent wins: 0 | Ties: 283

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Bypass Wins (vs Tie)

These labels appear more frequently when Bypass outperforms Agent than when they tie:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Vague Commit Message | 31.0% | 14.1% | +16.9pp | Yes (p<0.05) |
| Unclear | 24.1% | 13.1% | +11.1pp | No |
| Misprioritization | 13.8% | 4.9% | +8.8pp | No |

**Interpretation:** Labels like *Vague Commit Message* are significantly more common
when Bypass wins, suggesting these characteristics predict Bypass success.

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Complexity | 6.9% | 31.8% | -24.9pp | Yes (p<0.05) |
| Structural Change Bias | 13.8% | 23.3% | -9.5pp | No |
| Modification Bias | 3.4% | 12.7% | -9.3pp | No |

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
