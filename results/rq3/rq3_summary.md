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

Total labeled samples analyzed: **1100**

| Label | Count | Percentage |
|-------|-------|------------|
| Favored Complexity | 567 | 51.5% |
| Favored Simplicity | 543 | 49.4% |
| Structural Change Bias | 375 | 34.1% |
| Simple Commit Message | 251 | 22.8% |
| Modification Bias | 247 | 22.5% |
| Detailed Commit Message | 225 | 20.5% |
| Vague Commit Message | 210 | 19.1% |
| Unclear | 114 | 10.4% |
| Fix-Oriented | 92 | 8.4% |
| Feature-Oriented | 59 | 5.4% |

---

## Performance Comparison: Bypass vs Agent

### Overall Finding

**Bypass outperforms Agent** on average across all labels.
- Average Delta EM: **+0.246** (Bypass advantage)

### Labels Where Bypass Wins (Delta EM > +0.05)

For samples with these labels, the **Bypass method significantly outperforms Agent**:

| Label | Delta EM | Win Rate | N | Interpretation |
|-------|----------|----------|---|----------------|
| Detailed Commit Message | +0.307 | 30.7% | 179 | Strong Bypass advantage |
| Unclear | +0.298 | 29.8% | 94 | Moderate Bypass advantage |
| Favored Complexity | +0.295 | 29.5% | 420 | Moderate Bypass advantage |
| Fix-Oriented | +0.278 | 27.8% | 79 | Moderate Bypass advantage |
| Vague Commit Message | +0.273 | 27.3% | 165 | Moderate Bypass advantage |
| Simple Commit Message | +0.273 | 27.3% | 198 | Moderate Bypass advantage |
| Modification Bias | +0.264 | 26.4% | 178 | Moderate Bypass advantage |
| Structural Change Bias | +0.260 | 26.0% | 292 | Moderate Bypass advantage |
| Refactor-Oriented | +0.235 | 23.5% | 17 | Moderate Bypass advantage |
| Feature-Oriented | +0.231 | 23.1% | 52 | Moderate Bypass advantage |

---

## Statistical Significance

Statistical tests compare the performance *difference* between samples with vs without each label.
A significant result (p < 0.05) means the label is associated with different method effectiveness.

### Statistically Significant Results (p < 0.05)

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Favored Simplicity | exact_match | 0.0000 | Highly significant (p < 0.001) |
| Unclear | bleu3 | 0.0032 | Very significant (p < 0.01) |
| Unclear | similarity | 0.0040 | Very significant (p < 0.01) |
| Unclear | rouge_l | 0.0041 | Very significant (p < 0.01) |
| Favored Complexity | exact_match | 0.0195 | Significant (p < 0.05) |
| Test-Oriented | exact_match | 0.0346 | Significant (p < 0.05) |

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

**Sample breakdown:** Bypass wins: 226 | Agent wins: 0 | Ties: 646

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Bypass Wins (vs Tie)

These labels appear more frequently when Bypass outperforms Agent than when they tie:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Complexity | 54.9% | 45.8% | +9.0pp | Yes (p<0.05) |
| Detailed Commit Message | 24.3% | 19.2% | +5.1pp | No |

**Interpretation:** Labels like *Favored Complexity* are significantly more common
when Bypass wins, suggesting these characteristics predict Bypass success.

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | Significant? |
|-------|------------------|-----------|------------|--------------|
| Favored Simplicity | 36.3% | 56.0% | -19.8pp | Yes (p<0.05) |

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
