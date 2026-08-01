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

## Global Method Comparison (McNemar Test)

Paired test on exact match: we only use *discordant* pairs (one method correct, the other wrong).
Under the null that methods are equivalent, the number of pairs where Bypass is correct and Agent is wrong
should be symmetric with the reverse. One-sided exact binomial test: **Bypass is better than Agent on EM**.

| Quantity | Value |
|----------|-------|
| N (total pairs) | 872 |
| Agent correct, Bypass wrong (b) | 0 |
| Agent wrong, Bypass correct (c) | 226 |
| Discordant pairs (b + c) | 226 |
| P-value (exact binomial, one-sided) | 9.27e-69 |

**Conclusion:** Bypass is significantly better than Agent on exact match (p < 0.05).

---

## Selector Quality: Chosen vs Rejected Diff (McNemar Test)

This test evaluates the Bypass *selector* directly, independent of the single-agent baseline.
For each sample, Bypass produces two candidate diffs (A and B) and a selector picks one.
We compare the **chosen** diff's exact match against the **rejected** diff's exact match.

Discordant pairs:
- **b** = chosen correct, rejected wrong (selector picks the better diff)
- **c** = chosen wrong, rejected correct (selector picks the worse diff)

Under H0 (selector is random), b and c are symmetric.
One-sided exact binomial: H1 = selector wins more often than chance.

**Metric:** exact_match

| Quantity | Value |
|----------|-------|
| N (samples with both candidates) | 1907 |
| Both correct | 4 |
| Both wrong | 1388 |
| b: chosen correct, rejected wrong (selector wins) | 284 |
| c: chosen wrong, rejected correct (selector loses) | 231 |
| Discordant pairs (b + c) | 515 |
| P-value (exact binomial, one-sided H1: b > c) | 0.0109 |

**Conclusion:** The selector picks the better diff significantly more often than chance (p < 0.05). b=284 vs c=231.

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

### Mann-Whitney U Test

The Mann-Whitney U test is a non-parametric alternative to the T-test. It compares
the *distribution* of performance deltas (Bypass − Agent) between samples with vs
without each label, without assuming normality. We report it alongside the T-test
because performance metrics may not be normally distributed.

**Statistically significant Mann-Whitney results (p < 0.05):**

| Label | Metric | P-value | Significance Level |
|-------|--------|---------|-------------------|
| Favored Simplicity | exact_match | 3.22e-07 | Highly significant (p < 0.001) |
| Unclear | rouge_l | 0.0015 | Very significant (p < 0.01) |
| Unclear | similarity | 0.0017 | Very significant (p < 0.01) |
| Unclear | bleu3 | 0.0021 | Very significant (p < 0.01) |
| Favored Complexity | exact_match | 0.0192 | Significant (p < 0.05) |

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

**Metric:** Winner is determined by **exact match** (Bypass wins if Bypass EM > Agent EM; tie if equal).
The Chi-squared tests assess whether each label is more/less prevalent in Bypass-winning samples vs Tie samples.

**Sample breakdown:** Bypass wins: 226 | Agent wins: 0 | Ties: 646

*Note: Agent rarely wins on exact match, so we compare Bypass wins vs Ties.*

### Labels More Common When Bypass Wins (vs Tie)

These labels appear more frequently when Bypass outperforms Agent than when they tie:

| Label | % in Bypass Wins | % in Ties | Difference | P-value | Significant? |
|-------|------------------|-----------|------------|--------|--------------|
| Favored Complexity | 54.9% | 45.8% | +9.0pp | 0.0235 | Yes (p<0.05) |
| Detailed Commit Message | 24.3% | 19.2% | +5.1pp | 0.1208 | No |

**Interpretation:** Labels like *Favored Complexity* are significantly more common
when Bypass wins, suggesting these characteristics predict Bypass success.

### Labels More Common When Methods Tie (vs Bypass Win)

These labels appear more frequently when methods tie than when Bypass wins:

| Label | % in Bypass Wins | % in Ties | Difference | P-value | Significant? |
|-------|------------------|-----------|------------|--------|--------------|
| Favored Simplicity | 36.3% | 56.0% | -19.8pp | 4.76e-07 | Yes (p<0.05) |

**Interpretation:** Labels like *Favored Simplicity* are significantly more common
when methods tie, suggesting Bypass provides no advantage for these cases.

See `label_winner_correlation.csv` for the full breakdown including odds ratios and p-values.

---

## Per-Label Improvement Analysis (Fisher's Exact)

For each label we test whether the label changes the *probability of improvement* (Bypass EM > Agent EM).
**improve** = 1 if Bypass wins, 0 otherwise (tie or Agent wins). Fisher's exact test on the 2×2 table
(label present × improve). Effect sizes: risk difference, relative risk, and Haldane-Anscombe corrected odds ratio.

| Label | P(improve given label) | P(improve given no label) | Risk Diff | Rel. Risk | OR (HA) | Fisher p | Sig? |
|-------|--------------------|-------------------------|------------|-----------|---------|---------|------|
| Favored Complexity | 0.295 | 0.226 | +0.070 | 1.31 | 1.44 | 0.0204 | Yes |
| Detailed Commit Message | 0.307 | 0.247 | +0.061 | 1.25 | 1.36 | 0.1044 | No |
| Unclear | 0.298 | 0.254 | +0.043 | 1.17 | 1.25 | 0.3835 | No |
| Fix-Oriented | 0.278 | 0.257 | +0.021 | 1.08 | 1.13 | 0.6872 | No |
| Simple Commit Message | 0.273 | 0.255 | +0.018 | 1.07 | 1.10 | 0.6449 | No |
| Vague Commit Message | 0.273 | 0.256 | +0.017 | 1.07 | 1.10 | 0.6932 | No |
| Modification Bias | 0.264 | 0.258 | +0.006 | 1.02 | 1.04 | 0.8486 | No |
| Structural Change Bias | 0.260 | 0.259 | +0.002 | 1.01 | 1.01 | 1.0000 | No |
| Refactor-Oriented | 0.235 | 0.260 | -0.024 | 0.91 | 0.95 | 1.0000 | No |
| Feature-Oriented | 0.231 | 0.261 | -0.030 | 0.88 | 0.87 | 0.7448 | No |
| Misprioritization | 0.222 | 0.260 | -0.038 | 0.85 | 0.88 | 1.0000 | No |
| Favored Simplicity | 0.185 | 0.336 | -0.152 | 0.55 | 0.45 | 3.12e-07 | Yes |
| Test-Oriented | 0.077 | 0.262 | -0.185 | 0.29 | 0.34 | 0.2020 | No |

See `label_improvement_tests.csv` for full counts and `rq3_label_improvement_forest.png` for a forest plot of risk differences.

---

## Glossary

| Term | Definition |
|------|------------|
| Delta | Difference between Bypass and Agent scores (Bypass - Agent) |
| Exact Match (EM) | Binary metric: 1 if output exactly matches ground truth, 0 otherwise |
| Similarity | Continuous metric (0-1) measuring how similar output is to ground truth |
| Win Rate | Percentage of samples where one method outperformed the other |
| P-value | Probability of observing this difference by chance (lower = more significant) |
| T-test | Parametric test comparing means of two groups (assumes normality) |
| Mann-Whitney U | Non-parametric test comparing distributions of two groups (no normality assumption) |
| Chi-squared | Test for association between categorical variables (e.g., label prevalence vs winner type) |
