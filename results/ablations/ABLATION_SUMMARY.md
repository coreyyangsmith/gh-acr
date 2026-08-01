# Better-Judge Ablation Analysis Summary

Anchor method: `better_judge`
Ablations: `bj_no_summary`, `bj_no_judge`, `bj_no_plan`, `bj_no_review`

## Component contributions (mean Δ = BJ − ablation)

| Model | Component | Metric | N | Mean Δ | 95% CI | p |
|-------|-----------|--------|---|--------|--------|---|
| gpt-5-nano | Summarizer | exact_match | 1356 | -0.0066 | [-0.0243, 0.0118] | 0.494 |
| gpt-5-nano | Summarizer | similarity | 1356 | -0.0053 | [-0.0135, 0.0024] | 0.00683 |
| gpt-5-nano | Analyzer (routing) | exact_match | 1356 | 0.0708 | [0.0520, 0.0904] | 1.94e-13 |
| gpt-5-nano | Analyzer (routing) | similarity | 1356 | 0.0189 | [0.0110, 0.0271] | 5.47e-10 |
| gpt-5-nano | Planner+Reviewer | exact_match | 1356 | 0.0155 | [-0.0026, 0.0314] | 0.0987 |
| gpt-5-nano | Planner+Reviewer | similarity | 1356 | 0.0005 | [-0.0076, 0.0082] | 0.00409 |
| gpt-5-nano | Reviewer | exact_match | 1356 | 0.0029 | [-0.0155, 0.0218] | 0.81 |
| gpt-5-nano | Reviewer | similarity | 1356 | -0.0048 | [-0.0130, 0.0030] | 0.000559 |
| llama-3.1-8b-instruct | Summarizer | exact_match | 2501 | 0.0128 | [0.0026, 0.0236] | 0.0206 |
| llama-3.1-8b-instruct | Summarizer | similarity | 2501 | 0.0066 | [0.0010, 0.0127] | 1.81e-05 |
| llama-3.1-8b-instruct | Analyzer (routing) | exact_match | 2501 | 0.1499 | [0.1351, 0.1657] | 3.88e-90 |
| llama-3.1-8b-instruct | Analyzer (routing) | similarity | 2501 | 0.0587 | [0.0513, 0.0661] | 1.5e-115 |
| llama-3.1-8b-instruct | Planner+Reviewer | exact_match | 2501 | 0.0012 | [-0.0044, 0.0060] | 0.749 |
| llama-3.1-8b-instruct | Planner+Reviewer | similarity | 2501 | -0.0016 | [-0.0052, 0.0018] | 0.286 |
| llama-3.1-8b-instruct | Reviewer | exact_match | 2501 | -0.0008 | [-0.0060, 0.0048] | 0.888 |
| llama-3.1-8b-instruct | Reviewer | similarity | 2501 | -0.0020 | [-0.0056, 0.0012] | 0.253 |

## Notes

- `bj_no_plan` removes **both** planner and reviewer (confounded); prefer `bj_no_review` to isolate the review loop.
- Positive Δ means Better-Judge outperforms the ablation (removing that component hurts).
