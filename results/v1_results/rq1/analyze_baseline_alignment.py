"""Analyze baseline alignment with model scenarios.

This script investigates why Qwen's multi-agent (14.7% EM) exceeds 
Base A (10.7% EM) even though Qwen picks A 99.85% of the time.
"""

import pandas as pd
from pathlib import Path

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "2026_01_results_combined.csv")

print("=" * 70)
print("ANALYSIS: Baseline Alignment with Model Scenarios")
print("=" * 70)

# 1. Get unique scenario keys for each method
def get_scenario_keys(df, eval_method, model_name=None):
    """Get set of (id, file_name) tuples for a method."""
    mask = df["eval_method"] == eval_method
    if model_name:
        mask &= df["model_name"] == model_name
    subset = df[mask]
    return set(zip(subset["id"], subset["file_name"]))

# Get baseline scenarios
base_a_scenarios = get_scenario_keys(df, "base_a")
base_b_scenarios = get_scenario_keys(df, "base_b")

# Get model scenarios
qwen_scenarios = get_scenario_keys(df, "bypass7", "groq:qwen/qwen3-32b")
llama_scenarios = get_scenario_keys(df, "bypass7", "local:meta-llama/Llama-3.1-8B-Instruct")
gpt_scenarios = get_scenario_keys(df, "bypass7", "openai/gpt-5-nano")

print(f"\nScenario counts:")
print(f"  Base A:        {len(base_a_scenarios):,}")
print(f"  Base B:        {len(base_b_scenarios):,}")
print(f"  Qwen bypass7:  {len(qwen_scenarios):,}")
print(f"  Llama bypass7: {len(llama_scenarios):,}")
print(f"  GPT bypass7:   {len(gpt_scenarios):,}")

# 2. Compute baseline EM rates on each model's scenario subset
def get_baseline_em_on_subset(df, baseline_method, scenario_keys):
    """Compute baseline exact match rate on specific scenario subset."""
    baseline_df = df[df["eval_method"] == baseline_method].copy()
    baseline_df["key"] = list(zip(baseline_df["id"], baseline_df["file_name"]))
    subset = baseline_df[baseline_df["key"].isin(scenario_keys)]
    if len(subset) == 0:
        return 0, 0
    em_count = subset["exact_match"].sum()
    em_rate = em_count / len(subset) * 100
    return em_rate, len(subset)

print(f"\n" + "=" * 70)
print("Baseline Exact Match Rates on Model-Specific Scenario Subsets")
print("=" * 70)

for model_name, scenarios, label in [
    ("groq:qwen/qwen3-32b", qwen_scenarios, "Qwen"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", llama_scenarios, "Llama"),
    ("openai/gpt-5-nano", gpt_scenarios, "GPT-5-nano"),
]:
    base_a_em, base_a_n = get_baseline_em_on_subset(df, "base_a", scenarios)
    base_b_em, base_b_n = get_baseline_em_on_subset(df, "base_b", scenarios)
    
    print(f"\n{label} scenarios ({len(scenarios):,} total):")
    print(f"  Base A on this subset: {base_a_em:.1f}% EM (n={base_a_n:,})")
    print(f"  Base B on this subset: {base_b_em:.1f}% EM (n={base_b_n:,})")

# 3. Compare with model's actual multi-agent performance
print(f"\n" + "=" * 70)
print("Model Multi-Agent vs Matched Baseline Comparison")
print("=" * 70)

for model_name, scenarios, label in [
    ("groq:qwen/qwen3-32b", qwen_scenarios, "Qwen"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", llama_scenarios, "Llama"),
    ("openai/gpt-5-nano", gpt_scenarios, "GPT-5-nano"),
]:
    # Get model's multi-agent EM
    model_bypass = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_name)]
    model_em = model_bypass["exact_match"].sum() / len(model_bypass) * 100
    
    # Get matched baseline
    base_a_em, _ = get_baseline_em_on_subset(df, "base_a", scenarios)
    base_b_em, _ = get_baseline_em_on_subset(df, "base_b", scenarios)
    
    # Get bypass method distribution
    a_count = (model_bypass["bypass_method"] == "A").sum()
    b_count = (model_bypass["bypass_method"] == "B").sum()
    mix_count = (model_bypass["bypass_method"] == "MIX").sum()
    
    print(f"\n{label}:")
    print(f"  Bypass distribution: A={a_count:,} ({a_count/len(model_bypass)*100:.1f}%), "
          f"B={b_count:,} ({b_count/len(model_bypass)*100:.1f}%), MIX={mix_count}")
    print(f"  Multi-agent EM: {model_em:.1f}%")
    print(f"  Base A (matched): {base_a_em:.1f}%")
    print(f"  Base B (matched): {base_b_em:.1f}%")
    
    # Expected EM if random selection with same A/B ratio
    expected_em = (a_count * base_a_em + b_count * base_b_em) / len(model_bypass)
    print(f"  Expected EM (if selection = weighted baseline): {expected_em:.1f}%")
    print(f"  Actual - Expected: {model_em - expected_em:+.1f}pp (intelligent selection value)")

# 4. Check overlap between exact match scenarios
print(f"\n" + "=" * 70)
print("Analysis: Are models selecting A/B when they're correct?")
print("=" * 70)

for model_name, label in [
    ("groq:qwen/qwen3-32b", "Qwen"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ("openai/gpt-5-nano", "GPT-5-nano"),
]:
    model_bypass = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_name)].copy()
    model_bypass["key"] = list(zip(model_bypass["id"], model_bypass["file_name"]))
    
    # For each row, check if the selected parent was an exact match
    base_a_em_keys = set()
    base_b_em_keys = set()
    
    for _, row in df[df["eval_method"] == "base_a"].iterrows():
        if row["exact_match"]:
            base_a_em_keys.add((row["id"], row["file_name"]))
    
    for _, row in df[df["eval_method"] == "base_b"].iterrows():
        if row["exact_match"]:
            base_b_em_keys.add((row["id"], row["file_name"]))
    
    # Count correct selections
    correct_a_selections = 0  # Picked A when A was correct
    correct_b_selections = 0  # Picked B when B was correct
    incorrect_a_selections = 0  # Picked A when A was wrong
    incorrect_b_selections = 0  # Picked B when B was wrong
    
    for _, row in model_bypass.iterrows():
        key = (row["id"], row["file_name"])
        if row["bypass_method"] == "A":
            if key in base_a_em_keys:
                correct_a_selections += 1
            else:
                incorrect_a_selections += 1
        elif row["bypass_method"] == "B":
            if key in base_b_em_keys:
                correct_b_selections += 1
            else:
                incorrect_b_selections += 1
    
    print(f"\n{label}:")
    print(f"  Selected A when A was correct: {correct_a_selections:,}")
    print(f"  Selected A when A was wrong: {incorrect_a_selections:,}")
    print(f"  Selected B when B was correct: {correct_b_selections:,}")
    print(f"  Selected B when B was wrong: {incorrect_b_selections:,}")
    
    total_a = correct_a_selections + incorrect_a_selections
    total_b = correct_b_selections + incorrect_b_selections
    if total_a > 0:
        print(f"  A selection accuracy: {correct_a_selections/total_a*100:.1f}%")
    if total_b > 0:
        print(f"  B selection accuracy: {correct_b_selections/total_b*100:.1f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The discrepancy arises because:
1. Baselines (base_a, base_b) are evaluated on ALL 4,602 scenarios
2. Each model is tested on a DIFFERENT subset of ~3,300 scenarios
3. The subset each model is tested on may have different baseline EM rates

This means the 9.9% Base A EM rate is NOT comparable to Qwen's 14.7% 
multi-agent EM rate - they're computed on different populations!

For fair comparison, we should compute baseline EM rates on the 
SAME scenarios each model was tested on.
""")
