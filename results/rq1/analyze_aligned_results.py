"""Analyze aligned dataset for RQ1 results.

This script computes all metrics on the aligned dataset where all methods
are evaluated on the exact same scenarios for fair comparison.
"""

import pandas as pd
from pathlib import Path

# Load aligned data
data_path = Path(__file__).parent.parent.parent / "data" / "2026_01_results_aligned.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("RQ1 ANALYSIS ON ALIGNED DATASET")
print("=" * 80)

# Count scenarios
n_rows = len(df)
eval_methods = df["eval_method"].unique()
print(f"\nTotal rows: {n_rows:,}")
print(f"Eval methods: {sorted(eval_methods)}")

# Compute scenario count
base_a_df = df[df["eval_method"] == "base_a"]
n_scenarios = len(base_a_df)
print(f"\nNumber of aligned scenarios: {n_scenarios:,}")

# ============================================================================
# 1. BASELINE PERFORMANCE (on aligned scenarios)
# ============================================================================
print("\n" + "=" * 80)
print("1. BASELINE PERFORMANCE")
print("=" * 80)

for baseline in ["base_a", "base_b"]:
    b_df = df[df["eval_method"] == baseline]
    em_rate = b_df["exact_match"].sum() / len(b_df) * 100
    sim_mean = b_df["similarity"].mean()
    bleu_mean = b_df["bleu3"].mean()
    rouge_mean = b_df["rouge_l"].mean()
    
    print(f"\n{baseline} (n={len(b_df):,}):")
    print(f"  Exact Match: {em_rate:.1f}%")
    print(f"  Similarity:  {sim_mean:.3f}")
    print(f"  BLEU-3:      {bleu_mean:.3f}")
    print(f"  ROUGE-L:     {rouge_mean:.3f}")

# ============================================================================
# 2. MODEL PERFORMANCE (on aligned scenarios)
# ============================================================================
print("\n" + "=" * 80)
print("2. MODEL PERFORMANCE")
print("=" * 80)

models = [
    ("groq:qwen/qwen3-32b", "Qwen3-32B"),
    ("local:meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("openai/gpt-5-nano", "GPT-5-nano"),
]

# Store results for table
results = []

for model_id, model_name in models:
    print(f"\n{model_name}:")
    
    # Single-agent
    agent_df = df[(df["eval_method"] == "agent") & (df["model_name"] == model_id)]
    agent_em = agent_df["exact_match"].sum() / len(agent_df) * 100 if len(agent_df) > 0 else 0
    agent_sim = agent_df["similarity"].mean() if len(agent_df) > 0 else 0
    agent_bleu = agent_df["bleu3"].mean() if len(agent_df) > 0 else 0
    agent_rouge = agent_df["rouge_l"].mean() if len(agent_df) > 0 else 0
    
    # Multi-agent (bypass7)
    bypass_df = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_id)]
    bypass_em = bypass_df["exact_match"].sum() / len(bypass_df) * 100 if len(bypass_df) > 0 else 0
    bypass_sim = bypass_df["similarity"].mean() if len(bypass_df) > 0 else 0
    bypass_bleu = bypass_df["bleu3"].mean() if len(bypass_df) > 0 else 0
    bypass_rouge = bypass_df["rouge_l"].mean() if len(bypass_df) > 0 else 0
    
    print(f"  Single-Agent (n={len(agent_df):,}):")
    print(f"    EM: {agent_em:.1f}%, Sim: {agent_sim:.3f}, BLEU-3: {agent_bleu:.3f}, ROUGE-L: {agent_rouge:.3f}")
    print(f"  Multi-Agent (n={len(bypass_df):,}):")
    print(f"    EM: {bypass_em:.1f}%, Sim: {bypass_sim:.3f}, BLEU-3: {bypass_bleu:.3f}, ROUGE-L: {bypass_rouge:.3f}")
    print(f"  Δ EM: {bypass_em - agent_em:+.1f}pp, Δ Sim: {bypass_sim - agent_sim:+.3f}")
    
    results.append({
        "model": model_name,
        "n": len(agent_df),
        "single_em": agent_em,
        "multi_em": bypass_em,
        "delta_em": bypass_em - agent_em,
        "single_sim": agent_sim,
        "multi_sim": bypass_sim,
        "delta_sim": bypass_sim - agent_sim,
        "single_bleu": agent_bleu,
        "multi_bleu": bypass_bleu,
        "single_rouge": agent_rouge,
        "multi_rouge": bypass_rouge,
    })

# ============================================================================
# 3. BYPASS METHOD DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("3. BYPASS METHOD DISTRIBUTION")
print("=" * 80)

for model_id, model_name in models:
    bypass_df = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_id)]
    total = len(bypass_df)
    
    a_count = (bypass_df["bypass_method"] == "A").sum()
    b_count = (bypass_df["bypass_method"] == "B").sum()
    mix_count = (bypass_df["bypass_method"] == "MIX").sum()
    
    print(f"\n{model_name} (n={total:,}):")
    print(f"  A: {a_count:,} ({a_count/total*100:.2f}%)")
    print(f"  B: {b_count:,} ({b_count/total*100:.2f}%)")
    print(f"  MIX: {mix_count:,} ({mix_count/total*100:.2f}%)")

# ============================================================================
# 4. WIN/TIE/LOSS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. WIN/TIE/LOSS ANALYSIS (Multi-agent vs Single-agent)")
print("=" * 80)

for metric in ["exact_match", "similarity"]:
    print(f"\n--- {metric.upper()} ---")
    
    for model_id, model_name in models:
        agent_df = df[(df["eval_method"] == "agent") & (df["model_name"] == model_id)].copy()
        bypass_df = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_id)].copy()
        
        # Create keys for matching
        agent_df["key"] = list(zip(agent_df["id"], agent_df["file_name"]))
        bypass_df["key"] = list(zip(bypass_df["id"], bypass_df["file_name"]))
        
        # Merge
        merged = agent_df[["key", metric]].merge(
            bypass_df[["key", metric]], 
            on="key", 
            suffixes=("_single", "_multi")
        )
        
        # Count wins/ties/losses
        wins = (merged[f"{metric}_multi"] > merged[f"{metric}_single"]).sum()
        ties = (merged[f"{metric}_multi"] == merged[f"{metric}_single"]).sum()
        losses = (merged[f"{metric}_multi"] < merged[f"{metric}_single"]).sum()
        total = len(merged)
        
        print(f"\n{model_name}:")
        print(f"  Wins: {wins:,} ({wins/total*100:.1f}%)")
        print(f"  Ties: {ties:,} ({ties/total*100:.1f}%)")
        print(f"  Losses: {losses:,} ({losses/total*100:.1f}%)")

# ============================================================================
# 5. INTELLIGENT SELECTION VALUE
# ============================================================================
print("\n" + "=" * 80)
print("5. INTELLIGENT SELECTION VALUE ANALYSIS")
print("=" * 80)

print("""
This measures: Multi-agent EM - Expected EM (if random selection with same A/B ratio)
Positive values indicate the model selects parents intelligently.
""")

# Get baseline EM by scenario
base_a_em = df[df["eval_method"] == "base_a"].set_index(["id", "file_name"])["exact_match"]
base_b_em = df[df["eval_method"] == "base_b"].set_index(["id", "file_name"])["exact_match"]

for model_id, model_name in models:
    bypass_df = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_id)].copy()
    
    # Actual EM
    actual_em = bypass_df["exact_match"].sum() / len(bypass_df) * 100
    
    # A/B counts
    a_count = (bypass_df["bypass_method"] == "A").sum()
    b_count = (bypass_df["bypass_method"] == "B").sum()
    total = len(bypass_df)
    
    # Base A/B EM rates
    base_a_rate = base_a_em.reindex(list(zip(bypass_df["id"], bypass_df["file_name"]))).mean() * 100
    base_b_rate = base_b_em.reindex(list(zip(bypass_df["id"], bypass_df["file_name"]))).mean() * 100
    
    # Expected EM if A/B selection matches random with observed ratio
    expected_em = (a_count * base_a_rate + b_count * base_b_rate) / total
    
    # Intelligent selection value
    isv = actual_em - expected_em
    
    print(f"\n{model_name}:")
    print(f"  Actual Multi-Agent EM: {actual_em:.2f}%")
    print(f"  Expected EM (weighted baseline): {expected_em:.2f}%")
    print(f"  Intelligent Selection Value: {isv:+.2f}pp")
    
    # Also compare to baseline ceiling
    base_a_overall = df[df["eval_method"] == "base_a"]["exact_match"].mean() * 100
    base_b_overall = df[df["eval_method"] == "base_b"]["exact_match"].mean() * 100
    
    print(f"  Base A EM (aligned): {base_a_overall:.2f}%")
    print(f"  Base B EM (aligned): {base_b_overall:.2f}%")
    print(f"  Δ vs Base A: {actual_em - base_a_overall:+.2f}pp")
    print(f"  Δ vs Base B: {actual_em - base_b_overall:+.2f}pp")

# ============================================================================
# 6. SUMMARY TABLE (Markdown format)
# ============================================================================
print("\n" + "=" * 80)
print("6. SUMMARY TABLE (for paper)")
print("=" * 80)

print("\n### Exact Match Results")
print("| Model | Single-Agent EM | Multi-Agent EM | Δ EM | Base A | Base B |")
print("|-------|-----------------|----------------|------|--------|--------|")

base_a_em_overall = df[df["eval_method"] == "base_a"]["exact_match"].mean() * 100
base_b_em_overall = df[df["eval_method"] == "base_b"]["exact_match"].mean() * 100

for r in results:
    print(f"| {r['model']} | {r['single_em']:.1f}% | {r['multi_em']:.1f}% | {r['delta_em']:+.1f}pp | {base_a_em_overall:.1f}% | {base_b_em_overall:.1f}% |")

print("\n### Similarity Results")
print("| Model | Single-Agent Sim | Multi-Agent Sim | Δ Sim |")
print("|-------|------------------|-----------------|-------|")
for r in results:
    print(f"| {r['model']} | {r['single_sim']:.3f} | {r['multi_sim']:.3f} | {r['delta_sim']:+.3f} |")

print("\n### Bypass Distribution")
print("| Model | Choose A | Choose B | MIX |")
print("|-------|----------|----------|-----|")

for model_id, model_name in models:
    bypass_df = df[(df["eval_method"] == "bypass7") & (df["model_name"] == model_id)]
    total = len(bypass_df)
    a_pct = (bypass_df["bypass_method"] == "A").sum() / total * 100
    b_pct = (bypass_df["bypass_method"] == "B").sum() / total * 100
    mix_pct = (bypass_df["bypass_method"] == "MIX").sum() / total * 100
    print(f"| {model_name} | {a_pct:.1f}% | {b_pct:.1f}% | {mix_pct:.2f}% |")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
