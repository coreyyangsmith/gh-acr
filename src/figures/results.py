from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def analyze_results():
    """Analyze and visualize results from the agent evaluation CSV."""
    
    # Load the data
    data_path = Path("data/2025_07_26_agent_results.csv")
    df = pd.read_csv(data_path)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Agent Performance Analysis by Difficulty Level', fontsize=16, fontweight='bold')
    
    # Define ordered difficulties
    difficulties = ['easy', 'medium', 'hard']
    df['difficulty'] = pd.Categorical(df['difficulty'], categories=difficulties, ordered=True)
    
    # Group by difficulty
    difficulty_groups = df.groupby('difficulty')
    
    # 1. Exact Match Rate by Difficulty
    em_rates = difficulty_groups['exact_match'].mean().reindex(difficulties)
    axes[0, 0].bar(em_rates.index, em_rates.values, alpha=0.7, color=['green', 'orange', 'red'])
    axes[0, 0].set_title('Exact Match Rate by Difficulty')
    axes[0, 0].set_ylabel('Exact Match Rate')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(em_rates.values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 2. Average Similarity Score by Difficulty
    sim_scores = difficulty_groups['similarity'].mean().reindex(difficulties)
    axes[0, 1].bar(sim_scores.index, sim_scores.values, alpha=0.7, color=['green', 'orange', 'red'])
    axes[0, 1].set_title('Average Similarity Score by Difficulty')
    axes[0, 1].set_ylabel('Similarity Score')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(sim_scores.values):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 3. Average Total Cost by Difficulty
    avg_costs = difficulty_groups['total_cost'].mean().reindex(difficulties)
    axes[1, 0].bar(avg_costs.index, avg_costs.values, alpha=0.7, color=['green', 'orange', 'red'])
    axes[1, 0].set_title('Average Total Cost by Difficulty')
    axes[1, 0].set_ylabel('Average Cost ($)')
    for i, v in enumerate(avg_costs.values):
        axes[1, 0].text(i, v + max(avg_costs.values) * 0.01, f'${v:.4f}', ha='center', fontweight='bold')
    
    # 4. Average Processing Time by Difficulty
    avg_times = difficulty_groups['processing_time_s'].mean().reindex(difficulties)
    axes[1, 1].bar(avg_times.index, avg_times.values, alpha=0.7, color=['green', 'orange', 'red'])
    axes[1, 1].set_title('Average Processing Time by Difficulty')
    axes[1, 1].set_ylabel('Processing Time (seconds)')
    for i, v in enumerate(avg_times.values):
        axes[1, 1].text(i, v + max(avg_times.values) * 0.01, f'{v:.1f}s', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Performance trends plot
    plt.figure(figsize=(12, 8))
    
    # Create cumulative performance tracking
    df_sorted = df.sort_index()
    difficulties = ['easy', 'medium', 'hard']
    colors = ['green', 'orange', 'red']
    
    for difficulty, color in zip(difficulties, colors):
        diff_data = df_sorted[df_sorted['difficulty'] == difficulty]
        if len(diff_data) > 0:
            # Calculate cumulative exact match rate
            cumulative_em = diff_data['exact_match'].expanding().mean()
            plt.plot(range(len(cumulative_em)), cumulative_em, 
                    label=f'{difficulty.capitalize()} (n={len(diff_data)})', 
                    color=color, linewidth=2, marker='o', markersize=4)
    
    plt.title('Cumulative Exact Match Performance by Difficulty', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Scenarios Processed')
    plt.ylabel('Cumulative Exact Match Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive analysis tables
    print("="*80)
    print("COMPREHENSIVE AGENT PERFORMANCE ANALYSIS")
    print("="*80)
    print()
    
    # Overall summary statistics
    print("OVERALL SUMMARY STATISTICS")
    print("-" * 40)
    overall_stats = {
        'Total Scenarios': len(df),
        'Overall Exact Match Rate': f"{df['exact_match'].mean():.3f}",
        'Overall Similarity Score': f"{df['similarity'].mean():.3f}",
        'Total Cost': f"${df['total_cost'].sum():.4f}",
        'Average Cost per Scenario': f"${df['total_cost'].mean():.4f}",
        'Total Processing Time': f"{df['processing_time_s'].sum():.1f} seconds",
        'Average Processing Time': f"{df['processing_time_s'].mean():.1f} seconds"
    }
    
    for key, value in overall_stats.items():
        print(f"{key:<30}: {value}")
    print()
    
    # Difficulty-based performance comparison
    print("PERFORMANCE BY DIFFICULTY LEVEL")
    print("-" * 50)
    
    performance_table = pd.DataFrame()
    for difficulty in difficulties:
        subset = df[df['difficulty'] == difficulty]
        if len(subset) > 0:
            performance_table[difficulty.capitalize()] = [
                len(subset),
                f"{subset['exact_match'].mean():.3f}",
                f"{subset['similarity'].mean():.3f}",
                f"${subset['total_cost'].mean():.5f}",
                f"{subset['processing_time_s'].mean():.1f}s"
            ]
    
    performance_table.index = ['Count', 'Exact Match Rate', 'Avg Similarity', 'Avg Cost', 'Avg Time']
    print(performance_table.to_string())
    print()
    
    # Token usage analysis
    print("TOKEN USAGE ANALYSIS")
    print("-" * 30)
    
    token_cols = ['tokens_total_input', 'tokens_output', 'tokens_total']
    token_stats = df[token_cols + ['difficulty']].groupby('difficulty').agg({
        'tokens_total_input': ['mean', 'std', 'sum'],
        'tokens_output': ['mean', 'std', 'sum'],
        'tokens_total': ['mean', 'std', 'sum']
    }).round(0)
    
    print("Input Tokens by Difficulty:")
    input_tokens_df = pd.DataFrame({
        'Mean': token_stats[('tokens_total_input', 'mean')],
        'Std Dev': token_stats[('tokens_total_input', 'std')],
        'Total': token_stats[('tokens_total_input', 'sum')]
    })
    print(input_tokens_df.to_string())
    print()
    
    print("Output Tokens by Difficulty:")
    output_tokens_df = pd.DataFrame({
        'Mean': token_stats[('tokens_output', 'mean')],
        'Std Dev': token_stats[('tokens_output', 'std')],
        'Total': token_stats[('tokens_output', 'sum')]
    })
    print(output_tokens_df.to_string())
    print()
    
    print("Total Tokens by Difficulty:")
    total_tokens_df = pd.DataFrame({
        'Mean': token_stats[('tokens_total', 'mean')],
        'Std Dev': token_stats[('tokens_total', 'std')],
        'Total': token_stats[('tokens_total', 'sum')]
    })
    print(total_tokens_df.to_string())
    print()
    
    # Cost analysis
    print("DETAILED COST ANALYSIS")
    print("-" * 30)
    
    cost_stats = df.groupby('difficulty').agg({
        'cost_in': ['mean', 'std', 'sum'],
        'cost_out': ['mean', 'std', 'sum'],
        'total_cost': ['mean', 'std', 'sum']
    })
    
    cost_analysis_df = pd.DataFrame({
        'Input Cost (Mean)': cost_stats[('cost_in', 'mean')].map('${:.6f}'.format),
        'Output Cost (Mean)': cost_stats[('cost_out', 'mean')].map('${:.6f}'.format),
        'Total Cost (Mean)': cost_stats[('total_cost', 'mean')].map('${:.6f}'.format),
        'Total Cost (Sum)': cost_stats[('total_cost', 'sum')].map('${:.4f}'.format)
    })
    print(cost_analysis_df.to_string())
    print()
    
    # Performance vs Cost efficiency
    print("PERFORMANCE VS COST EFFICIENCY")
    print("-" * 40)
    
    efficiency_df = pd.DataFrame()
    for difficulty in difficulties:
        subset = df[df['difficulty'] == difficulty]
        if len(subset) > 0:
            em_rate = subset['exact_match'].mean()
            avg_cost = subset['total_cost'].mean()
            efficiency = em_rate / avg_cost if avg_cost > 0 else 0
            
            efficiency_df[difficulty.capitalize()] = [
                f"{em_rate:.3f}",
                f"${avg_cost:.6f}",
                f"{efficiency:.0f}"
            ]
    
    efficiency_df.index = ['Exact Match Rate', 'Average Cost', 'EM per Dollar (×1000)']
    print(efficiency_df.to_string())
    print()
    
    # Correlation analysis
    print("CORRELATION ANALYSIS")
    print("-" * 25)
    
    numeric_cols = ['exact_match', 'similarity', 'total_cost', 'processing_time_s', 
                   'tokens_total', 'tokens_total_input', 'tokens_output']
    correlation_matrix = df[numeric_cols].corr()
    
    print("Key Correlations:")
    print(f"Exact Match vs Similarity: {correlation_matrix.loc['exact_match', 'similarity']:.3f}")
    print(f"Total Cost vs Processing Time: {correlation_matrix.loc['total_cost', 'processing_time_s']:.3f}")
    print(f"Total Tokens vs Total Cost: {correlation_matrix.loc['tokens_total', 'total_cost']:.3f}")
    print(f"Exact Match vs Total Cost: {correlation_matrix.loc['exact_match', 'total_cost']:.3f}")
    print()
    
    # Success rate by repository
    print("TOP PERFORMING REPOSITORIES (by Exact Match Rate)")
    print("-" * 55)
    
    repo_performance = df.groupby('repo').agg({
        'exact_match': ['count', 'mean', 'sum'],
        'similarity': 'mean',
        'total_cost': 'mean'
    }).round(3)
    
    repo_performance.columns = ['Count', 'EM_Rate', 'EM_Count', 'Avg_Similarity', 'Avg_Cost']
    repo_performance = repo_performance[repo_performance['Count'] >= 1]  # At least 1 scenario
    repo_performance = repo_performance.sort_values('EM_Rate', ascending=False)
    
    print(repo_performance.head(10).to_string())
    print()
    
    # Failure analysis
    print("FAILURE ANALYSIS")
    print("-" * 20)
    
    failed_scenarios = df[df['exact_match'] == False]
    print(f"Total Failed Scenarios: {len(failed_scenarios)}")
    print(f"Failure Rate by Difficulty:")
    
    failure_rates = df.groupby('difficulty').apply(
        lambda x: 1 - x['exact_match'].mean()
    ).round(3)
    
    for difficulty, rate in failure_rates.items():
        count = len(df[df['difficulty'] == difficulty])
        failed_count = len(failed_scenarios[failed_scenarios['difficulty'] == difficulty])
        print(f"  {difficulty.capitalize()}: {rate:.3f} ({failed_count}/{count})")
    
    print()
    print("="*80)

if __name__ == "__main__":
    analyze_results()
