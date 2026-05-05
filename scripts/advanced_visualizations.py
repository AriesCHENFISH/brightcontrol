#!/usr/bin/env python3
"""
Generate advanced visualizations for brightness control experiment results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
RESULTS_DIR = Path("/data1/cx/brightcontrol/results/day2_compare")
OUTPUT_DIR = RESULTS_DIR / "advanced_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all relevant data files."""
    prompt_df = pd.read_csv(RESULTS_DIR / "by_prompt_summary.csv")
    category_df = pd.read_csv(RESULTS_DIR / "by_category_summary.csv")
    method_df = pd.read_csv(RESULTS_DIR / "method_comparison_table.csv")
    return prompt_df, category_df, method_df

def create_line_chart(prompt_df):
    """Line chart: delta vs prompt for each method."""
    plt.figure(figsize=(14, 8))
    
    # Methods to plot (delta columns)
    delta_cols = [col for col in prompt_df.columns if 'delta' in col]
    # Exclude baseline delta
    delta_cols = [col for col in delta_cols if 'baseline' not in col]
    
    # Prepare data: melt to long format
    melt_df = prompt_df.melt(id_vars=['prompt_id', 'prompt'], 
                             value_vars=delta_cols,
                             var_name='Method', value_name='Delta')
    # Clean method names
    melt_df['Method'] = melt_df['Method'].str.replace('_delta_vs_base', '')
    
    # Plot each method
    for method in melt_df['Method'].unique():
        method_data = melt_df[melt_df['Method'] == method].sort_values('prompt_id')
        plt.plot(method_data['prompt_id'], method_data['Delta'], 
                 marker='o', linewidth=2, label=method)
    
    plt.xlabel('Prompt ID', fontsize=12)
    plt.ylabel('Brightness Delta (vs Base)', fontsize=12)
    plt.title('Brightness Adjustment by Method Across Prompts', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "line_chart_delta_by_prompt.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap(prompt_df):
    """Heatmap: method vs prompt delta matrix."""
    # Extract delta columns and pivot
    delta_cols = [col for col in prompt_df.columns if 'delta' in col and 'baseline' not in col]
    heatmap_data = prompt_df[['prompt_id'] + delta_cols].set_index('prompt_id')
    heatmap_data.columns = [col.replace('_delta_vs_base', '') for col in heatmap_data.columns]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", cmap="RdBu_r", 
                center=0, cbar_kws={'label': 'Brightness Delta'})
    plt.xlabel('Prompt ID', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.title('Brightness Delta Heatmap: Method vs Prompt', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_method_vs_prompt.png", dpi=300)
    plt.close()

def create_radar_chart(prompt_df):
    """Radar chart: multi-dimensional comparison of methods."""
    from math import pi
    
    # Aggregate by method across prompts (mean absolute delta)
    delta_cols = [col for col in prompt_df.columns if 'delta' in col and 'baseline' not in col]
    method_names = [col.replace('_delta_vs_base', '') for col in delta_cols]
    
    # Calculate mean absolute delta for each method per prompt category
    # Use absolute values to show magnitude of effect
    abs_means = [prompt_df[col].abs().mean() for col in delta_cols]
    
    # Number of variables
    N = len(method_names)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialise the radar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], method_names, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4], ["0.1", "0.2", "0.3", "0.4"], color="grey", size=10)
    plt.ylim(0, 0.5)
    
    # Plot data
    values = abs_means
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', marker='o', label='Mean Abs Delta')
    ax.fill(angles, values, alpha=0.25)
    
    plt.title('Radar Chart: Mean Absolute Brightness Adjustment by Method', 
              size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radar_chart_methods.png", dpi=300)
    plt.close()

def create_scatter_plot(prompt_df):
    """Scatter plot: base brightness vs delta for each method."""
    plt.figure(figsize=(12, 8))
    
    # Methods to plot
    delta_cols = [col for col in prompt_df.columns if 'delta' in col and 'baseline' not in col]
    method_names = [col.replace('_delta_vs_base', '') for col in delta_cols]
    
    # Create subplots
    n_methods = len(method_names)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (method, delta_col) in enumerate(zip(method_names, delta_cols)):
        ax = axes[idx]
        scatter = ax.scatter(prompt_df['base_mean_lum'], prompt_df[delta_col], 
                            c=prompt_df['prompt_id'], cmap='viridis', s=100, alpha=0.7)
        
        # Add horizontal and vertical lines at zero
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Base Brightness', fontsize=10)
        ax.set_ylabel('Delta', fontsize=10)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for prompt_id
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prompt ID')
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Brightness Delta vs Base Brightness by Method', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_base_vs_delta.png", dpi=300)
    plt.close()

def create_violin_plot(prompt_df):
    """Violin plot: distribution of deltas for each method."""
    plt.figure(figsize=(14, 8))
    
    # Prepare data in long format
    delta_cols = [col for col in prompt_df.columns if 'delta' in col]
    melt_df = prompt_df.melt(value_vars=delta_cols,
                             var_name='Method', value_name='Delta')
    # Clean method names
    melt_df['Method'] = melt_df['Method'].str.replace('_delta_vs_base', '')
    
    # Create violin plot
    sns.violinplot(x='Method', y='Delta', data=melt_df, inner='quartile', palette='muted')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Delta')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Brightness Delta', fontsize=12)
    plt.title('Distribution of Brightness Deltas by Method', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "violin_plot_delta_distribution.png", dpi=300)
    plt.close()

def create_parallel_coordinates(prompt_df):
    """Parallel coordinates plot for multi-dimensional comparison."""
    from pandas.plotting import parallel_coordinates
    
    # Select a subset of methods for clarity
    delta_cols = [col for col in prompt_df.columns if 'delta' in col and 'baseline' not in col]
    # Take first 4 methods
    delta_cols = delta_cols[:4]
    
    # Prepare data
    plot_df = prompt_df[['prompt_id'] + delta_cols].copy()
    plot_df.columns = ['prompt_id'] + [col.replace('_delta_vs_base', '') for col in delta_cols]
    
    plt.figure(figsize=(14, 8))
    parallel_coordinates(plot_df, 'prompt_id', colormap='viridis', alpha=0.7)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Brightness Delta', fontsize=12)
    plt.title('Parallel Coordinates: Brightness Deltas Across Methods', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parallel_coordinates.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading data...")
    prompt_df, category_df, method_df = load_data()
    print(f"Loaded {len(prompt_df)} prompts, {len(category_df)} categories, {len(method_df)} methods")
    
    print("Creating line chart...")
    create_line_chart(prompt_df)
    
    print("Creating heatmap...")
    create_heatmap(prompt_df)
    
    print("Creating radar chart...")
    create_radar_chart(prompt_df)
    
    print("Creating scatter plot...")
    create_scatter_plot(prompt_df)
    
    print("Creating violin plot...")
    create_violin_plot(prompt_df)
    
    print("Creating parallel coordinates plot...")
    create_parallel_coordinates(prompt_df)
    
    print(f"All visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()