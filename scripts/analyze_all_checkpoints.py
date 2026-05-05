#!/usr/bin/env python3
"""
Analyze all checkpoints (including long-run) and generate comprehensive visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/data1/cx/brightcontrol/results")
LONG_RUN_DIR = BASE_DIR / "day2_longrun_eval"
SHORT_RUN_DIR = BASE_DIR / "lora_smoke_eval"
OUTPUT_DIR = BASE_DIR / "day2_comprehensive_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Checkpoint groups
CHECKPOINT_GROUPS = {
    "v_prediction": [
        ("v_prediction_ckpt200", SHORT_RUN_DIR / "v_prediction_ckpt200"),
        ("v_prediction_ckpt500", SHORT_RUN_DIR / "v_prediction_ckpt500"),
    ],
    "noise_offset_002": [
        ("noise_offset_002_ckpt200", SHORT_RUN_DIR / "noise_offset_002_ckpt200"),
        ("noise_offset_002_ckpt500", SHORT_RUN_DIR / "noise_offset_002_ckpt500"),
    ],
    "noise_offset_005_short": [
        ("noise_offset_005_ckpt200", SHORT_RUN_DIR / "noise_offset_005_ckpt200"),
        ("noise_offset_005_ckpt500", SHORT_RUN_DIR / "noise_offset_005_ckpt500"),
    ],
    "noise_offset_005_longrun": [
        ("noise_offset_005_longrun_ckpt500", LONG_RUN_DIR / "noise_offset_005_longrun_ckpt500"),
        ("noise_offset_005_longrun_ckpt1000", LONG_RUN_DIR / "noise_offset_005_longrun_ckpt1000"),
        ("noise_offset_005_longrun_ckpt1500", LONG_RUN_DIR / "noise_offset_005_longrun_ckpt1500"),
        ("noise_offset_005_longrun_ckpt2000", LONG_RUN_DIR / "noise_offset_005_longrun_ckpt2000"),
        ("noise_offset_005_longrun_ckpt2500", LONG_RUN_DIR / "noise_offset_005_longrun_ckpt2500"),
    ]
}

# Base model
BASE_CHECKPOINT = SHORT_RUN_DIR / "base"

def load_metrics(checkpoint_path):
    """Load metrics.csv from a checkpoint directory."""
    metrics_file = checkpoint_path / "metrics.csv"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} does not exist")
        return None
    df = pd.read_csv(metrics_file)
    return df

def compute_prompt_averages(df):
    """Compute average luminance per prompt (average over seeds)."""
    # Group by prompt_id and compute mean
    avg_df = df.groupby('prompt_id').agg({
        'mean_luminance': 'mean',
        'prompt': 'first'
    }).reset_index()
    return avg_df

def load_all_checkpoints():
    """Load all checkpoints and compute deltas relative to base."""
    # Load base model
    base_df = load_metrics(BASE_CHECKPOINT)
    if base_df is None:
        raise FileNotFoundError(f"Base metrics not found at {BASE_CHECKPOINT}")
    base_avg = compute_prompt_averages(base_df)
    base_lum = dict(zip(base_avg['prompt_id'], base_avg['mean_luminance']))
    
    # Load all checkpoints
    all_data = []
    for group_name, checkpoints in CHECKPOINT_GROUPS.items():
        for ckpt_name, ckpt_path in checkpoints:
            df = load_metrics(ckpt_path)
            if df is None:
                continue
            avg_df = compute_prompt_averages(df)
            
            for _, row in avg_df.iterrows():
                prompt_id = row['prompt_id']
                prompt = row['prompt']
                lum = row['mean_luminance']
                base_lum_val = base_lum.get(prompt_id)
                if base_lum_val is None:
                    continue
                delta = lum - base_lum_val
                
                all_data.append({
                    'group': group_name,
                    'checkpoint': ckpt_name,
                    'prompt_id': prompt_id,
                    'prompt': prompt,
                    'luminance': lum,
                    'base_luminance': base_lum_val,
                    'delta': delta
                })
    
    return pd.DataFrame(all_data), base_lum

def create_training_steps_plot(df):
    """Line plot: delta vs training steps for noise_offset_005_longrun group."""
    longrun_df = df[df['group'] == 'noise_offset_005_longrun'].copy()
    if longrun_df.empty:
        print("No long-run data found")
        return
    
    # Extract step number from checkpoint name
    def extract_step(name):
        import re
        match = re.search(r'ckpt(\d+)', name)
        return int(match.group(1)) if match else 0
    
    longrun_df['step'] = longrun_df['checkpoint'].apply(extract_step)
    longrun_df = longrun_df.sort_values('step')
    
    # Pivot for plotting
    pivot_df = longrun_df.pivot_table(index='step', columns='prompt_id', values='delta', aggfunc='mean')
    
    plt.figure(figsize=(14, 8))
    for prompt_id in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[prompt_id], marker='o', linewidth=2, label=f'Prompt {prompt_id}')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Brightness Delta (vs Base)', fontsize=12)
    plt.title('Brightness Adjustment vs Training Steps (noise_offset=0.05 Long Run)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Prompt ID')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_steps_vs_delta.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a version with mean delta across all prompts
    mean_delta = longrun_df.groupby('step')['delta'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(mean_delta['step'], mean_delta['delta'], marker='s', linewidth=3, markersize=10, color='darkred')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Mean Brightness Delta (vs Base)', fontsize=12)
    plt.title('Mean Brightness Adjustment vs Training Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_delta_vs_training_steps.png", dpi=300)
    plt.close()

def create_heatmap_all_checkpoints(df):
    """Heatmap: checkpoint vs prompt delta."""
    # Create a matrix: checkpoints vs prompts
    pivot_df = df.pivot_table(index='checkpoint', columns='prompt_id', values='delta', aggfunc='mean')
    
    # Sort checkpoints by group and step
    def sort_key(name):
        # Extract group and step for sorting
        if 'longrun' in name:
            step = int(name.split('ckpt')[-1])
            return (3, step)  # longrun last
        elif '005' in name:
            step = int(name.split('ckpt')[-1])
            return (2, step)
        elif '002' in name:
            step = int(name.split('ckpt')[-1])
            return (1, step)
        else:  # v_prediction
            step = int(name.split('ckpt')[-1])
            return (0, step)
    
    pivot_df = pivot_df.loc[sorted(pivot_df.index, key=sort_key)]
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, 
                cbar_kws={'label': 'Brightness Delta'}, linewidths=0.5)
    plt.xlabel('Prompt ID', fontsize=12)
    plt.ylabel('Checkpoint', fontsize=12)
    plt.title('Brightness Delta Heatmap: All Checkpoints vs Prompts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_all_checkpoints.png", dpi=300)
    plt.close()

def create_violin_by_group(df):
    """Violin plot: distribution of deltas by checkpoint group."""
    plt.figure(figsize=(14, 8))
    
    # Order groups logically
    group_order = ['v_prediction', 'noise_offset_002', 'noise_offset_005_short', 'noise_offset_005_longrun']
    plot_df = df[df['group'].isin(group_order)].copy()
    
    sns.violinplot(x='group', y='delta', data=plot_df, inner='quartile', palette='muted', order=group_order)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Delta')
    plt.xlabel('Checkpoint Group', fontsize=12)
    plt.ylabel('Brightness Delta', fontsize=12)
    plt.title('Distribution of Brightness Deltas by Checkpoint Group', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "violin_by_group.png", dpi=300)
    plt.close()

def create_radar_by_prompt_category(df):
    """Radar chart: deltas by prompt category for each group."""
    from math import pi
    
    # Define prompt categories based on prompt_id
    # 0: dark, 1: bright, 2: white_bg, 3: black_bg, 4: backlit, 5: dark_scene, 6: balanced, 7: normal
    prompt_categories = {
        'dark': [0],
        'bright': [1],
        'white_bg': [2],
        'black_bg': [3],
        'backlit': [4],
        'dark_scene': [5],
        'balanced': [6],
        'normal': [7]
    }
    
    # Calculate mean delta for each group and category
    category_data = []
    for group in df['group'].unique():
        group_df = df[df['group'] == group]
        for cat_name, prompt_ids in prompt_categories.items():
            cat_delta = group_df[group_df['prompt_id'].isin(prompt_ids)]['delta'].mean()
            category_data.append({
                'group': group,
                'category': cat_name,
                'mean_delta': cat_delta
            })
    
    cat_df = pd.DataFrame(category_data)
    
    # Prepare data for radar chart
    groups = ['v_prediction', 'noise_offset_002', 'noise_offset_005_short', 'noise_offset_005_longrun']
    categories = list(prompt_categories.keys())
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialise the radar plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-0.2, -0.1, 0, 0.1, 0.2], ["-0.2", "-0.1", "0", "0.1", "0.2"], color="grey", size=10)
    plt.ylim(-0.25, 0.25)
    
    # Plot each group
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    for idx, group in enumerate(groups):
        group_data = cat_df[cat_df['group'] == group]
        values = group_data.sort_values('category')['mean_delta'].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', marker='o', 
                label=group, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    plt.title('Radar Chart: Mean Brightness Delta by Prompt Category and Group', 
              size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radar_by_category.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_surface_if_possible(df):
    """Attempt to create a 3D surface plot (if matplotlib 3D is available)."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Filter for longrun data
        longrun_df = df[df['group'] == 'noise_offset_005_longrun'].copy()
        if longrun_df.empty:
            return
        
        # Extract step number
        def extract_step(name):
            import re
            match = re.search(r'ckpt(\d+)', name)
            return int(match.group(1)) if match else 0
        
        longrun_df['step'] = longrun_df['checkpoint'].apply(extract_step)
        
        # Create meshgrid
        steps = sorted(longrun_df['step'].unique())
        prompts = sorted(longrun_df['prompt_id'].unique())
        
        X, Y = np.meshgrid(steps, prompts)
        Z = np.zeros_like(X, dtype=float)
        
        for i, step in enumerate(steps):
            for j, prompt in enumerate(prompts):
                delta = longrun_df[(longrun_df['step'] == step) & (longrun_df['prompt_id'] == prompt)]['delta']
                if not delta.empty:
                    Z[j, i] = delta.mean()
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Prompt ID', fontsize=10)
        ax.set_zlabel('Brightness Delta', fontsize=10)
        ax.set_title('3D Surface: Brightness Delta vs Training Step and Prompt', fontsize=12, fontweight='bold')
        
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Brightness Delta')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "3d_surface_delta.png", dpi=300)
        plt.close()
        
    except ImportError:
        print("3D plotting not available")
    except Exception as e:
        print(f"3D plot failed: {e}")

def create_summary_table(df):
    """Create summary CSV tables."""
    # Summary by checkpoint
    checkpoint_summary = df.groupby(['group', 'checkpoint']).agg({
        'delta': ['mean', 'std', 'min', 'max'],
        'luminance': 'mean',
        'base_luminance': 'first'
    }).round(4)
    checkpoint_summary.to_csv(OUTPUT_DIR / "checkpoint_summary.csv")
    
    # Summary by group
    group_summary = df.groupby('group').agg({
        'delta': ['mean', 'std', 'min', 'max'],
        'luminance': 'mean'
    }).round(4)
    group_summary.to_csv(OUTPUT_DIR / "group_summary.csv")
    
    # Summary by prompt
    prompt_summary = df.groupby(['prompt_id', 'prompt']).agg({
        'delta': ['mean', 'std', 'min', 'max'],
        'luminance': 'mean',
        'base_luminance': 'first'
    }).round(4)
    prompt_summary.to_csv(OUTPUT_DIR / "prompt_summary.csv")

def main():
    print("Loading all checkpoints...")
    df, base_lum = load_all_checkpoints()
    print(f"Loaded {len(df)} data points from {df['checkpoint'].nunique()} checkpoints")
    print(f"Groups: {df['group'].unique()}")
    
    print("\nCreating summary tables...")
    create_summary_table(df)
    
    print("Creating training steps plot...")
    create_training_steps_plot(df)
    
    print("Creating heatmap...")
    create_heatmap_all_checkpoints(df)
    
    print("Creating violin plot by group...")
    create_violin_by_group(df)
    
    print("Creating radar chart by category...")
    create_radar_by_prompt_category(df)
    
    print("Attempting 3D surface plot...")
    create_3d_surface_if_possible(df)
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()