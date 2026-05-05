#!/usr/bin/env python3
"""
Advanced visualizations for day2 experiment results.
Generates professional, complex visualizations beyond simple bar charts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as mpath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants
RESULTS_DIR = Path("results/day2_compare")
OUTPUT_DIR = Path("results/advanced_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all experiment data."""
    data = {}
    
    # Load comparison tables
    data['method_table'] = pd.read_csv(RESULTS_DIR / "method_comparison_table.csv")
    data['category_table'] = pd.read_csv(RESULTS_DIR / "by_category_summary.csv")
    data['prompt_table'] = pd.read_csv(RESULTS_DIR / "by_prompt_summary.csv")
    
    # Load individual metrics for distribution analysis
    metrics_files = list(Path("results/lora_smoke_eval").glob("*/metrics.csv"))
    data['all_metrics'] = []
    for f in metrics_files:
        df = pd.read_csv(f)
        df['method'] = f.parent.name
        data['all_metrics'].append(df)
    
    data['metrics_df'] = pd.concat(data['all_metrics'], ignore_index=True)
    
    return data

def create_radar_chart(data):
    """Create radar chart comparing methods across categories."""
    df = data['method_table']
    
    # Select methods to compare (excluding base and baseline)
    methods_to_plot = ['v_prediction_ckpt200', 'v_prediction_ckpt500', 
                      'noise_offset_002_ckpt200', 'noise_offset_002_ckpt500',
                      'noise_offset_005_ckpt200', 'noise_offset_005_ckpt500']
    
    # Categories for radar chart
    categories = ['dark_mean', 'bright_mean', 'white_bg_mean', 
                 'black_bg_mean', 'backlight_mean', 'neutral_mean']
    
    # Prepare data
    plot_data = []
    for method in methods_to_plot:
        method_data = df[df['method'] == method].iloc[0]
        values = [method_data[cat] for cat in categories]
        plot_data.append({
            'method': method,
            'values': values,
            'overall': method_data['overall_mean']
        })
    
    # Create radar chart
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Custom colors based on method type
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
    
    for idx, method_data in enumerate(plot_data):
        values = method_data['values']
        values += values[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=method_data['method'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    category_names = ['Dark', 'Bright', 'White BG', 'Black BG', 'Backlight', 'Neutral']
    ax.set_xticklabels(category_names, fontsize=12)
    
    # Set radial labels
    ax.set_rlabel_position(30)
    ax.set_ylim(0.15, 0.65)
    
    # Add title and legend
    plt.title('Multi-dimensional Method Comparison\nPerformance Across Lighting Categories', 
              size=16, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'radar_chart_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved radar chart to {OUTPUT_DIR / 'radar_chart_methods.png'}")

def create_heatmap_grid(data):
    """Create a grid of heatmaps showing method performance."""
    category_df = data['category_table']
    
    # Extract delta values for heatmap
    delta_cols = [col for col in category_df.columns if 'delta' in col]
    methods = [col.replace('_delta_vs_base', '') for col in delta_cols]
    
    # Calculate grid dimensions based on number of methods
    n_methods = len(methods)
    n_cols = min(4, int(np.ceil(np.sqrt(n_methods))))
    n_rows = int(np.ceil(n_methods / n_cols))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    categories = category_df['category'].values
    vmin, vmax = category_df[delta_cols].min().min(), category_df[delta_cols].max().max()
    
    # Create custom colormap (blue for negative, red for positive)
    cmap = LinearSegmentedColormap.from_list('delta_cmap', ['#2166ac', '#f7f7f7', '#b2182b'])
    
    for idx, (col, method) in enumerate(zip(delta_cols, methods)):
        ax = fig.add_subplot(gs[idx])
        values = category_df[[col]].values
        im = ax.imshow(values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Add text annotations
        for i in range(len(categories)):
            val = values[i, 0]
            color = 'white' if abs(val) > (vmax - vmin) * 0.3 else 'black'
            ax.text(0, i, f'{val:.3f}', ha='center', va='center', 
                    color=color, fontweight='bold')
        
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=10)
        ax.set_xticks([])
        ax.set_title(f'{method}\nBrightness Δ vs Base', fontsize=11, fontweight='bold')
        ax.grid(False)
    
    # Add colorbar (position adjusted based on grid width)
    cbar_left = 0.88 + 0.02 * n_cols
    cbar_ax = fig.add_axes([cbar_left, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Brightness Change (Δ)', fontsize=12)
    
    plt.suptitle('Method Performance Heatmap Grid\nBrightness Change by Category', 
                 fontsize=18, fontweight='bold', y=0.95)
    plt.savefig(OUTPUT_DIR / 'heatmap_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap grid to {OUTPUT_DIR / 'heatmap_grid.png'}")

def create_training_progress_analysis(data):
    """Analyze training progress across checkpoints."""
    df = data['method_table']
    
    # Extract checkpoint information
    df['method_type'] = df['method'].apply(lambda x: 'v_prediction' if 'v_prediction' in x else 
                                                    'offset002' if 'offset002' in x else 
                                                    'offset005' if 'offset005' in x else 
                                                    'baseline')
    
    def extract_checkpoint(method_name):
        """Extract checkpoint number from method name."""
        if 'ckpt200' in method_name:
            return 200
        elif 'ckpt500' in method_name:
            return 500
        elif 'ckpt800' in method_name:
            return 800
        elif 'final' in method_name:
            return 500
        elif 'base' in method_name:
            return 0
        else:
            # Try to extract any number from the string
            import re
            numbers = re.findall(r'\d+', method_name)
            if numbers:
                return int(numbers[-1])
            return 0
    
    df['checkpoint'] = df['method'].apply(extract_checkpoint)
    
    # Create progress analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot 1: Overall brightness progression
    ax = axes[0]
    for method_type in ['v_prediction', 'offset002', 'offset005']:
        subset = df[df['method_type'] == method_type].sort_values('checkpoint')
        ax.plot(subset['checkpoint'], subset['overall_mean'], 'o-', 
                linewidth=2, markersize=8, label=method_type)
    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Overall Brightness', fontsize=12)
    ax.set_title('Brightness Evolution by Method', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dark scene performance progression
    ax = axes[1]
    for method_type in ['v_prediction', 'offset002', 'offset005']:
        subset = df[df['method_type'] == method_type].sort_values('checkpoint')
        ax.plot(subset['checkpoint'], subset['dark_mean'], 'o-', 
                linewidth=2, markersize=8, label=method_type)
    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Dark Scene Brightness', fontsize=12)
    ax.set_title('Dark Scene Performance Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bright scene performance progression
    ax = axes[2]
    for method_type in ['v_prediction', 'offset002', 'offset005']:
        subset = df[df['method_type'] == method_type].sort_values('checkpoint')
        ax.plot(subset['checkpoint'], subset['bright_mean'], 'o-', 
                linewidth=2, markersize=8, label=method_type)
    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Bright Scene Brightness', fontsize=12)
    ax.set_title('Bright Scene Performance Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance range across categories
    ax = axes[3]
    category_cols = ['dark_mean', 'bright_mean', 'white_bg_mean', 
                    'black_bg_mean', 'backlight_mean', 'neutral_mean']
    
    for idx, method_type in enumerate(['v_prediction', 'offset002', 'offset005']):
        subset = df[df['method_type'] == method_type].sort_values('checkpoint')
        for i, checkpoint in enumerate(subset['checkpoint']):
            row = subset[subset['checkpoint'] == checkpoint].iloc[0]
            values = [row[col] for col in category_cols]
            positions = np.arange(len(values)) + idx * 0.2 + i * 0.05
            ax.scatter(positions, values, s=50, alpha=0.7, 
                      label=f'{method_type} ckpt{checkpoint}' if i == 0 else '')
    
    ax.set_xticks(np.arange(len(category_cols)))
    ax.set_xticklabels(['Dark', 'Bright', 'WhiteBG', 'BlackBG', 'Backlight', 'Neutral'], 
                      rotation=45, fontsize=10)
    ax.set_ylabel('Brightness', fontsize=12)
    ax.set_title('Category-wise Performance Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Delta vs base heatmap-like visualization
    ax = axes[4]
    methods_to_show = ['v_prediction_ckpt200', 'v_prediction_ckpt500',
                      'noise_offset_002_ckpt200', 'noise_offset_002_ckpt500',
                      'noise_offset_005_ckpt200', 'noise_offset_005_ckpt500']
    
    deltas = []
    for method in methods_to_show:
        row = df[df['method'] == method]
        if len(row) > 0:
            delta = row['overall_mean'].values[0] - df[df['method'] == 'base']['overall_mean'].values[0]
            deltas.append(delta)
    
    colors = ['red' if d > 0 else 'blue' for d in deltas]
    bars = ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.7)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([m.split('_')[0] + '_' + m.split('_')[-1] for m in methods_to_show], 
                      rotation=45, fontsize=10)
    ax.set_ylabel('Δ Brightness vs Base', fontsize=12)
    ax.set_title('Overall Brightness Change', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                f'{delta:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    # Plot 6: Method effectiveness ranking
    ax = axes[5]
    effectiveness = []
    for method in methods_to_show:
        row = df[df['method'] == method]
        if len(row) > 0:
            # Calculate effectiveness score (lower is better for dark scenes, higher for bright)
            dark_score = abs(row['dark_mean'].values[0] - 0.2)  # Target dark
            bright_score = abs(row['bright_mean'].values[0] - 0.6)  # Target bright
            effectiveness.append(1.0 / (dark_score + bright_score + 1e-6))
    
    sorted_idx = np.argsort(effectiveness)[::-1]
    sorted_methods = [methods_to_show[i] for i in sorted_idx]
    sorted_scores = [effectiveness[i] for i in sorted_idx]
    
    bars = ax.bar(range(len(sorted_scores)), sorted_scores, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(sorted_scores))))
    ax.set_xticks(range(len(sorted_scores)))
    ax.set_xticklabels([m.split('_')[0] + '_' + m.split('_')[-1] for m in sorted_methods], 
                      rotation=45, fontsize=10)
    ax.set_ylabel('Effectiveness Score', fontsize=12)
    ax.set_title('Method Effectiveness Ranking', fontsize=14, fontweight='bold')
    
    plt.suptitle('Comprehensive Training Progress Analysis', 
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_progress_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training progress analysis to {OUTPUT_DIR / 'training_progress_analysis.png'}")

def create_interactive_style_parallel_coordinates(data):
    """Create parallel coordinates plot for multi-dimensional comparison."""
    df = data['method_table']
    
    # Select methods and features for parallel coordinates
    methods_to_plot = ['base', 'baseline_final', 
                      'v_prediction_ckpt200', 'v_prediction_ckpt500',
                      'noise_offset_002_ckpt200', 'noise_offset_002_ckpt500',
                      'noise_offset_005_ckpt200', 'noise_offset_005_ckpt500']
    
    features = ['overall_mean', 'dark_mean', 'bright_mean', 
               'white_bg_mean', 'black_bg_mean', 'backlight_mean', 'neutral_mean']
    
    # Prepare data
    plot_df = df[df['method'].isin(methods_to_plot)].copy()
    
    # Normalize features for parallel coordinates
    normalized_data = []
    for feature in features:
        min_val = plot_df[feature].min()
        max_val = plot_df[feature].max()
        plot_df[f'{feature}_norm'] = (plot_df[feature] - min_val) / (max_val - min_val)
    
    # Create parallel coordinates plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Colors based on method type
    method_colors = {}
    color_palette = plt.cm.tab20(np.linspace(0, 1, len(methods_to_plot)))
    for i, method in enumerate(methods_to_plot):
        method_colors[method] = color_palette[i]
    
    # Plot each method
    for idx, row in plot_df.iterrows():
        method = row['method']
        values = [row[f'{feature}_norm'] for feature in features]
        ax.plot(range(len(features)), values, 'o-', 
                linewidth=2, markersize=8, 
                color=method_colors[method], 
                label=method, alpha=0.8)
    
    # Customize plot
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(['Overall', 'Dark', 'Bright', 'WhiteBG', 'BlackBG', 'Backlight', 'Neutral'], 
                      fontsize=12, rotation=30)
    ax.set_ylabel('Normalized Brightness', fontsize=14)
    ax.set_title('Parallel Coordinates: Multi-dimensional Method Comparison', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add secondary y-axis with actual values
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    # Map normalized values back to original range for select features
    overall_min = plot_df['overall_mean'].min()
    overall_max = plot_df['overall_mean'].max()
    ax2.set_ylabel(f'Actual Brightness Range: [{overall_min:.3f}, {overall_max:.3f}]', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parallel coordinates plot to {OUTPUT_DIR / 'parallel_coordinates.png'}")

def create_comprehensive_report(data):
    """Create a comprehensive PDF-style report with all visualizations."""
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = OUTPUT_DIR / 'comprehensive_analysis_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.7, 'Day 2 Experiment Analysis Report', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.6, 'Advanced Visualizations and Insights', 
                ha='center', va='center', fontsize=18, style='italic')
        ax.text(0.5, 0.4, 'Generated from brightness control experiments', 
                ha='center', va='center', fontsize=14)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Summary statistics page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Create summary table
        df = data['method_table']
        summary_text = "Experiment Summary Statistics\n\n"
        summary_text += f"Total methods evaluated: {len(df)}\n"
        summary_text += f"Methods analyzed:\n"
        for method in df['method']:
            summary_text += f"  • {method}\n"
        summary_text += f"\nKey Findings:\n"
        summary_text += f"  • Best dark scene performance: {df.loc[df['dark_mean'].idxmin(), 'method']}\n"
        summary_text += f"  • Best bright scene performance: {df.loc[df['bright_mean'].idxmax(), 'method']}\n"
        summary_text += f"  • Most balanced method: {df.loc[(df['overall_mean'] - 0.4).abs().idxmin(), 'method']}\n"
        
        ax.text(0.1, 0.9, summary_text, fontsize=12, fontfamily='monospace', 
                verticalalignment='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Add visualizations
        print("Generating report pages...")
        
        # Radar chart
        create_radar_chart(data)
        fig = plt.imread(OUTPUT_DIR / 'radar_chart_methods.png')
        fig_plt, ax_plt = plt.subplots(figsize=(11, 8.5))
        ax_plt.imshow(fig)
        ax_plt.axis('off')
        ax_plt.set_title('Radar Chart: Multi-dimensional Comparison', fontsize=16, fontweight='bold')
        pdf.savefig(fig_plt, bbox_inches='tight')
        plt.close()
        
        # Heatmap grid
        create_heatmap_grid(data)
        fig = plt.imread(OUTPUT_DIR / 'heatmap_grid.png')
        fig_plt, ax_plt = plt.subplots(figsize=(11, 8.5))
        ax_plt.imshow(fig)
        ax_plt.axis('off')
        ax_plt.set_title('Heatmap Grid: Category-wise Performance', fontsize=16, fontweight='bold')
        pdf.savefig(fig_plt, bbox_inches='tight')
        plt.close()
        
        # Training progress analysis
        create_training_progress_analysis(data)
        fig = plt.imread(OUTPUT_DIR / 'training_progress_analysis.png')
        fig_plt, ax_plt = plt.subplots(figsize=(11, 8.5))
        ax_plt.imshow(fig)
        ax_plt.axis('off')
        ax_plt.set_title('Training Progress Analysis', fontsize=16, fontweight='bold')
        pdf.savefig(fig_plt, bbox_inches='tight')
        plt.close()
        
        # Parallel coordinates
        create_interactive_style_parallel_coordinates(data)
        fig = plt.imread(OUTPUT_DIR / 'parallel_coordinates.png')
        fig_plt, ax_plt = plt.subplots(figsize=(11, 8.5))
        ax_plt.imshow(fig)
        ax_plt.axis('off')
        ax_plt.set_title('Parallel Coordinates: Multi-feature Analysis', fontsize=16, fontweight='bold')
        pdf.savefig(fig_plt, bbox_inches='tight')
        plt.close()
        
        # Conclusion page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        conclusion_text = "Conclusions and Recommendations\n\n"
        conclusion_text += "1. noise_offset=0.05 shows the most promising results for\n"
        conclusion_text += "   bidirectional brightness control.\n\n"
        conclusion_text += "2. v_prediction methods tend to increase overall brightness,\n"
        conclusion_text += "   which may not be desirable for dark scenes.\n\n"
        conclusion_text += "3. Checkpoint 500 generally provides more stable and\n"
        conclusion_text += "   pronounced effects than checkpoint 200.\n\n"
        conclusion_text += "4. The selected long-run training (noise_offset=0.05, 2500 steps)\n"
        conclusion_text += "   should further refine these effects.\n\n"
        conclusion_text += "Recommended next steps:\n"
        conclusion_text += "  • Evaluate long-run model on DataBench-100\n"
        conclusion_text += "  • Test interpolation between different offset values\n"
        conclusion_text += "  • Explore combination of v_prediction and noise_offset\n"
        
        ax.text(0.1, 0.9, conclusion_text, fontsize=14, fontweight='bold',
                verticalalignment='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"Saved comprehensive report to {pdf_path}")

def main():
    """Main function to generate all visualizations."""
    print("Loading experiment data...")
    data = load_data()
    
    print("Creating advanced visualizations...")
    
    # Create individual visualizations
    create_radar_chart(data)
    create_heatmap_grid(data)
    create_training_progress_analysis(data)
    create_interactive_style_parallel_coordinates(data)
    
    # Create comprehensive report
    create_comprehensive_report(data)
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    print("\nGenerated files:")
    for file in OUTPUT_DIR.glob("*"):
        print(f"  • {file.name}")

if __name__ == "__main__":
    main()