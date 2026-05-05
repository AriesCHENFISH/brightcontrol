#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np

# Define method configurations
METHODS = [
    {"name": "base", "path": "results/lora_smoke_eval/base/metrics.csv"},
    {"name": "baseline_final", "path": "results/lora_smoke_eval/baseline_run1_final/metrics.csv"},
    {"name": "baseline_ckpt200", "path": "results/lora_smoke_eval/baseline_run1_ckpt200/metrics.csv"},
    {"name": "baseline_ckpt800", "path": "results/lora_smoke_eval/baseline_run1_ckpt800/metrics.csv"},
    {"name": "v_prediction_ckpt200", "path": "results/lora_smoke_eval/v_prediction_ckpt200/metrics.csv"},
    {"name": "v_prediction_ckpt500", "path": "results/lora_smoke_eval/v_prediction_ckpt500/metrics.csv"},
    {"name": "noise_offset_002_ckpt200", "path": "results/lora_smoke_eval/noise_offset_002_ckpt200/metrics.csv"},
    {"name": "noise_offset_002_ckpt500", "path": "results/lora_smoke_eval/noise_offset_002_ckpt500/metrics.csv"},
    {"name": "noise_offset_005_ckpt200", "path": "results/lora_smoke_eval/noise_offset_005_ckpt200/metrics.csv"},
    {"name": "noise_offset_005_ckpt500", "path": "results/lora_smoke_eval/noise_offset_005_ckpt500/metrics.csv"},
]

# Prompt categories
PROMPT_CATEGORY = {
    0: "dark",
    1: "bright",
    2: "white_bg",
    3: "black_bg",
    4: "backlight",
    5: "dark",
    6: "neutral",
    7: "neutral",
}

OUTPUT_DIR = Path("results/day2_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_method_data(method):
    """Load metrics.csv for a method, return None if file doesn't exist"""
    path = Path(method["path"])
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    
    df = pd.read_csv(path)
    df["category"] = df["prompt_id"].map(PROMPT_CATEGORY)
    return df

def main():
    rows = []
    
    for method in METHODS:
        df = load_method_data(method)
        if df is None:
            # Add placeholder row with NaN values
            rows.append({
                "method": method["name"],
                "overall_mean": np.nan,
                "dark_mean": np.nan,
                "bright_mean": np.nan,
                "white_bg_mean": np.nan,
                "black_bg_mean": np.nan,
                "backlight_mean": np.nan,
                "neutral_mean": np.nan,
                "num_prompts": np.nan,
                "num_images": np.nan,
            })
            continue
        
        # Calculate statistics
        overall_mean = df["mean_luminance"].mean()
        
        # Category means
        cat_means = df.groupby("category")["mean_luminance"].mean()
        
        row = {
            "method": method["name"],
            "overall_mean": overall_mean,
            "dark_mean": cat_means.get("dark", np.nan),
            "bright_mean": cat_means.get("bright", np.nan),
            "white_bg_mean": cat_means.get("white_bg", np.nan),
            "black_bg_mean": cat_means.get("black_bg", np.nan),
            "backlight_mean": cat_means.get("backlight", np.nan),
            "neutral_mean": cat_means.get("neutral", np.nan),
            "num_prompts": df["prompt_id"].nunique(),
            "num_images": len(df),
        }
        rows.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Sort methods for better readability
    order = [
        "base",
        "baseline_final",
        "baseline_ckpt200",
        "baseline_ckpt800",
        "v_prediction_ckpt200",
        "v_prediction_ckpt500",
        "noise_offset_002_ckpt200",
        "noise_offset_002_ckpt500",
        "noise_offset_005_ckpt200",
        "noise_offset_005_ckpt500",
    ]
    result_df["order"] = result_df["method"].apply(lambda x: order.index(x) if x in order else len(order))
    result_df = result_df.sort_values("order").drop(columns="order")
    
    # Save to CSV
    output_path = OUTPUT_DIR / "method_comparison_table.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    
    print("Method Comparison Table:")
    print(result_df.to_string())
    print(f"\nSaved to {output_path}")
    
    # Also create a markdown version (optional)
    try:
        md_path = OUTPUT_DIR / "method_comparison_table.md"
        result_df.to_markdown(md_path, index=False)
        print(f"Markdown version saved to {md_path}")
    except ImportError:
        print("Note: tabulate module not available, skipping markdown export")
    except Exception as e:
        print(f"Note: Could not save markdown: {e}")

if __name__ == "__main__":
    main()