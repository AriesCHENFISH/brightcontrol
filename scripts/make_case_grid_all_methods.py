#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import math

# Define methods to include (only those with existing metrics.csv for now)
METHODS = [
    {"name": "base", "path": "results/lora_smoke_eval/base"},
    {"name": "baseline_final", "path": "results/lora_smoke_eval/baseline_run1_final"},
    {"name": "baseline_ckpt200", "path": "results/lora_smoke_eval/baseline_run1_ckpt200"},
    {"name": "baseline_ckpt800", "path": "results/lora_smoke_eval/baseline_run1_ckpt800"},
    # The following will be included if directories exist
    {"name": "v_prediction_ckpt200", "path": "results/lora_smoke_eval/v_prediction_ckpt200"},
    {"name": "v_prediction_ckpt500", "path": "results/lora_smoke_eval/v_prediction_ckpt500"},
    {"name": "noise_offset_002_ckpt200", "path": "results/lora_smoke_eval/noise_offset_002_ckpt200"},
    {"name": "noise_offset_002_ckpt500", "path": "results/lora_smoke_eval/noise_offset_002_ckpt500"},
    {"name": "noise_offset_005_ckpt200", "path": "results/lora_smoke_eval/noise_offset_005_ckpt200"},
    {"name": "noise_offset_005_ckpt500", "path": "results/lora_smoke_eval/noise_offset_005_ckpt500"},
]

# Prompt IDs (0-7)
PROMPT_IDS = list(range(8))

# Grid configuration
THUMB_SIZE = (256, 256)
GRID_COLS = len(METHODS)  # One column per method
GRID_ROWS = len(PROMPT_IDS)  # One row per prompt

OUTPUT_DIR = Path("results/day2_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_image_path(method_dir, prompt_id, seed=0):
    """Get image path for a method, prompt, and seed"""
    img_dir = Path(method_dir) / "images"
    img_path = img_dir / f"p{prompt_id:02d}_s{seed}.png"
    return img_path if img_path.exists() else None

def main():
    # Filter methods that actually have directories
    valid_methods = []
    for method in METHODS:
        path = Path(method["path"])
        if path.exists():
            valid_methods.append(method)
        else:
            print(f"Warning: {method['path']} not found, skipping")
    
    if not valid_methods:
        print("No valid method directories found!")
        return
    
    print(f"Found {len(valid_methods)} valid methods:")
    for m in valid_methods:
        print(f"  - {m['name']}")
    
    # Create a grid for each prompt (horizontal layout: methods as columns)
    for prompt_id in PROMPT_IDS:
        # Load images for this prompt across all methods
        images = []
        method_labels = []
        
        for method in valid_methods:
            img_path = get_image_path(method["path"], prompt_id, seed=0)
            if img_path:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.fit(img, THUMB_SIZE)
                images.append(img)
                method_labels.append(method["name"])
            else:
                # Create a placeholder blank image
                placeholder = Image.new("RGB", THUMB_SIZE, color=(200, 200, 200))
                images.append(placeholder)
                method_labels.append(f"{method['name']} (missing)")
        
        # Create grid: one row, each method as a column
        grid_width = len(images) * THUMB_SIZE[0]
        grid_height = THUMB_SIZE[1]
        grid = Image.new("RGB", (grid_width, grid_height), "white")
        
        for i, img in enumerate(images):
            x = i * THUMB_SIZE[0]
            y = 0
            grid.paste(img, (x, y))
        
        # Save individual prompt grid
        prompt_grid_path = OUTPUT_DIR / f"grid_prompt{prompt_id:02d}.png"
        grid.save(prompt_grid_path)
        print(f"Saved grid for prompt {prompt_id} to {prompt_grid_path}")
    
    # Create a combined grid with all prompts and methods
    # Rows: prompts, Columns: methods
    combined_grid_width = len(valid_methods) * THUMB_SIZE[0]
    combined_grid_height = len(PROMPT_IDS) * THUMB_SIZE[1]
    combined_grid = Image.new("RGB", (combined_grid_width, combined_grid_height), "white")
    
    for prompt_idx, prompt_id in enumerate(PROMPT_IDS):
        for method_idx, method in enumerate(valid_methods):
            img_path = get_image_path(method["path"], prompt_id, seed=0)
            if img_path:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.fit(img, THUMB_SIZE)
            else:
                img = Image.new("RGB", THUMB_SIZE, color=(200, 200, 200))
            
            x = method_idx * THUMB_SIZE[0]
            y = prompt_idx * THUMB_SIZE[1]
            combined_grid.paste(img, (x, y))
    
    combined_path = OUTPUT_DIR / "grid_all_methods.png"
    combined_grid.save(combined_path)
    print(f"Saved combined grid to {combined_path}")

if __name__ == "__main__":
    main()