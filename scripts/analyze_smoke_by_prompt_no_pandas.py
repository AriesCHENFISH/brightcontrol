#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict

# Only include runs that exist
RUNS = [
    ("base", "results/lora_smoke_eval/base/metrics.csv"),
    ("baseline", "results/lora_smoke_eval/baseline_run1_final/metrics.csv"),
    # The following metrics.csv files may not exist yet
    # ("vpred", "results/lora_smoke_eval/v_prediction_ckpt200/metrics.csv"),
    # ("offset002", "results/lora_smoke_eval/noise_offset_002_ckpt200/metrics.csv"),
    # ("offset005", "results/lora_smoke_eval/noise_offset_005_ckpt200/metrics.csv"),
]

OUT = Path("results/day2_compare/by_prompt_summary.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_mean_by_prompt(path):
    """Returns dict: (prompt_id, prompt) -> mean_luminance"""
    path = Path(path)
    if not path.exists():
        return None
    
    # Collect luminances per prompt
    prompt_data = defaultdict(list)
    prompt_texts = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = int(row['prompt_id'])
            prompt = row['prompt']
            lum = float(row['mean_luminance'])
            prompt_data[prompt_id].append(lum)
            prompt_texts[prompt_id] = prompt
    
    # Compute means
    result = []
    for prompt_id, lums in prompt_data.items():
        mean_lum = sum(lums) / len(lums)
        result.append({
            'prompt_id': prompt_id,
            'prompt': prompt_texts[prompt_id],
            'mean_lum': mean_lum
        })
    
    # Sort by prompt_id
    result.sort(key=lambda x: x['prompt_id'])
    return result

def main():
    tables = {}
    for run_name, path in RUNS:
        data = load_mean_by_prompt(path)
        if data is None:
            print(f"Warning: {path} not found, skipping {run_name}")
            continue
        tables[run_name] = data
    
    if not tables:
        print("No data found!")
        return
    
    # Start with base data
    base_data = tables.get('base')
    if not base_data:
        print("Base data not found!")
        return
    
    # Build merged rows
    merged_rows = []
    for base_row in base_data:
        prompt_id = base_row['prompt_id']
        prompt = base_row['prompt']
        
        row = {
            'prompt_id': prompt_id,
            'prompt': prompt,
            'base_mean_lum': base_row['mean_lum']
        }
        
        # Add other runs
        for run_name, data in tables.items():
            if run_name == 'base':
                continue
            
            # Find matching prompt in this run
            match = None
            for d in data:
                if d['prompt_id'] == prompt_id:
                    match = d
                    break
            
            if match:
                row[f'{run_name}_mean_lum'] = match['mean_lum']
                row[f'{run_name}_delta_vs_base'] = match['mean_lum'] - base_row['mean_lum']
            else:
                row[f'{run_name}_mean_lum'] = ''
                row[f'{run_name}_delta_vs_base'] = ''
        
        merged_rows.append(row)
    
    # Write to CSV
    fieldnames = ['prompt_id', 'prompt', 'base_mean_lum']
    for run_name in tables.keys():
        if run_name != 'base':
            fieldnames.append(f'{run_name}_mean_lum')
            fieldnames.append(f'{run_name}_delta_vs_base')
    
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    
    print(f"Saved to {OUT}")
    
    # Print summary
    print("\nSummary:")
    for row in merged_rows:
        print(f"Prompt {row['prompt_id']}: base={row['base_mean_lum']:.4f}", end="")
        for run_name in tables.keys():
            if run_name != 'base':
                val = row.get(f'{run_name}_mean_lum', '')
                if val != '':
                    print(f", {run_name}={val:.4f}", end="")
        print()

if __name__ == "__main__":
    main()