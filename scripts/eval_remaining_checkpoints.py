#!/usr/bin/env python3
"""
Evaluate remaining checkpoints (2000 and 2500) for the long-run training.
"""
import subprocess
import os
import sys
from pathlib import Path

# Configuration
BASE_MODEL = "/data1/cx/sd15/"
PROMPT_FILE = "/data1/cx/brightcontrol/prompts/eval_smoke_8.txt"
OUTPUT_ROOT = "/data1/cx/brightcontrol/results/day2_longrun_eval/"
CUDA_DEVICE = "4"  # Use GPU 4

# Remaining checkpoints to evaluate
checkpoints = [
    {
        "name": "noise_offset_005_longrun_ckpt2000",
        "lora_dir": "/data1/cx/brightcontrol/train_logs/day2_best_method_longrun_offset005/checkpoint-2000",
    },
    {
        "name": "noise_offset_005_longrun_ckpt2500",
        "lora_dir": "/data1/cx/brightcontrol/train_logs/day2_best_method_longrun_offset005/checkpoint-2500",
    },
]

def main():
    # Create output root
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    
    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    
    for ckpt in checkpoints:
        out_dir = Path(OUTPUT_ROOT) / ckpt["name"]
        
        # Skip if already evaluated (has metrics.csv)
        if (out_dir / "metrics.csv").exists():
            print(f"Skipping {ckpt['name']} (already evaluated)")
            continue
        
        print(f"\nEvaluating {ckpt['name']}...")
        print(f"  LoRA dir: {ckpt['lora_dir']}")
        print(f"  Output dir: {out_dir}")
        
        # Build command
        cmd = [
            sys.executable, "eval_lora_smoke.py",
            "--model_path", BASE_MODEL,
            "--lora_dir", ckpt["lora_dir"],
            "--prompt_file", PROMPT_FILE,
            "--out_dir", str(out_dir),
            "--steps", "30",
            "--guidance", "7.5",
            "--height", "512",
            "--width", "512",
        ]
        
        try:
            # Run evaluation
            result = subprocess.run(
                cmd,
                env=env,
                cwd="/data1/cx/brightcontrol/scripts",
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes per checkpoint
            )
            
            if result.returncode == 0:
                print(f"  Success: {ckpt['name']}")
                print(f"  Output: {result.stdout[-500:] if result.stdout else 'No output'}")
            else:
                print(f"  Failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr[-1000:] if result.stderr else 'No stderr'}")
                # Continue with next checkpoint
                
        except subprocess.TimeoutExpired:
            print(f"  Timeout for {ckpt['name']}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()