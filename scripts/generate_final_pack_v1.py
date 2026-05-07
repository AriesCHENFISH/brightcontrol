#!/usr/bin/env python3
"""
Generate final_pack_v1 deliverables:
  Result 1: longrun_checkpoint_curve.png + longrun_checkpoint_summary.csv
  Result 2: promptbench_40.json
  Result 3: databench100_eval_summary.csv + databench100_eval_details.csv
  Result 4: case_grid_best_vs_base.png
"""
import json, csv, os, re, sys, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

ROOT = Path("/data1/cx/brightcontrol")
BASE_MODEL = "/data1/cx/sd15/"
OUT_DIR = ROOT / "results/final_pack_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def compute_mean_luminance(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(y.mean())

def extract_step(ckpt_name):
    m = re.search(r'ckpt(\d+)', str(ckpt_name))
    return int(m.group(1)) if m else 0

# ====================================================================
# RESULT 1: Long-run checkpoint trend
# ====================================================================
def result1_longrun():
    print("=" * 60)
    print("[Result 1] Generating longrun checkpoint trend...")
    longrun_dir = ROOT / "results/day2_longrun_eval"
    base_csv = ROOT / "results/lora_smoke_eval/base/metrics.csv"

    # Load base
    base_df = pd.read_csv(base_csv)
    base_avg = base_df.groupby('prompt_id')['mean_luminance'].mean()
    base_mean = base_avg.mean()

    checkpoints = sorted(
        [d for d in longrun_dir.iterdir() if d.name.startswith('noise_offset_005_longrun_ckpt')],
        key=lambda d: extract_step(d.name)
    )

    rows = []
    for ckpt_dir in checkpoints:
        step = extract_step(ckpt_dir.name)
        csv_path = ckpt_dir / "metrics.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        prompt_avg = df.groupby('prompt_id')['mean_luminance'].mean()
        overall_mean = prompt_avg.mean()
        deltas = prompt_avg - base_avg
        delta_mean = deltas.mean()
        delta_std = deltas.std(ddof=0) if len(deltas) > 1 else 0.0
        delta_min = deltas.min()
        delta_max = deltas.max()
        count = len(deltas)
        rows.append({
            'checkpoint': step,
            'mean_luminance': round(overall_mean, 6),
            'delta_mean': round(delta_mean, 6),
            'delta_std': round(delta_std, 6),
            'delta_min': round(delta_min, 6),
            'delta_max': round(delta_max, 6),
            'count': count,
        })
        print(f"  ckpt {step:>4d}:  delta_mean={delta_mean:.4f}  delta_std={delta_std:.4f}  mean_lum={overall_mean:.4f}")

    if not rows:
        print("  ERROR: No checkpoint data found")
        return

    # Save CSV
    csv_path = OUT_DIR / "longrun_checkpoint_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  Saved {csv_path}")

    # Plot
    df_plot = pd.DataFrame(rows)
    steps = df_plot['checkpoint'].values
    delta_mean = df_plot['delta_mean'].values
    delta_std = df_plot['delta_std'].values
    mean_lum = df_plot['mean_luminance'].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: delta mean
    color1 = '#1f77b4'
    ax1.errorbar(steps, delta_mean, yerr=delta_std, fmt='o-', capsize=5,
                 capthick=1.5, elinewidth=1.5, markersize=8,
                 color=color1, ecolor='gray', label=r'$\Delta$ mean luminance (vs base)')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero reference')
    ax1.fill_between(steps, delta_mean - delta_std, delta_mean + delta_std,
                     alpha=0.15, color=color1, label=r'$\pm 1\sigma$ (prompt divergence)')
    ax1.set_xlabel('Checkpoint Step', fontsize=13)
    ax1.set_ylabel(r'$\Delta$ Mean Luminance (vs base)', fontsize=13, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Secondary axis: std
    color2 = '#d62728'
    ax2 = ax1.twinx()
    ax2.plot(steps, delta_std, 's--', color=color2, markersize=7,
             label=r'$\Delta$ std (prompt divergence)')
    ax2.set_ylabel(r'$\Delta$ Std (prompt-level divergence)', fontsize=13, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    ax1.set_title('Long-run offset noise checkpoint trend', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / "longrun_checkpoint_curve.png"
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {png_path}")

# ====================================================================
# RESULT 2: PromptBench-40
# ====================================================================
def result2_promptbench():
    print("=" * 60)
    print("[Result 2] Generating promptbench_40.json...")
    src = ROOT / "prompts/promptbench_v0_40.json"
    dst = ROOT / "prompts/promptbench_40.json"
    data = json.loads(src.read_text(encoding='utf-8'))
    # Validate
    assert len(data) == 40, f"Expected 40, got {len(data)}"
    cats = set(x['category'] for x in data)
    required = {'dark', 'bright', 'backlight', 'white_bg', 'black_bg'}
    assert required.issubset(cats), f"Missing categories: {required - cats}"
    # Fix background_purity consistency
    for item in data:
        if item['category'] == 'white_bg':
            item['background_purity'] = 'white'
        elif item['category'] == 'black_bg':
            item['background_purity'] = 'black'
        elif item['category'] in ('dark', 'bright', 'backlight'):
            if item['background_purity'] is not None:
                item['background_purity'] = None
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"  Saved {dst}  ({len(data)} items, {len(cats)} categories)")

# ====================================================================
# RESULT 3: DataBench-100 evaluation (generates images)
# ====================================================================
DATABENCH_PATH = ROOT / "data/databench_100.json"
DATABENCH_CACHE = OUT_DIR / "databench_images"
METHODS_DB = [
    {
        'name': 'base',
        'lora_dir': None,
        'label': 'base',
    },
    {
        'name': 'baseline_lora',
        'lora_dir': str(ROOT / "train_logs/lora_smoke/baseline_run1" / "pytorch_lora_weights.safetensors"),
        'label': 'baseline_lora',
    },
    {
        'name': 'offset005_best',
        'lora_dir': str(ROOT / "train_logs/day2_best_method_longrun_offset005/checkpoint-2500" / "pytorch_lora_weights.safetensors"),
        'label': 'offset005_best',
    },
]

def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, use_safetensors=True,
        local_files_only=True, safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    return pipe, device

def generate_databench_images(pipe, items, lora_path, out_img_dir, method_name):
    """Generate one image per item using lora_path (or None for base)."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    # Unload previous LoRA
    pipe.unload_lora_weights()
    if lora_path:
        pipe.load_lora_weights(lora_path)
    device = pipe.device
    rows = []
    for idx, item in enumerate(items):
        prompt = item['text']
        item_id = item.get('image_id', idx)
        img_path = out_img_dir / f"{item_id}.png"
        if img_path.exists():
            # Reuse cached
            img = Image.open(img_path).convert("RGB")
        else:
            generator = torch.Generator(device=device).manual_seed(42)
            img = pipe(
                prompt=prompt, num_inference_steps=30, guidance_scale=7.5,
                generator=generator, height=512, width=512,
            ).images[0]
            img.save(img_path)
        gen_lum = compute_mean_luminance(img)
        target_lum = item['mean_luminance']
        abs_err = abs(gen_lum - target_lum)
        rows.append({
            'method': method_name,
            'item_id': item_id,
            'label': item['brightness_label'],
            'prompt': prompt,
            'target_luminance': target_lum,
            'generated_mean_luminance': gen_lum,
            'abs_error': abs_err,
            'image_path': str(img_path),
        })
        if (idx + 1) % 20 == 0:
            print(f"    [{method_name}] {idx+1}/{len(items)} generated, avg_abs_err={np.mean([r['abs_error'] for r in rows]):.4f}")
    return rows

def result3_databench():
    print("=" * 60)
    print("[Result 3] DataBench-100 evaluation (generating images on GPU)...")
    items = json.loads(DATABENCH_PATH.read_text(encoding='utf-8'))
    print(f"  Loaded {len(items)} items from DataBench-100")
    pipe, device = load_pipeline()
    print(f"  Pipeline loaded on {device}")

    all_details = []
    for meth in METHODS_DB:
        out_img_dir = DATABENCH_CACHE / meth['name']
        print(f"  Generating for {meth['name']} ...")
        start = time.time()
        detail_rows = generate_databench_images(pipe, items, meth['lora_dir'], out_img_dir, meth['name'])
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.0f}s, {len(detail_rows)} images")
        all_details.extend(detail_rows)

    # Build summary
    df = pd.DataFrame(all_details)
    summary_rows = []
    for method in df['method'].unique():
        sub = df[df['method'] == method]
        mae = sub['abs_error'].mean()
        mae_dark = sub[sub['label'] == 'dark']['abs_error'].mean() if 'dark' in sub['label'].values else None
        mae_bright = sub[sub['label'] == 'bright']['abs_error'].mean() if 'bright' in sub['label'].values else None
        mae_neutral = sub[sub['label'] == 'neutral']['abs_error'].mean() if 'neutral' in sub['label'].values else None
        summary_rows.append({
            'method': method,
            'num_samples': len(sub),
            'mae': round(mae, 6),
            'mae_dark': round(mae_dark, 6) if mae_dark is not None else '',
            'mae_bright': round(mae_bright, 6) if mae_bright is not None else '',
            'mae_neutral': round(mae_neutral, 6) if mae_neutral is not None else '',
            'mean_generated_luminance': round(sub['generated_mean_luminance'].mean(), 6),
            'mean_target_luminance': round(sub['target_luminance'].mean(), 6),
        })

    # Save summary
    sum_csv = OUT_DIR / "databench100_eval_summary.csv"
    pd.DataFrame(summary_rows).to_csv(sum_csv, index=False, encoding='utf-8')
    print(f"  Saved {sum_csv}")

    # Save details
    det_csv = OUT_DIR / "databench100_eval_details.csv"
    df.to_csv(det_csv, index=False, encoding='utf-8')
    print(f"  Saved {det_csv}")

    # Print summary
    for r in summary_rows:
        print(f"    {r['method']:20s}  MAE={r['mae']:.4f}  dark={r['mae_dark']}  bright={r['mae_bright']}  neutral={r['mae_neutral']}")

# ====================================================================
# RESULT 4: Case grid (base vs baseline_lora vs offset005_best)
# ====================================================================
CASE_PROMPT_IDS = [0, 1, 2, 3, 4, 6]  # dark, bright, white_bg, black_bg, backlight, neutral
CASE_LABELS = ['Dark', 'Bright', 'White BG', 'Black BG', 'Backlight', 'Neutral']
CASE_METHODS = [
    ('base', 'results/lora_smoke_eval/base'),
    ('baseline_lora', 'results/lora_smoke_eval/baseline_run1_final'),
    ('offset005_best', 'results/day2_longrun_eval/noise_offset_005_longrun_ckpt2500'),
]

def result4_case_grid():
    print("=" * 60)
    print("[Result 4] Generating case grid...")
    thumb = (280, 280)
    n_rows = len(CASE_METHODS)
    n_cols = len(CASE_PROMPT_IDS)

    grid_w = n_cols * thumb[0]
    grid_h = n_rows * thumb[1]
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for row_idx, (method_name, method_path) in enumerate(CASE_METHODS):
        for col_idx, pid in enumerate(CASE_PROMPT_IDS):
            img_path = ROOT / method_path / "images" / f"p{pid:02d}_s0.png"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                lum = compute_mean_luminance(img)
                # Check if black image (safety filter)
                if lum < 0.01:
                    print(f"  WARNING: near-black image at {img_path}, rechecking...")
            else:
                img = Image.new("RGB", thumb, (80, 80, 80))
                print(f"  WARNING: missing {img_path}")
            img = ImageOps.fit(img, thumb)
            x = col_idx * thumb[0]
            y = row_idx * thumb[1]
            grid.paste(img, (x, y))

    # Add row labels
    draw = ImageDraw.Draw(grid)
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # We'll add labels via matplotlib since PIL label placement is tedious
    # Save the raw grid and overlay with matplotlib
    raw_grid_path = OUT_DIR / "_case_grid_raw.png"
    grid.save(raw_grid_path)

    # Create annotated figure with matplotlib
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    for row_idx, (method_name, _) in enumerate(CASE_METHODS):
        for col_idx, pid in enumerate(CASE_PROMPT_IDS):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            img_path = ROOT / CASE_METHODS[row_idx][1] / "images" / f"p{pid:02d}_s0.png"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img_thumb = ImageOps.fit(img, thumb)
                ax.imshow(img_thumb)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12, color='gray')
            if row_idx == 0:
                ax.set_title(CASE_LABELS[col_idx], fontsize=16, fontweight='bold', pad=8)
            if col_idx == 0:
                ax.set_ylabel(method_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.tight_layout(pad=0.5)
    png_path = OUT_DIR / "case_grid_best_vs_base.png"
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {png_path}")
    # Clean up raw
    if raw_grid_path.exists():
        raw_grid_path.unlink()

# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    # Result 1 (analysis only)
    result1_longrun()
    # Result 2 (copy+fix)
    result2_promptbench()
    # Result 3 (generates images on GPU, time-consuming)
    result3_databench()
    # Result 4 (existing images)
    result4_case_grid()
    print("=" * 60)
    print("ALL DONE. Files in:", OUT_DIR)
