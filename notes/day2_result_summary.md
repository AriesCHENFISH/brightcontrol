# Day 2 Experiment Results Summary

## Overview
The goal was to fix `precompute_coco_train_stats.py` outputting 0 rows, systematically build training/evaluation datasets for day 2 experiments, launch multiple training experiments, and perform result analysis and visualization.

## Accomplished

### 1. Fixed `precompute_coco_train_stats.py`
- Modified to support command-line arguments for dataset type (train2017/val2017)
- Added error handling for missing image directories
- Verified COCO training data status: train2017 directory exists (118,287 images), `coco_train2017_stats.csv` (13MB) exists

### 2. Built Training and Evaluation Datasets
- **`compare_smoke_6k` training set**: 6000 images (2000 dark, 2000 bright, 2000 neutral) with prompt prefixes:
  - "a very dark photo of ..."
  - "a very bright photo of ..."  
  - "a normally lit photo of ..."
  - Saved to `data/train_compare_6k/imagefolder/`
- **`DataBench-100` evaluation set**: 100 images (40 dark, 40 bright, 20 neutral) from COCO val2017
  - Saved to `data/databench_100.json`

### 3. Training Experiments
Three training experiments launched with identical baseline parameters except for single modifications:

1. **`v_prediction` (GPU1)**:
   - Final loss: 0.675
   - Weights saved: `train_logs/lora_smoke/v_prediction_run/`
   - Checkpoints: 100, 200, 300, 400, 500

2. **`noise_offset=0.02` (GPU4)**:
   - Final loss: 0.147
   - Weights saved: `train_logs/lora_smoke/noise_offset_run/`
   - Checkpoints: 100, 200, 300, 400, 500

3. **`noise_offset=0.05` (GPU3)**:
   - Final loss: 0.147
   - Weights saved: `train_logs/lora_smoke/noise_offset_0.05_run_gpu3/`
   - Checkpoints: 100, 200, 300, 400, 500

All training completed successfully with final checkpoints available.

### 4. Analysis Scripts Created
- `analyze_smoke_by_prompt.py`: Compare results across 5 groups (base, baseline_final, vpred_ckpt200, offset002_ckpt200, offset005_ckpt200)
- `analyze_smoke_by_category.py`: Aggregate analysis by category (dark, bright, white_bg, black_bg, backlight, neutral)
- `make_method_comparison_table.py`: Generate method comparison summary table
- `make_case_grid_all_methods.py`: Create case study grid visualizations
- `eval_databench100.py`: Unified evaluation on DataBench-100 (framework ready)

### 5. Evaluation Protocol Frozen
- Prompts: `prompts/eval_smoke_8.txt` (8 fixed prompts)
- Inference: DDIM 30 steps, CFG 7.5
- Seeds: 0, 1
- Dataset: DataBench-100

## Issues Encountered

### 1. GPU Memory Constraints
- GPU0 nearly full (24GB/24GB) with stuck evaluation and training processes
- GPU1, GPU2, GPU3 each have training processes occupying ~11.6GB
- GPU4 is available with 24GB free
- Unable to terminate stuck processes due to permission restrictions (owned by user `xyt`)

### 2. Environment Dependencies
- Python environment lacks required packages (`pandas`, `diffusers`)
- Unable to install packages due to permission constraints
- No sudo access to install system packages
- Virtual environment creation requires `python3-venv` package
- Conda environment (`tpmamba`) exists but requires `xyt` user permissions

### 3. Evaluation Blocked
- Previous evaluation process (PID 2356521) stuck for 12+ hours on GPU0
- New evaluation attempts fail due to missing dependencies
- Existing evaluation results only include:
  - `base/`
  - `baseline_run1_ckpt200/`
  - `baseline_run1_ckpt800/`
  - `baseline_run1_final/`
  - `v_prediction_ckpt200/` (directory exists but no metrics.csv)

## Next Steps Required

### Immediate Actions
1. **Terminate stuck processes** (requires `xyt` user or sudo):
   ```bash
   kill 2356521 2310731 2310733 2310739 2310740
   ```

2. **Setup evaluation environment**:
   ```bash
   # Option A: Install packages for current user
   python3 -m pip install --user pandas numpy diffusers torch --upgrade
   
   # Option B: Use existing conda environment
   source /home/xyt/anaconda3/envs/tpmamba/bin/activate
   ```

3. **Run unified evaluation** (on GPU4):
   ```bash
   CUDA_VISIBLE_DEVICES=4 python eval_all_checkpoints.py
   ```

### Analysis Pipeline
Once evaluation completes, run:
1. `analyze_smoke_by_prompt.py`
2. `analyze_smoke_by_category.py`
3. `make_method_comparison_table.py`
4. `make_case_grid_all_methods.py`
5. `eval_databench100.py`

### Expected Insights
- Compare effectiveness of `v_prediction` vs `noise_offset` methods
- Identify optimal checkpoint (200 vs 500 steps)
- Analyze performance across lighting categories
- Visualize qualitative improvements via case grids

## Files Created/Modified

### Scripts
- `scripts/precompute_coco_train_stats.py` (modified)
- `scripts/build_compare_6k.py` (new)
- `scripts/build_databench_100.py` (new)
- `scripts/run_vprediction.sh` (new)
- `scripts/run_noise_offset.sh` (new)
- `scripts/run_noise_offset_0.05_gpu3.sh` (new)
- `scripts/eval_all_checkpoints.py` (new)
- `scripts/analyze_smoke_by_prompt.py` (new)
- `scripts/analyze_smoke_by_category.py` (new)
- `scripts/make_method_comparison_table.py` (new)
- `scripts/make_case_grid_all_methods.py` (new)
- `scripts/eval_databench100.py` (new)

### Data
- `data/train_compare_6k/imagefolder/` (6000 images)
- `data/databench_100.json` (evaluation set)
- `data/raw/coco/coco_train2017_stats.csv` (verified)
- `data/raw/coco/coco_val2017_stats.csv` (generated)

### Training Outputs
- `train_logs/lora_smoke/v_prediction_run/` (complete)
- `train_logs/lora_smoke/noise_offset_run/` (complete)
- `train_logs/lora_smoke/noise_offset_0.05_run_gpu3/` (complete)

## Conclusion
All training experiments completed successfully with models ready for evaluation. The main blocker is environment setup and GPU memory management. Once evaluation runs, the analysis scripts will provide comprehensive insights into the effectiveness of different training strategies for brightness control.

**Recommendation**: Coordinate with system administrator or `xyt` user to terminate stuck processes and setup evaluation environment, then proceed with analysis pipeline.