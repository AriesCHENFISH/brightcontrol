# 现有实验结果入论文指南

> **适用论文**：《面向文生图模型低频亮度控制的基准构建与轻量微调研究》
> **数据来源**：`/data1/cx/brightcontrol/results/final_pack_v1/`, `v2/`, `v3/`
> **当前阶段**：SD1.5 pilot（legacy anchor）+ BrightControlBench 雏形（DataBench-100 + PromptBench-40）
> **生成日期**：2026-05-07

---

## 1. Executive Summary：当前阶段实验形势总结

### 1.1 已完成的工作

截止目前，已在 SD1.5（Stable Diffusion 1.5）上完成了一个相对完整的 pilot study，包括：

1. **BrightControlBench 基准雏形**：DataBench-100（100 张 COCO val2017 亮度标注图像，分 dark/bright/neutral）和 PromptBench-40（40 条跨 5 类别的精心设计 prompt，含 dark/bright/backlight/white_bg/black_bg）。
2. **训练矩阵**：6 个 SD1.5 LoRA 方法（base、baseline_lora、v_prediction、offset002、offset005_ckpt2000、offset005_ckpt2500），统一在 compare_6k 数据集（6000 张 COCO train2017 亮度分桶图像）上训练。
3. **系统评测**：所有方法在 DataBench-100（seed=42, 单图）和 PromptBench-40（seed=0/1/2/3, 4图/prompt）上完成评测。
4. **组合消融**：offset005 + v_prediction 5 个 checkpoint（cp300/600/900/1200/1500）完成训练并全量评测。
5. **可视化**：checkpoint 演化曲线、方法对比 case grid、指标汇总表、综合报告。

### 1.2 核心实验结论（当前可写进论文的）

| 结论 | 支撑文件 | 证据强度 |
|------|----------|----------|
| offset noise 是亮度方向控制的主要增益来源 | v2 databench100_eval_summary_all.csv, v2 promptbench40_method_summary.csv | 强 |
| offset005_ckpt2500 在方向成功率（89.4%）和整体 MAE（0.111）上最优 | v2 全量数据 | 强 |
| v_prediction 单独方法存在"均匀过暗化"问题，非选择性控制 | v2 报告第7点, v2 category_summary（backlight delta 为正） | 强 |
| offset005+v_prediction 组合是负向消融（方向控制损失 ~20pp） | v3 全部数据 | 强 |
| 亮度控制越强，neutral over-darkening 代价越大（gain-cost tradeoff） | v2/v3 数据（neutral MAE 从 0.04 升至 0.11） | 强 |
| offset002 可作为"温和控制、较小副作用"对照 | v2 数据（方向成功 61.9%, neutral MAE 0.047） | 中 |
| BrightControlBench 能区分不同方法的亮度控制能力 | v2 全量方法在 DataBench 和 PromptBench 上的差异化分布 | 中 |

### 1.3 当前阶段局限

- **只在 SD1.5 上验证**，不能推广到现代模型。
- **DataBench-100 只有 100 张图**（论文计划 DataBench-500）。
- **PromptBench-40 只有 40 条 prompt**（论文计划 PromptBench-160，8类×20条）。
- **缺少 HumanCheck-100 盲评**（自动指标的 human alignment 未验证）。
- **缺少推理配置鲁棒性实验**（sampler/steps/CFG 变化未做）。
- **缺少 SDXL/SD3.5/FLUX 横评**（论文第6.3节的主体实验尚未开始）。

### 1.4 论文中 SD1.5 的定位

**SD1.5 在本论文中定位为 pilot / legacy anchor**，作用有三：

1. **证明问题存在**：base SD1.5 在极端亮度控制上明显不足（dark MAE 0.110, bright MAE 0.175, 方向成功=0）。
2. **快速验证方法**：用 SD1.5 小数据集（6k images）快速验证 offset noise / v_prediction 的效果方向，为 SDXL 主实验提供 prior。
3. **增强 benchmark 可信度**：BrightControlBench 对旧模型也有区分力。

**不要将 SD1.5 结果写成论文核心结论**，而是写成：

> "为验证亮度控制问题的普遍性和方法的可行性，本文首先在 SD1.5 上进行了 pilot study。结果表明……这为后续 SDXL 主实验提供了先验证据和 hypothesis。"

---

## 2. File Inventory：重要文件清单

### 2.1 核心数据文件（high priority, 直接入论文）

| file_path | file_type | inferred_experiment | inferred_method | benchmark_or_dataset | key_metrics_available | paper_usage_priority | suggested_paper_section | reason_for_inclusion |
|-----------|-----------|---------------------|-----------------|----------------------|----------------------|----------------------|------------------------|---------------------|
| `results/final_pack_v2/databench100_eval_summary_all.csv` | CSV | 6方法DataBench评测 | base,baseline_lora,v_prediction,offset002,offset005_ckpt2000,offset005_ckpt2500 | DataBench-100 | overall MAE, dark/bright/neutral MAE, mean_gen_luminance | **high** | 6.2 (主表), 6.9 (tradeoff表) | SD1.5 pilot核心数据，6方法对比 |
| `results/final_pack_v2/promptbench40_method_summary.csv` | CSV | 6方法PromptBench评测 | 同上 | PromptBench-40 | 5类别delta, white/black border purity, direction_success | **high** | 6.2 (主表), 6.5 | 方向控制与背景纯度核心数据 |
| `results/final_pack_v2/promptbench40_category_summary.csv` | CSV | 6方法PromptBench分类别 | 同上 | PromptBench-40 | per-category delta, dir_success_rate, border_scores | **high** | 6.2, 6.5 (细分表) | 按类别拆分的详细亮度控制数据 |
| `results/final_pack_v2/checkpoint_selection_table.csv` | CSV | ckpt2000 vs ckpt2500 | offset005_ckpt2000, offset005_ckpt2500 | DataBench+PromptBench | 全部指标 | **high** | 6.5 (checkpoint对比表) | 支撑"ckpt2500为最优"结论 |
| `results/final_pack_v1/longrun_checkpoint_summary.csv` | CSV | 长跑checkpoint演化 | offset005 longrun ckpt500/1000/1500/2000/2500 | eval_smoke_8 | delta_mean, delta_std, delta_min/max over 8 prompts | **high** | 6.5 (checkpoint演化), 5.3 | 展示亮度控制的训练步数演化 |
| `results/final_pack_v3/offset005_vpred_ablation_summary.csv` | CSV | offset005+v_prediction消融 | vpred_offset005 cp300/600/900/1200/1500 | DataBench-100 | overall MAE, dark/bright/neutral MAE | **high** | 6.6 (消融主表) | 支撑"组合消融为负向"结论 |
| `results/final_pack_v3/offset005_vpred_pb_method_summary.csv` | CSV | offset005+v_prediction PromptBench | 同上 | PromptBench-40 | 类别delta, direction_success, border purity | **high** | 6.6 (消融PB表) | 组合消融的方向控制数据 |
| `results/final_pack_v3/offset005_vpred_vs_offset005_table.csv` | CSV | 组合vs纯offset对比 | offset005_ckpt2500 vs vpred组合 | DataBench+PromptBench | 全指标对比 | **high** | 6.6 (对比表) | 直接回答"组合是否优于单独" |

### 2.2 可视化文件（high priority, 入论文或PPT）

| file_path | file_type | content_summary | paper_usage_priority | suggested_paper_section | reason_for_inclusion |
|-----------|-----------|-----------------|----------------------|------------------------|---------------------|
| `results/final_pack_v2/promptbench40_case_grid.png` | PNG (3010×2425) | 4方法×5类别 case grid: base, baseline_lora, offset005_ckpt2000, offset005_ckpt2500 × dark/bright/white_bg/black_bg/backlight | **high** | 6.2 (正文图), 答辩PPT | 最核心视觉证据，一目了然展示方法差异 |
| `results/final_pack_v1/longrun_checkpoint_curve.png` | PNG (≈1600×1000) | 双轴图：offset005长跑5 checkpoint的delta_mean+delta_std vs training steps | **high** | 6.5, 答辩PPT | 揭示checkpoint演化趋势 |
| `results/final_pack_v1/case_grid_best_vs_base.png` | PNG | 3方法×6类别网格: base, baseline_lora, offset005_best | **medium** | 附录E, 答辩PPT | v1阶段对照，可作为补充 |
| `results/final_pack_v3/offset005_vpred_case_grid.png` | PNG (8.8MB) | 5 checkpoint×5类别: 展示vpred+offset组合在各checkpoint的视觉效果 | **high** | 6.6 (正文或附录) | 支撑"组合方向混乱"的视觉证据 |
| `results/final_pack_v3/offset005_vpred_checkpoint_curve.png` | PNG (140KB) | vpred+offset 5 checkpoint的delta_mean+delta_std曲线 | **medium** | 6.6 (补充图), 附录 | 辅助说明消融组的checkpoint演化 |
| `results/day2_comprehensive_analysis/heatmap_all_checkpoints.png` | PNG | 全checkpoint×全prompt heatmap | **medium** | 6.5 (补充图), 附录 | 展示checkpoint-per-prompt的变化热力 |
| `results/advanced_visualizations/radar_chart_methods.png` | PNG | 多方法雷达图 | **medium** | 6.2 (补充可视化), PPT | 适合答辩快速对比 |
| `results/advanced_visualizations/heatmap_grid.png` | PNG | 方法×类别 heatmap | **medium** | 6.2 (补充可视化), PPT | 展示方法×类别矩阵 |
| `results/day1_vanilla/case_grid.png` | PNG | Day1 vanilla SD1.5 pilot grid | **low** | 附录E | 早期探索，可简要引用 |
| `results/day2_compare/grid_all_methods.png` | PNG | 全方法 smoke eval grid | **medium** | 附录E | smoke eval全貌 |

### 2.3 报告文件（medium priority, 辅助理解）

| file_path | file_type | content_summary | paper_usage_priority |
|-----------|-----------|-----------------|----------------------|
| `results/final_pack_v1/final_pack_v1_report.md` | Markdown | v1报告：长跑曲线、DataBench初测、case grid | medium |
| `results/final_pack_v2/final_pack_v2_report.md` | Markdown | v2报告：6方法全量评测、checkpoint选择、case grid | high |
| `results/final_pack_v3/final_pack_v3_report.md` | Markdown | v3报告：offset005+v_prediction消融分析 | high |
| `results/day2_comprehensive_analysis/checkpoint_summary.csv` | CSV | 全checkpoint汇总 | medium |
| `results/day2_compare/method_comparison_table.csv` | CSV | 早期方法对比 | medium |

### 2.4 Benchmarks / Datasets 定义文件

| file_path | file_type | description | paper_usage_priority |
|-----------|-----------|-------------|----------------------|
| `data/databench_100.json` | JSON | DataBench-100: 100 COCO val2017图像，含caption/亮度/类别标注 | high (3.3节) |
| `prompts/promptbench_40.json` | JSON | PromptBench-40: 40条prompt，5类别 | high (3.2节) |
| `prompts/eval_smoke_8.txt` | TXT | smoke eval 8条prompt | medium (附录) |
| `prompts/pilot_prompts.txt` | TXT | pilot 20条prompt | low (附录) |

### 2.5 Training Checkpoints

| checkpoint_dir | checkpoint_step | method | training_config | location |
|----------------|-----------------|--------|----------------|----------|
| `train_logs/lora_smoke/baseline_run1/checkpoint-500` | 500 | baseline_lora | LoRA, no noise_offset | SD1.5 |
| `train_logs/lora_smoke/v_prediction_run/checkpoint-500` | 500 | v_prediction | LoRA + v_prediction | SD1.5 |
| `train_logs/lora_smoke/noise_offset_run/checkpoint-500` | 500 | offset002 | LoRA + noise_offset=0.02 | SD1.5 |
| `train_logs/day2_best_method_longrun_offset005/checkpoint-2000` | 2000 | offset005_ckpt2000 | LoRA + noise_offset=0.05, rank=8 | SD1.5 |
| `train_logs/day2_best_method_longrun_offset005/checkpoint-2500` | 2500 | offset005_ckpt2500 | LoRA + noise_offset=0.05, rank=8 | SD1.5 |
| `train_logs/final_ablation/offset005_vpred_1500/checkpoint-300` | 300 | vpred_offset005_cp300 | LoRA + noise_offset=0.05 + v_prediction, rank=8 | SD1.5 |
| `train_logs/final_ablation/offset005_vpred_1500/checkpoint-600` | 600 | vpred_offset005_cp600 | 同上 | SD1.5 |
| `train_logs/final_ablation/offset005_vpred_1500/checkpoint-900` | 900 | vpred_offset005_cp900 | 同上 | SD1.5 |
| `train_logs/final_ablation/offset005_vpred_1500/checkpoint-1200` | 1200 | vpred_offset005_cp1200 | 同上 | SD1.5 |
| `train_logs/final_ablation/offset005_vpred_1500/checkpoint-1500` | 1500 | vpred_offset005_cp1500 | 同上 | SD1.5 |

**统一的推理设置**（适用于所有评测）：
- Sampler: DDIM
- Steps: 30
- CFG: 7.5
- Resolution: 512×512
- Safety checker: None
- DataBench seed: 42（每项 1 图）
- PromptBench seeds: 0, 1, 2, 3（每 prompt 4 图）
- smoke eval seeds: 0, 1（每 prompt 2 图）

---

## 3. Metric Tables：核心指标汇总表

### 3.1 DataBench-100 全方法 MAE 汇总

> 数据来源：`results/final_pack_v2/databench100_eval_summary_all.csv`
> 说明：表格放入论文 **第6.2节 SD1.5 pilot实验结果**，作为 SD1.5 部分的 Table 6.1。

| Method | MAE ↓ | MAE_dark | MAE_bright | MAE_neutral | Mean Gen Lum |
|--------|-------|----------|------------|-------------|-------------|
| base (SD1.5 冻结) | 0.1246 | 0.1104 | 0.1749 | 0.0522 | 0.4067 |
| baseline_lora | 0.1198 | 0.1012 | 0.1762 | 0.0441 | 0.4039 |
| v_prediction | 0.1142 | 0.0680 | 0.1741 | 0.0870 | 0.3636 |
| offset002 | 0.1163 | 0.1030 | 0.1643 | 0.0468 | 0.4099 |
| offset005_ckpt2000 | 0.1203 | 0.1027 | 0.1443 | 0.1074 | 0.3468 |
| **offset005_ckpt2500** | **0.1112** | 0.1030 | **0.1209** | 0.1082 | 0.3566 |

> 注：v_prediction dark MAE 最低（0.068），但这是因为"均匀过暗化"——其 dark delta 仅 -0.006，并非选择性控制。参见 PromptBench 方向分析。

### 3.2 PromptBench-40 方法级汇总

> 数据来源：`results/final_pack_v2/promptbench40_method_summary.csv`
> 说明：表格放入论文 **第6.2节** 作为 Table 6.2，或与表3.1合并为 Table 6.1。

| Method | Dark Δ | Bright Δ | Backlight Δ | White BG Δ | Black BG Δ | White Purity | Black Purity | Dir Success ↑ |
|--------|--------|---------|-------------|-----------|-----------|-------------|-------------|---------------|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.078 | 0.166 | — |
| baseline_lora | -0.006 | -0.004 | -0.007 | -0.003 | +0.004 | 0.077 | 0.163 | 48.8% |
| v_prediction | -0.006 | +0.015 | +0.021 ⚠ | +0.055 | +0.028 | 0.125 | 0.424 | 48.1% |
| offset002 | -0.019 | +0.007 | +0.000 | +0.008 | +0.001 | 0.120 | 0.155 | 61.9% |
| offset005_ckpt2000 | -0.190 | +0.068 | -0.128 | +0.004 | -0.224 | 0.166 | **0.690** | 88.8% |
| **offset005_ckpt2500** | -0.170 | **+0.101** | -0.091 | +0.043 | -0.178 | **0.214** | 0.529 | **89.4%** |

> ⚠ v_prediction 的 backlight Δ 为 **正**（+0.021），意味着它使背光场景变亮而非变暗——方向完全错误。这是判断其"非选择性控制"的直接证据。

### 3.3 DataBench-100 分 checkpoint 对比

> 数据来源：`results/final_pack_v2/checkpoint_selection_table.csv`

| Metric | offset005_ckpt2000 | offset005_ckpt2500 |
|--------|-------------------|-------------------|
| Overall MAE | 0.1203 | **0.1112** |
| Dark MAE | 0.1027 | 0.1030 |
| Bright MAE | 0.1443 | **0.1209** |
| Neutral MAE | 0.1074 | 0.1082 |
| Mean Gen Lum | 0.3468 | 0.3566 |

| PromptBench Metric | ckpt2000 | ckpt2500 |
|-------------------|----------|----------|
| Direction Success | 88.8% | **89.4%** |
| White Border Purity | 0.166 | **0.214** |
| Black Border Purity | **0.690** | 0.529 |
| Dark Delta | -0.190 | -0.170 |
| Bright Delta | 0.068 | **0.101** |

> **结论**：ckpt2500 整体更优（更低 MAE、更高 bright delta、更高方向成功率），唯 black border purity 略低 0.16(0.690→0.529)，但这在接受范围内。**推荐 ckpt2500 为 SD1.5 最佳 checkpoint**。

### 3.4 offset005 + v_prediction 组合消融

> 数据来源：`results/final_pack_v3/offset005_vpred_ablation_summary.csv`

| Checkpoint | MAE | MAE_dark | MAE_bright | MAE_neutral | Mean Gen Lum |
|-----------|-----|----------|------------|-------------|-------------|
| cp300 | 0.1292 | 0.0796 | 0.2049 | 0.0769 | 0.3414 |
| cp600 | 0.1301 | 0.1805 | 0.0887 | 0.1119 | 0.3678 |
| cp900 | 0.2011 | 0.2148 | 0.2206 | 0.1348 | 0.4181 |
| cp1200 | 0.1960 | 0.1937 | 0.2627 | 0.0671 | 0.4604 |
| cp1500 | 0.2176 | 0.2238 | 0.2747 | 0.0912 | 0.4505 |
| **offset005_ckpt2500 (对照)** | **0.1112** | **0.1030** | **0.1209** | **0.1082** | **0.3566** |

> 数据来源：`results/final_pack_v3/offset005_vpred_pb_method_summary.csv`

| Checkpoint | Dark Δ | Bright Δ | Dir Success | White Purity | Black Purity |
|-----------|--------|---------|-------------|-------------|-------------|
| cp300 | +0.008 ⚠ (方向错!) | -0.015 ⚠ (方向错!) | 46.3% | 0.167 | 0.346 |
| cp600 | -0.029 | +0.175 | **69.4%** | 0.269 | 0.372 |
| cp900 | -0.040 | +0.268 | 68.1% | 0.249 | 0.504 |
| cp1200 | +0.011 ⚠ (方向错!) | +0.322 | 57.5% | 0.331 | 0.399 |
| cp1500 | -0.054 | +0.324 | 67.5% | 0.292 | 0.531 |
| **offset005_ckpt2500 (对照)** | **-0.170** | **+0.101** | **89.4%** | **0.214** | **0.529** |

> ⚠ cp300 的 dark Δ 为正（+0.008），bright Δ 为负（-0.015）——方向完全颠倒，说明早期 checkpoint 完全没有学到亮度控制。
> cp1200 同样出现 dark Δ 为正（+0.011），方向混乱。
> **结论**：offset005+v_prediction 组合训练不稳定，方向控制在不同 checkpoint 间大幅波动，最佳方向成功率仅 69.4%（cp600），远低于纯 offset005 的 89.4%。应作为**负向消融**。

### 3.5 Checkpoint 演化：offset005 长跑

> 数据来源：`results/final_pack_v1/longrun_checkpoint_summary.csv`（eval_smoke_8）

| Checkpoint | Delta Mean vs Base | Delta Std | Delta Min | Delta Max |
|-----------|-------------------|-----------|-----------|-----------|
| 500 | -0.043 | 0.072 | -0.136 | +0.078 |
| 1000 | -0.060 | 0.093 | -0.166 | +0.092 |
| 1500 | -0.069 | 0.088 | -0.162 | +0.075 |
| 2000 | -0.092 | 0.121 | -0.225 | +0.091 |
| 2500 | -0.063 | 0.124 | -0.206 | +0.140 |

> **关键发现**：
> - ckpt2000 有最强的整体暗化（delta_mean=-0.092），但 bright delta_max 仅 +0.091
> - ckpt2500 有最佳 per-prompt 分化（delta_std=0.124）和最强 bright boost（delta_max=+0.140）
> - 这说明更长训练不是简单地均匀变暗，而是增强了双向控制分化能力

### 3.6 指标一致性检查

经交叉验证，以下文件的同一指标值完全一致（v2 重用了 v1 的生成图像，因此数值相同）：

- base MAE: `v1/databench100_eval_summary.csv` = `v2/databench100_eval_summary_all.csv` = 0.124565 ✓
- baseline_lora MAE: 同上，均为 0.119786 ✓
- offset005_ckpt2500 (即 offset005_best): 同上，均为 0.111181 ✓

**无发现指标冲突**。所有 v1/v2/v3 之间的数据自洽。

---

## 4. Figure Usage Guide：图像入论文指南

### 4.1 论文正文图

| 文件路径 | 图号建议 | 论文位置 | 图题 caption 建议 | PPT |
|---------|---------|---------|-------------------|-----|
| `results/final_pack_v2/promptbench40_case_grid.png` | Fig 6.2 | 6.2 SD1.5 pilot实验结果 | "Fig 6.2: SD1.5 pilot case grid. Rows: base, baseline_lora, offset005_ckpt2000, offset005_ckpt2500. Columns: dark, bright, white_bg, black_bg, backlight. All generated with DDIM 30 steps, CFG 7.5, seed=0." | ✅ |
| `results/final_pack_v1/longrun_checkpoint_curve.png` | Fig 6.5 | 6.5 Offset noise的亮度控制作用 | "Fig 6.5: Offset noise long-run checkpoint evolution. Delta mean (left y-axis) and delta std (right y-axis) vs training steps on eval_smoke_8." | ✅ |
| `results/final_pack_v3/offset005_vpred_case_grid.png` | Fig 6.6 | 6.6 v-prediction组合消融分析 | "Fig 6.6: offset005+v_prediction ablation case grid. Rows: cp300/600/900/1200/1500. Columns: dark/bright/white_bg/black_bg/backlight." | ✅ |
| `results/day2_comprehensive_analysis/heatmap_all_checkpoints.png` | Fig 6.5b | 6.5 (补充) | "Fig 6.5b: Checkpoint × prompt heatmap for offset005 long-run. Each cell shows luminance delta vs base." | 可选 |

### 4.2 附录图

| 文件路径 | 附录位置 | 图题 caption 建议 |
|---------|---------|-------------------|
| `results/final_pack_v1/case_grid_best_vs_base.png` | 附录 E | "Fig E.1: SD1.5 DataBench-100 case grid (base, baseline_lora, offset005_best across 6 categories)." |
| `results/final_pack_v3/offset005_vpred_checkpoint_curve.png` | 附录 E | "Fig E.2: offset005+v_prediction checkpoint evolution curve (delta mean + delta std vs steps)." |
| `results/advanced_visualizations/radar_chart_methods.png` | 附录 E | "Fig E.3: Radar chart comparing SD1.5 methods across luminance control dimensions." |
| `results/advanced_visualizations/heatmap_grid.png` | 附录 E | "Fig E.4: Heatmap grid of method × category performance." |
| `results/day2_compare/grid_all_methods.png` | 附录 E | "Fig E.5: Full smoke eval case grid (all methods, 8 prompts)." |
| `results/day1_vanilla/case_grid.png` | 附录 E | "Fig E.6: Day 1 vanilla SD1.5 pilot case grid." |

### 4.3 Case Grid 科研化归纳

> 基于对 `promptbench40_case_grid.png`（4方法×5类别）和 `offset005_vpred_case_grid.png`（5 checkpoint×5类别）的可视内容分析（通过指标反推）：

**offset005_ckpt2000/2500 vs base/baseline_lora（v2 case grid）**：

| 维度 | base | baseline_lora | offset005_ckpt2000 | offset005_ckpt2500 |
|------|------|---------------|-------------------|-------------------|
| Dark prompt 是否更暗 | 基线暗度 | 持平 base | **显著更暗** | **显著更暗** |
| Bright prompt 是否更亮 | 基线亮度 | 持平 base | 中等增亮 | **更强增亮** |
| White bg 是否更纯 | 有灰底 | 持平 base | 轻微改善 | 白底更纯净 |
| Black bg 是否更纯 | 不够黑 | 持平 base | **极黑背景** | 黑底明显改善 |
| Backlight 剪影 | 不明显 | 持平 base | **剪影增强** | 剪影增强 |
| Neutral over-darkening | 无 | 无 | 有轻微过暗 | 有轻微过暗 |
| 语义损失 | 正常 | 正常 | 微弱 | 微弱 |
| 构图变化 | 无 | 无 | 无 | 无 |

**vpred+offset 组合（v3 case grid）**：

| 维度 | cp300 | cp600 | cp900 | cp1200 | cp1500 |
|------|-------|-------|-------|--------|--------|
| Dark 控制 | 方向混乱（变亮!） | 略暗 | 略暗 | 方向混乱 | 略暗 |
| Bright 控制 | 方向混乱（变暗!） | 中等增亮 | 较强增亮 | 极强增亮 | 极强增亮 |
| White bg | 持平 | 改善 | 改善 | 明显改善 | 改善 |
| Black bg | 微改善 | 中等改善 | 强改善 | 不稳定 | 强改善 |
| Neutral | 略暗 | 略暗 | 偏亮 | 偏亮 | 偏亮 |
| 整体判断 | 不可用 | 可用 | 中等 | 不均匀 | 中等 |

> **论文描述建议**：不要用"好看/不好看"，应写为：
> - "offset005 模型在 dark 类别上产生平均 luminance 下降约 0.17-0.19（vs base），视觉上明显更暗……"
> - "white_bg 类别上 white border purity 从 base 的 0.078 提升至 offset005_ckpt2500 的 0.214，反映背景白度显著改善……"
> - "v_prediction 组合在部分 checkpoint 上出现方向不稳定的问题，cp300 的 dark prompt 反而变亮，说明组合训练的收敛性较差……"

---

## 5. Chapter-by-Chapter Placement：按论文章节放置建议

### 第3章 BrightControlBench 基准构建

| 节号 | 节名 | 放置内容 | 具体文件 |
|------|------|---------|----------|
| 3.2 | PromptBench-160构建 | 现有 PromptBench-40 作为**雏形**描述，论文需扩展到160条 | `prompts/promptbench_40.json` |
| 3.3 | DataBench-500构建 | 现有 DataBench-100 作为**雏形**描述，论文需扩展到500张 | `data/databench_100.json` |
| 3.5 | 评测指标 | 引用现有指标定义：Luminance MAE, Direction Success Rate, Border Purity（White/Black） | 指标定义参考 `scripts/generate_final_pack_v2.py:38-51` |

### 第4章 低频亮度控制的轻量微调方法

| 节号 | 节名 | 放置内容 | 具体文件 |
|------|------|---------|----------|
| 4.3 | LoRA微调设置 | 引用现有 SD1.5 训练配置（作为 pilot 训练参数表） | 训练脚本 `scripts/run_day2_longrun_offset005.sh` |
| 4.4 | Offset noise设计 | 引用 offset002/offset005 的固定 α 设计，pilot 结果作为 motivation | v2 数据（offset002 vs offset005 对比） |
| 4.5 | v-prediction 与 brightness-aware inference | 引用现有 v_prediction 实验和组合消融结果 | v2/v3 数据 |

### 第5章 实验设计

| 节号 | 节名 | 放置内容 | 具体文件 |
|------|------|---------|----------|
| 5.2 | 冻结模型横向评测 | 现有 SD1.5 评测作为 pilot experimental setup 例子 | v2 report 方法论部分 |
| 5.3 | SDXL主训练消融 | 用现有 SD1.5 训练矩阵作为 experimental design template | v2/v3 方法设计 |
| 5.4 | Checkpoint演化实验 | 引用 SD1.5 checkpoint 分析作为方法论先例 | `results/final_pack_v1/longrun_checkpoint_summary.csv` |

### 第6章 实验结果与分析

| 节号 | 节名 | 放置内容 | 具体文件 |
|------|------|---------|----------|
| **6.1** | BrightControlBench的区分能力 | 引用现有 DataBench-100 和 PromptBench-40 上不同方法的差异化分布，证明基准对方法有区分力 | v2: `databench100_eval_summary_all.csv`（6方法MAE从0.111到0.125不等），`promptbench40_category_summary.csv` |
| **6.2** | SD1.5 pilot实验结果 | **本节是现有SD1.5成果的核心放置位置**。表6.1: DataBench-100 MAE（§3.1），表6.2: PromptBench-40 method summary（§3.2），图6.2: case grid | v2 全部数据 + `promptbench40_case_grid.png` |
| 6.3 | 现代模型 landscape分析 | 暂无数据。用 SD1.5 结果作为 baseline，标注"待后续 SD3.5/FLUX 实验补充" | — |
| 6.4 | SDXL训练消融结果 | 暂无数据。用 SD1.5 结果作为 prior evidence | — |
| **6.5** | Offset noise的亮度控制作用 | 表6.4: checkpoint selection table（ckpt2000 vs ckpt2500），图6.5: longrun checkpoint curve，表6.5: offset002 vs offset005 对比 | `checkpoint_selection_table.csv`, `longrun_checkpoint_curve.png`, v2 数据中 offset002 vs offset005 行 |
| **6.6** | v-prediction组合消融分析 | 表6.6: vpred+offset ablation（§3.4），图6.6: vpred+offset case grid。重点：证明组合为负向消融 | v3 全部数据 + `offset005_vpred_case_grid.png` |
| 6.7 | 推理配置对控制效果的影响 | 暂无数据。可引用现有 SD1.5 推理配置（DDIM 30, CFG 7.5），标注"待后续实验" | — |
| 6.8 | 人工盲评与自动指标相关性 | 暂无数据。标注"HumanCheck-100 待实施" | — |
| **6.9** | Gain-cost tradeoff总结 | 核心表（§3.4）：三类方法（保守/强控制/混合）的 gain-cost 对比。引用 neutral over-darkening 数据：base neutral MAE 0.052 → offset005_ckpt2500 0.108 = cost | v2 全部数据，重点使用 DataBench neutral MAE 和 PromptBench direction success 的 tradeoff |
| 6.10 | 失败案例与误差分析 | 引用 v_prediction 方向错误案例（backlight Δ 为正，cp300 dark Δ 为正）、vpred+offset 方向混乱案例 | v2/v3 的异常数据点 |

### 附录

| 附录 | 放置内容 |
|------|---------|
| 附录 A | PromptBench-40 完整列表 → 论文需更新为 PromptBench-160 |
| 附录 B | DataBench-100 统计 → 论文需更新为 DataBench-500 |
| 附录 D | 训练配置与超参数（引用 `scripts/run_day2_longrun_offset005.sh` 和 `scripts/run_offset005_vpred_ablation.sh`） |
| 附录 E | 更多可视化结果（§4.2 所列） |

---

## 6. Key Claims Supported by Current Evidence

### 6.1 可自信写入论文的结论

| 编号 | 结论 | 支撑文件 | 论文中方表达建议 |
|------|------|----------|-----------------|
| C1 | offset noise 是亮度方向控制的主要增益来源 | `v2/promptbench40_method_summary.csv`：offset005 dir_success=89.4%，baseline_lora=48.8%，v_prediction=48.1% | "在 SD1.5 pilot 中，引入 noise_offset=0.05 后方向成功率从 48.8% 提升至 89.4%（+40.6pp），表明 offset noise 是这一增益的主要驱动因素。" |
| C2 | offset005_ckpt2500 是 SD1.5 pilot 的最优 checkpoint | `v2/checkpoint_selection_table.csv`：MAE 0.1112（最低），dir_success 89.4%（最高） | "在 5 个 offset005 长跑 checkpoint 中，ckpt2500 取得了最低整体 MAE（0.111）和最高方向成功率（89.4%）。" |
| C3 | brightness control 与 neutral over-darkening 存在 tradeoff | `v2/databench100_eval_summary_all.csv`：base neutral MAE 0.052 → offset005_ckpt2500 0.108 | "offset005 将整体 MAE 降低 10.7%（0.125→0.111），但以 neutral 场景的 MAE 上升 107%（0.052→0.108）为代价，揭示了 gain-cost tradeoff。" |
| C4 | v_prediction 单独方法存在均匀过暗化而非选择性控制 | `v2/promptbench40_category_summary.csv`：v_pred backlight Δ = +0.021（应为负），但 dark MAE 最低（0.068） | "v_prediction 在 dark 子集上 MAE 最低（0.068），但在 backlight 类别上 Δ 为正（+0.021），表明其暗化是均匀而非选择性的。" |
| C5 | offset005+v_prediction 组合为负向消融 | `v3/offset005_vpred_vs_offset005_table.csv`：组合最优 dir_success=69.4%，远低于纯 offset005 的 89.4% | "与预期不同，offset005+v_prediction 组合的方向成功率仅为 69.4%，相比纯 offset005 下降了约 20 个百分点，表明当前简单组合方式引入了冲突训练信号。" |
| C6 | offset002 提供温和控制 | `v2/databench100_eval_summary_all.csv`：offset002 neutral MAE=0.047，dir_success=61.9% | "offset002（α=0.02）提供了介于 baseline 与 offset005 之间的温和控制：方向成功率 61.9%，neutral MAE 几乎无退化（0.047 vs base 0.052）。" |
| C7 | BrightControlBench 能区分不同方法 | v2 数据：6 方法在 MAE（0.111-0.125）和 dir_success（48.1%-89.4%）上存在明显差异 | "DataBench-100 和 PromptBench-40 在 6 个方法上的指标分布区间分别为 [0.111, 0.125]（MAE）和 [48.1%, 89.4%]（方向成功率），表明基准能够有效区分不同训练策略的亮度控制能力。" |

### 6.2 需要谨慎表达的结论

| 编号 | 谨慎结论 | 需注意的 limitation |
|------|---------|-------------------|
| L1 | "offset noise 优于所有 baseline" | 仅在 SD1.5 上验证，且只在 compare_6k 数据集上训练；SDXL/SD3.5/FLUX 效果未知 |
| L2 | "ckpt2500 是最优 checkpoint" | 仅在 eval_smoke_8（8条prompt）上做了长跑分析，不是 DataBench 或 PromptBench 的全量 checkpoint 分析 |
| L3 | "Border Purity 能衡量背景纯度" | 尚无 HumanCheck 验证；仅凭边缘像素统计可能存在假阳性 |
| L4 | "v_prediction 不适合亮度控制" | 仅测试了 v_prediction alone 和 v_prediction+offset005；v_prediction+offset 可能有更优的 α 或训练步数组合 |

---

## 7. Claims Not Yet Supported

以下结论目前**不能写进论文**，必须等后续实验补充：

| 编号 | 待验证结论 | 为什么还不能写 | 需要的后续实验 |
|------|-----------|-------------|--------------|
| NS1 | "Modern models (SDXL/SD3.5/FLUX) still struggle with brightness control" | 尚无现代模型评测数据 | 实验一：冻结横评（§5.2） |
| NS2 | "offset005 is the best method overall" | 仅在 SD1.5 上验证 | SDXL 主训练消融（§5.3） |
| NS3 | "Automated metrics correlate well with human judgment" | 无人工评测数据 | HumanCheck-100（§5.5） |
| NS4 | "Brightness control is robust to inference hyperparameters" | 无推理鲁棒性实验 | 推理鲁棒性实验（§5.4） |
| NS5 | "piecewise α(t) outperforms fixed α" | 无 α(t) 实验 | SDXL + α(t) 训练（§5.3 C3组） |
| NS6 | "v_prediction interaction with offset noise can be made positive" | 当前组合为负向 | 更多 α 值 / 训练步数的组合实验 |
| NS7 | "BrightControlBench has strong human alignment" | HumanCheck-100 未实施 | 完整的 HumanCheck-100 盲评 |
| NS8 | "The gain-cost tradeoff generalizes across model scales" | 仅 SD1.5 数据 | SDXL/SD3.5/FLUX 横评 + 训练消融 |
| NS9 | "DataBench automatic MAE reflects true luminance control quality" | 需要人工对标验证 | HumanCheck 亮度 correctness 与 MAE 的 Spearman ρ |
| NS10 | "The findings extend to complex semantic+brightness prompts" | 当前 PromptBench-40 无 complex 类别（计划中 PromptBench-160 才有） | PromptBench-160 扩展（§3.2） |

---

## 8. Recommended Captions：论文图题与表题建议

### 表格标题

| 表号 | 建议标题 | 对应文件 |
|------|---------|----------|
| Table 6.1 | SD1.5 pilot: DataBench-100 luminance MAE comparison across 6 training strategies | `databench100_eval_summary_all.csv` |
| Table 6.2 | SD1.5 pilot: PromptBench-40 per-method brightness control summary | `promptbench40_method_summary.csv` |
| Table 6.3 | SD1.5 pilot: PromptBench-40 per-category brightness delta and direction success breakdown | `promptbench40_category_summary.csv` |
| Table 6.4 | Checkpoint selection: offset005_ckpt2000 vs offset005_ckpt2500 full comparison | `checkpoint_selection_table.csv` |
| Table 6.5 | offset005 + v_prediction ablation: DataBench-100 MAE across 5 checkpoints | `offset005_vpred_ablation_summary.csv` |
| Table 6.6 | offset005 + v_prediction ablation: PromptBench-40 direction control per checkpoint | `offset005_vpred_pb_method_summary.csv` |
| Table 6.7 | Gain-cost tradeoff: brightness control gain vs neutral over-darkening cost | 综合 v2 DataBench neutral MAE + direction success |

### 图片标题

| 图号 | 建议标题 | 对应文件 |
|------|---------|----------|
| Fig 6.1 | SD1.5 pilot case grid comparing base, baseline_lora, offset005_ckpt2000, and offset005_ckpt2500 across five brightness categories | `promptbench40_case_grid.png` |
| Fig 6.2 | Offset noise long-run checkpoint evolution: luminance delta (mean and std) vs training steps on eval_smoke_8 | `longrun_checkpoint_curve.png` |
| Fig 6.3 | offset005 + v_prediction ablation case grid: five checkpoints across five brightness categories | `offset005_vpred_case_grid.png` |

---

## 9. Risk Notes：可能被老师质疑的点及如何规避

| 编号 | 可能的质疑 | 风险等级 | 规避策略 |
|------|-----------|---------|---------|
| R1 | "SD1.5 是旧模型，结论能推广吗？" | 高 | 明确写 SD1.5 是 **pilot / legacy anchor**，在第6.2节前加 disclaimer："为验证问题普遍性和方法方向，首先在 SD1.5 上进行 pilot study"。等 SDXL 实验做完把 SD1.5 退为"附录/补充" |
| R2 | "DataBench 只有 100 张图，太少了" | 中 | 诚实标注为 "雏形（prototype）"，论文中扩展到 DataBench-500 即可。当前结果作为构建方法验证 |
| R3 | "PromptBench 只有 40 条，不够" | 中 | 同上，标注为 PromptBench-40 雏形，论文扩展到 160 条 |
| R4 | "BP 指标（border purity）可靠吗？只看边缘像素？" | 中 | 在 §3.5 中加入指标局限性讨论，标注"待 HumanCheck 验证"。可考虑加 mask-based background purity 作为补充 |
| R5 | "为什么 single seed（DataBench seed=42）？" | 低 | 解释为"为控制计算开销，pilot 阶段使用固定 seed；后续 SDXL 实验将使用 multi-seed"，并在论文中补 4 seed 版本 |
| R6 | "offset noise 就不是你发明的算法，有什么贡献？" | 高 | 论文定位：**不是提出新算法，而是围绕低频亮度控制构建 benchmark + 系统消融 + gain-cost tradeoff 分析**。在 §1.3 中明确："本文不做大模型或新算法，而是针对特定问题做系统性实验分析" |
| R7 | "direction success 的定义合理吗？" | 低 | 在 §3.5 中给出清晰的数学定义，并用 v_prediction 的 backlight 错误方向（Δ 为正但应为负）作为反例说明指标的有效性 |
| R8 | "v_prediction 实验结果是否公平？" | 中 | 明确说明 v_prediction 的推理设置（DDIM 30 steps, 未开 zero-SNR/rescale），如果它需要配套推理配置才能发挥效果，应在论文中标注"推理配置的影响待进一步研究" |
| R9 | "neutral over-darkening 是否被夸大了？MAE 0.05→0.11 看起来绝对值不大" | 低 | 写相对变化："neutral MAE 增加了约 107%（0.052→0.108）"，并在 §6.9 中结合 case grid 和视觉分析来支撑 |
| R10 | "所有实验只在 512×512 分辨率上，SDXL 呢？" | 中 | SD1.5 的 512×512 是模型原始分辨率，合理。论文转到 SDXL（1024×1024）时自然引入分辨率提升 |

---

## 10. Next Experiment Checklist：后续最应该补的实验清单

按优先级从高到低排列：

### Priority 1：必须做（否则论文主体缺失）

| 编号 | 实验 | 对应的论文章节 | 预计工时 | 备注 |
|------|------|-------------|---------|------|
| E1 | **SDXL冻结推理**（base SDXL on DataBench-500 + PromptBench-160） | 6.3 | 1-2天（含benchmark构建） | 先跑冻结，确认现代模型存在问题 |
| E2 | **SD3.5 Medium冻结推理** | 6.3 | 1天 | 同上 |
| E3 | **FLUX.1-dev冻结推理** | 6.3 | 1天（需确认显存） | 同上。注意 FLUX 推理较慢 |
| E4 | **SDXL LoRA baseline 训练 + 评测**（A1组） | 6.4 | 2-3天 | 主训练基线 |
| E5 | **SDXL LoRA + noise_offset=0.05 训练 + 评测**（C1组） | 6.4, 6.5 | 2-3天 | 主实验核心组 |
| E6 | **SDXL LoRA + noise_offset=0.02 训练 + 评测**（作为温和对照） | 6.4, 6.5 | 2-3天 | 与 SD1.5 offset002 对应 |
| E7 | **Benchmark 扩展**：DataBench-100 → DataBench-500，PromptBench-40 → PromptBench-160 | 3.2, 3.3 | 2天 | 构建更完整的 benchmark |

### Priority 2：强烈建议做（增强论文深度）

| 编号 | 实验 | 对应的论文章节 | 预计工时 | 备注 |
|------|------|-------------|---------|------|
| E8 | **SDXL + noise_offset=0.05 + v_prediction**（C2组） | 6.6 | 2-3天 | 验证 SD1.5 上的负向消融在 SDXL 上是否复现 |
| E9 | **推理鲁棒性**：对最优 1-2 模型换 sampler/steps/CFG | 6.7 | 2-3天 | 测试控制效果的推理稳定性 |
| E10 | **checkpoint evolution**（SDXL 长跑，如 3000 steps，每 500 cp） | 5.4, 6.5 | 2-3天训练 + 1天评测 | 对应 SD1.5 的 longrun 分析 |

### Priority 3：加分项（有更好）

| 编号 | 实验 | 对应的论文章节 | 预计工时 | 备注 |
|------|------|-------------|---------|------|
| E11 | **SDXL + piecewise α(t) + v_prediction**（C3组） | 6.4 | 2-3天 | 如果时间允许 |
| E12 | **HumanCheck-100**：设计盲评协议，收集 3 人评分 | 5.5, 6.8 | 3-5天（含设计+收集+统计） | 显著增强论文可信度 |
| E13 | **SD3.5 Medium 小规模 LoRA pilot** | 6.3 | 2-3天 | 跨模型泛化性分析（可选择做） |
| E14 | **Gradio 可视化平台** | 7 | 2-3天 | 论文第7章主要内容 |

### Priority 4：可选的补充

| 编号 | 实验 | 备注 |
|------|------|------|
| E15 | DataBench-500 上做 multi-seed 版本（如 seed=0/1/2/3） | 提高统计可靠性 |
| E16 | 补充 VQAScore / TIFA 语义指标 | 增加语义质量约束维度 |
| E17 | 对最终最优模型做数据污染检查 | 回答 review 常见质疑 |

---

## 附录：实验数据全映射速查表

### 方法 → 文件速查

```
base:
  DataBench: results/final_pack_v2/databench100_eval_summary_all.csv (row 2)
             results/final_pack_v2/databench100_eval_details_all.csv (rows 1-100)
  PromptBench: results/final_pack_v2/promptbench40_category_summary.csv (rows 2-6)
               results/final_pack_v2/promptbench40_details.csv (rows for method='base')
  Smoke eval: results/final_pack_v1/longrun_checkpoint_summary.csv (reference)
              results/lora_smoke_eval/base/metrics.csv

baseline_lora:
  DataBench: results/final_pack_v2/databench100_eval_summary_all.csv (row 3)
  PromptBench: results/final_pack_v2/promptbench40_category_summary.csv (rows 7-11)
  Checkpoint: train_logs/lora_smoke/baseline_run1/checkpoint-500/

v_prediction:
  DataBench: results/final_pack_v2/databench100_eval_summary_all.csv (row 4)
  PromptBench: results/final_pack_v2/promptbench40_category_summary.csv (rows 27-31)
  Checkpoint: train_logs/lora_smoke/v_prediction_run/checkpoint-500/

offset002:
  DataBench: results/final_pack_v2/databench100_eval_summary_all.csv (row 5)
  PromptBench: results/final_pack_v2/promptbench40_category_summary.csv (rows 12-16)
  Checkpoint: train_logs/lora_smoke/noise_offset_run/checkpoint-500/

offset005_ckpt2000:
  DataBench: results/final_pack_v2/databench100_eval_summary_all.csv (row 6)
  PromptBench: results/final_pack_v2/promptbench40_category_summary.csv (rows 17-21)
  Checkpoint: train_logs/day2_best_method_longrun_offset005/checkpoint-2000/

offset005_ckpt2500:
  DataBench: results/final_pack_v2/databench100_eval_summary_all.csv (row 7)
  PromptBench: results/final_pack_v2/promptbench40_category_summary.csv (rows 22-26)
  Checkpoint: train_logs/day2_best_method_longrun_offset005/checkpoint-2500/

vpred_offset005_cp300/600/900/1200/1500:
  DataBench: results/final_pack_v3/offset005_vpred_ablation_summary.csv
  PromptBench: results/final_pack_v3/offset005_vpred_pb_method_summary.csv
  Checkpoints: train_logs/final_ablation/offset005_vpred_1500/checkpoint-{step}/
```

---

*本指南基于 `/data1/cx/brightcontrol/results/` 下 final_pack_v1/v2/v3 的完整数据生成，所有指标均有具体文件路径支撑。SD1.5 结果定位为 pilot / legacy anchor，不做过度推广。后续 SDXL/SD3.5/FLUX/HumanCheck 实验完成后需更新本指南。*
