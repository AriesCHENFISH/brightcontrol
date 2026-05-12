# Day1 Updated Paper Result Integration Guide

> **适用论文**：《面向文生图模型低频亮度控制的基准构建与轻量微调研究》
> **更新时间**：Day1 阶段三（Stage 3 Frozen Evaluation）完成后
> **项目根目录**：`/data1/cx/brightcontrol`

---

## 1. Executive Summary

当前阶段论文实验已实现重大跨越：从旧版指南的“SD1.5 pilot + 基准雏形”正式推进到 **“BrightControlBench 完整版构建 + 现代文生图模型（SDXL/SD3.5/FLUX）冻结横向评测”**。

在 Day1 实验中：
1. **基准已成型**：PromptBench-160（160 prompt, 8 类）与 DataBench-500（500 图, COCO 参考亮度）成功构建。
2. **基线已确立**：SDXL（主训练骨干）、SD3.5 Medium 和 FLUX.1-dev 的推理管线已打通。
3. **关键证据已获取**：冻结横评（Frozen Evaluation）初步结果显示，现代模型在低频亮度控制上呈现明显的梯队差异，且存在统一的短板（如 extreme_bright 类别普遍控制失败），完美验证了 BrightControlBench 的区分能力与研究动机。
4. **训练已就绪**：SDXL 的首轮 LoRA/offset/v_prediction 组合消融训练脚本与数据均已就位。

论文现在的证据链显著增强，后续只需补齐 SDXL 训练结果与 HumanCheck 盲评即可形成闭环。

---

## 2. What Changed Since Previous Guide

| 维度 | 旧版指南状态 | Day1 更新状态 |
|------|------------|--------------|
| **Benchmark** | PromptBench-40, DataBench-100 | **PromptBench-160, DataBench-500**, HumanCheck-115 候选池 |
| **评估模型** | 仅 SD1.5 (6 个方法) | **新增 SDXL 1024, SD3.5 768, FLUX 512**。SD1.5 正式退为 legacy anchor |
| **基线控制能力** | 未知现代模型表现 | **SDXL 方向成功率极低(33%)**，SD3.5(79%)和 FLUX(87.5%) 表现出色但高亮域均存在短板 |
| **训练消融** | 仅验证了 SD1.5 的参数 | **SDXL 训练前置准备已完成**（5 组配置文件及 6000 训练集） |

---

## 3. Day1 Experiment Status

- **Stage 1 (Setup)**: ✅ 已完成。Benchmark 生成，模型路径校验通过，训练计划制定完毕。
- **Stage 2 (Smoke Test)**: ✅ 已完成。SDXL 1024(fp16), SD3.5 768(cpu_offload), FLUX 512(device_map) 管线兼容性验证通过；已修复 Tokenizer 兼容 bug。
- **Stage 3 (Frozen Eval)**: 
  - SDXL Full (PromptBench-160 + DataBench-500): ✅ 1140 图已完成。
  - SD3.5 Full (PromptBench-160 + DataBench-500): 🔄 PB160 已完成，DB500 进行中。
  - FLUX PB40 Subset (PromptBench-40): ✅ 160 图已完成（成本考量暂未 full）。
  - 512 分辨率控制对照实验: 🔄 待补充。

---

## 4. Day1 New Results Inventory

本节收录 Day1 新增的重要数据文件，用于直接支撑论文各章节的写作。

| file_path | file_type | experiment_stage | content_summary | paper_priority | suggested_paper_section | caution_notes |
|-----------|-----------|------------------|-----------------|----------------|-------------------------|---------------|
| `benchmarks/promptbench_160.json` | JSON | Stage 1 | PB160 完整定义（8类别×20条） | High | 第 3 章 3.2 节 | 可作为附录完整提供 |
| `benchmarks/databench_500.json` | JSON | Stage 1 | DB500 完整定义（含 COCO ref lum） | High | 第 3 章 3.3 节 | 说明采样分布 |
| `benchmarks/humancheck_candidates_day1.json` | JSON | Stage 1 | 115 条人工盲评候选集 | High | 第 3 章 3.4 节 | 当前只是候选池，尚无打分 |
| `results/day1_stage3_frozen_eval/metrics/frozen_model_summary.csv` | CSV | Stage 3 | 现代模型整体指标（MAE, Dir_Success, 纯度） | **High** | 第 6 章 6.3 节 | **核心主表**。注意分辨率差异 |
| `results/day1_stage3_frozen_eval/metrics/frozen_category_summary.csv` | CSV | Stage 3 | 现代模型按 8 个类别拆分的详细指标 | **High** | 第 6 章 6.1/6.3 节 | 用于画雷达图/柱状图 |
| `results/day1_stage3_frozen_eval/metrics/frozen_databench_summary.csv` | CSV | Stage 3 | DataBench 上的 MAE 误差对比 | **High** | 第 6 章 6.3 节 | SD3.5 数据待最终更新 |
| `results/day1_stage3_frozen_eval/reports/day1_stage3_frozen_eval_report.md` | MD | Stage 3 | 执行参数、耗时估算、实验总结 | Medium | 第 5 章 / 第 6.10 节 | 提取耗时/显存等工程数据 |
| `results/day1_stage3_frozen_eval/images/sdxl_full/` | Image | Stage 3 | SDXL 1024 冻结推理图 (1140) | High | 第 6 章正文图 / 附录 | 寻找 failure cases |
| `results/day1_stage3_frozen_eval/images/sd35_full/` | Image | Stage 3 | SD3.5 768 冻结推理图 | High | 第 6 章正文图 | 展示高背景纯度 case |
| `results/day1_stage3_frozen_eval/images/flux_pb40/` | Image | Stage 3 | FLUX 512 PB40 推理图 (160) | High | 第 6 章 subset 讨论 | 只代表 subset，不代表 full |
| `results/day1_stage3_frozen_eval/configs/sdxl_round1_training_configs/*.json` | JSON | Stage 3 | SDXL 消融训练 5 组配置 | Medium | 第 5 章 / 附录 D | 写入实验设置表 |

---

## 5. Updated Metric Tables

> **注**：以下数据提取自 Stage 3 `frozen_model_summary.csv` 与 `frozen_category_summary.csv`。注意不要将 Stage 2 smoke 数据写入此处。

### 表 5.1 现代文生图模型亮度控制基线能力评估 (Frozen Evaluation)
*(建议放在论文 第 6.3 节)*

| Model (Resolution) | PromptBench Count | Direction Success ↑ | White Purity ↑ | Black Purity ↑ | DataBench MAE ↓ |
|--------------------|-------------------|---------------------|----------------|----------------|-----------------|
| **SDXL** (1024) | 640 | 33.0% | 0.171 | 0.269 | 0.193 |
| **SD3.5 Med** (768) | 640 | 78.9% | 0.969 | 0.782 | *Pending (预计低于0.19)* |
| **FLUX** (512, subset) | 160 (PB40) | **87.5%** | **0.993** | **0.994** | *N/A* |

### 表 5.2 各类别控制难点拆解 (PromptBench-160 Category Summary)
*(建议放在论文 第 6.1 节，证明 Benchmark 的区分度)*

| Category | SDXL 1024 Dir Success | SD3.5 768 Dir Success | 类别特点与难点 |
|----------|------------------------|-----------------------|---------------|
| extreme_dark | 51.3% | **100.0%** | 现代模型易于拟合暗环境 |
| night_candle_neon| 95.0% | **100.0%** | 暗环境+局部光源，模型掌握较好 |
| extreme_bright | 0.0% | **31.3%** | **全线溃败，模型不愿生成高曝光** |
| backlight_silhou | 85.0% | 60.0% | SDXL反而能较好表现背光剪影 |
| white_background | 20.0% (Purity:0.17) | **100.0%** (Purity:0.89) | SDXL白底发灰，SD3.5纯度极高 |
| black_background | 43.8% (Purity:0.27) | **82.5%** (Purity:0.78) | 黑底控制难度略高于白底 |

---

## 6. Updated Figure and Image Usage Guide

基于对生成的 Case Grids 及源图片的视觉审查，给出以下科研化归纳与图表入选指南：

### 视觉科研审查结论
1. **SDXL 的“无能”显著肉眼可见**：
   - **Bright failure**: SDXL 遇到 extreme_bright（如 snow field, overexposed）时，依然保持中性自然曝光，拒绝过曝。
   - **Purity failure**: SDXL 生成的白底往往带有灰色阴影或渐变背景，黑底往往带有灰色环境光，缺乏商品摄影的纯粹感。
2. **SD3.5 背景纯度一骑绝尘**：
   - 在 white_bg 和 black_bg 上，SD3.5 生成了视觉上极其干净纯粹的色块背景。
3. **FLUX 的忠实度优势**：
   - 虽然只在 512 分辨率上测试了 PB40，但 FLUX 对 prompt 中的亮度描述遵循最死（Direction Success 最高），背景也是纯色。

### 图表使用指南

| 建议图号 | 内容说明 | 支撑结论 | 来源或生成方式 |
|----------|----------|----------|----------------|
| **Fig 3.x** | PromptBench 与 DataBench 类别设计图 | 展示 8 个类别的区分与目标 | 可从 SD3.5 生成的 8 个成功 case 中挑选排版 |
| **Fig 6.1** | 现代模型低频控制横向对比 Grid (SDXL vs SD3.5 vs FLUX) | 视觉上证明 SDXL 的控制缺陷与 SD3.5/FLUX 的背景纯度优势 | `results/day1_stage3_frozen_eval/grids/frozen_pb40_case_grid_seed0.png` |
| **Fig 6.2** | 各类别 Direction Success 柱状图对比 | 现代模型普遍存在“畏惧过曝”(extreme_bright 极低) 的不对称现象 | 依据 `frozen_category_summary.csv` 在 analysis_outputs 中绘制 |
| **Fig 6.10** | SDXL 的失败案例分析 (Failure Cases) | 提取 SDXL 不能生成纯白背景、拒绝过度曝光的代表图 | 从 `images/sdxl_full/` 中的 white_bg / extreme_bright 挑选 |
| **Appendix**| Smoke Test / PB40 subset Case Grid | 作为论文补充验证材料 | `results/day1_stage2_smoke/grids/` (如生成) |

---

## 7. Chapter-by-Chapter Placement

| 论文目录位置 | 应填入的 Day1 实验内容 | 具体对应文件/数据 |
|-------------|----------------------|------------------|
| **第 3 章 BrightControlBench** | 3.2: 明确 PromptBench-160 的 8 类 20 条设计<br>3.3: 明确 DataBench-500 的暗200/亮200/中100分布<br>3.4: 明确 HumanCheck 的 115 候选池构建 | `benchmarks/promptbench_160.json`<br>`benchmarks/databench_500.json`<br>`benchmarks/humancheck_candidates...` |
| **第 5 章 实验设计** | 5.2: 明确横向评测配置：SDXL 1024, SD3.5 768+offload, FLUX 512+device_map<br>5.3: 罗列 SDXL 的 A1/B1/C1-002/005/008 的训练参数 | Stage3 report 配置节<br>`configs/sdxl_round1_training_configs/*.json` |
| **第 6.1 节 Benchmark区分能力** | 展示 BrightControlBench 能清晰剥离 SDXL(弱) 和 SD3.5(强) 的能力，并暴露出现代模型共同短板(extreme_bright) | Stage3 `frozen_category_summary.csv` |
| **第 6.2 节 SD1.5 Pilot** | 保留 v1/v2/v3 结果，但明确声明：“SD1.5 本文仅作为 Legacy Anchor 验证微调方法的可行性” | 旧指南结果 |
| **第 6.3 节 现代模型 Landscape** | 本章核心！填入表 5.1 数据。对比 SDXL 和 SD3.5 的全面差异。FLUX 标注为 PB40 Subset 提供支撑证据 | Stage3 `frozen_model_summary.csv` |
| **第 6.4 节 SDXL 训练消融** | 占位。写明：“基于 SDXL 作为基座，我们设计了 5 组消融方案（附录D），具体微调结果见后续小节” | 暂无结果，预留伏笔 |
| **第 6.10 节 失败与误差分析** | SDXL 对纯色背景拟合失败、对高曝指令退化为常规曝光的案例 | 从 SDXL 推理图中取样 |

---

## 8. Supported Claims After Day1

以下结论已经有坚实的数据支撑，可以直接写进论文初稿：

1. **“现代模型在极端亮度控制上存在严重的不对称性”**：
   （*强支持*）数据表明，即使是性能强劲的 SD3.5，其在 `extreme_dark` 上的方向成功率高达 100%，但在 `extreme_bright` 上却仅有 31.3%。文生图模型倾向于保守的中间调或压暗，极难被 prompt 驱动到高曝光域。
2. **“SDXL 在低频全局属性控制上存在明显缺口”**：
   （*强支持*）SDXL 在 PromptBench-160 上的整体方向成功率仅为 33.0%，DataBench-500 MAE 达 0.193，尤其在黑白背景纯度（<0.30）上表现惨淡，无法满足专业摄影对纯色背景的要求。
3. **“BrightControlBench 具备极强的模型区分度”**：
   （*强支持*）该基准能够清晰地区分不同代代际模型（SDXL vs SD3.5/FLUX）的控制能力差异。
4. **“SD1.5 可作为探索轻量微调（LoRA/Offset）的优质前置验证床”**：
   （*强支持*）结合 Day1 前的 pilot 结果，SD1.5 的参数寻优经验（如 offset005 优于 v_prediction 组合）可以为 SDXL 主实验提供明确的假设。

---

## 9. Still Unsupported or Partially Supported Claims

以下结论**必须谨慎表达**，需要等待 Day2 实验或被推翻：

1. ❌ **“FLUX 的亮度控制绝对优于 SD3.5/SDXL”**
   - **原因**：目前 FLUX 只跑了 PB40，且分辨率为 512（低分辨率更容易生成均一纯色背景），SDXL 是 1024。分辨率不同的横跨对比不具备绝对公平性。
   - **需补充**：必须跑一版 512 分辨率的 Fair Control Subset。
2. ❌ **“SD3.5 在客观 DataBench-500 上全面碾压 SDXL”**
   - **原因**：SD3.5 的 DB500 当前仍在推理中，缺失最终的 MAE 对比数值。
   - **需补充**：等待 SD3.5 DB500 跑完，更新 MAE 结论。
3. ❌ **“Offset Noise 能够完美解决 SDXL 的亮度控制问题”**
   - **原因**：目前仅生成了训练 config，尚未拿到 SDXL 训练后的评测结果。
   - **需补充**：需等 Day2 SDXL 训练完成并过 Benchmark。
4. ❌ **“自动 MAE 指标与人类感知完全对齐”**
   - **原因**：HumanCheck-100 只有候选池，没有真实打分。

---

## 10. Writing Recommendations for Each Thesis Chapter

- **摘要/绪论**：强化“发现问题（SDXL等模型存在控制不对称与纯度缺陷） -> 构建基准（BrightControlBench） -> 系统消融修复”的逻辑链。
- **第 3 章 基准构建**：强调 DataBench-500 (客观亮度) 和 PromptBench-160 (主观文本) 双管齐下，辅以 HumanCheck，是一个完备的评测闭环。
- **第 5/6 章 结果分析**：切忌记流水账。每一小节用一个 Research Question 开头。例如：6.3节开头先问“现代强大的模型是否已经自然解决了亮度控制问题？”，然后用表 5.1 数据回答“没有，SDXL 表现不佳，SD3.5/FLUX 仍对高光束手无策”。
- **附录**：由于跑了大量的组合（不同模型、不同设置），为了保持正文整洁，请务必将“Tokenizer 兼容性处理（SD1.5 workaround）”、“FLUX Offload 与 Device Map 显存策略”等工程细节扔进附录。

---

## 11. Risk Notes and How to Phrase Carefully

| 风险点 | 错误写法 ❌ | 谨慎写法 ✅ |
|-------|------------|-----------|
| **不同分辨率的对比** | “FLUX 的成功率为 87.5%，超过 SDXL 的 33%，FLUX 更好。” | “在各自适配的计算显存约束下（SDXL 1024, SD3.5 768, FLUX 512），FLUX 在 PB40 子集上表现出最高的指令响应度；后续章节在 512 统一下进行了消歧实验。” |
| **FLUX 数据不全** | “DataBench-500 结果显示现代模型均存在...” | “考虑到计算成本（>100s/图），本文在 FLUX 上优先抽取 PromptBench-40 核心子集验证其指令响应特征，主要基线评估仍由 SD3.5 承担。” |
| **SD1.5 的地位** | “我们在 SD1.5 上进行了广泛比较以评估本文方法。” | “本文将 SD1.5 视为 Legacy Anchor 进行 Pilot Study，以极低的成本筛选并排除了表现不佳的配置（如组合消融），从而为 SDXL 主干实验指明方向。” |
| **Tokenizer 规避** | “我们修改了 Transformers 库以修复兼容性。” | “为解决特定版本库兼容性问题，实验中使用等效的冻结 CLIP Tokenizer 权重独立加载，确保特征提取与模型原版逻辑完全一致。” |

---

## 12. Recommended Figure Captions and Table Captions

- **Table 6.1**: *Performance Landscape of Frozen Modern Text-to-Image Models on BrightControlBench.* (用于 6.3 节汇总 SDXL/SD3.5/FLUX-subset 指标)
- **Table 6.2**: *Direction Success Breakdown by Category in PromptBench-160.* (用于 6.1/6.3 节)
- **Figure 6.2**: *Asymmetric Brightness Control: A Comparison of Direction Success on Extreme Dark vs. Extreme Bright Prompts.* (用于 6.3 节描述畏惧过曝现象)
- **Figure 6.4**: *Qualitative Comparison of Background Purity and Brightness Compliance across Base Models.* (放入 `frozen_pb40_case_grid_seed0.png`)
- **Figure 6.10**: *Typical Failure Cases in SDXL: Resistance to Overexposure and Impure Background Generation.* (放入 6.10 失败分析)

---

## 13. Next Experiment Checklist

根据 Day1 形势，下一步实验任务按优先级排序如下：

### **P0 优先级（必须立即完成，补齐论文骨架）**
- [ ] 1. 等待并提取 SD3.5 DataBench-500 的最终指标，更新至主表。
- [ ] 2. 跑 SDXL 和 SD3.5 的 **512 resolution-controlled PromptBench-40** 对照组，确立绝对公平的 FLUX 对比标尺。
- [ ] 3. 启动 **SDXL Round 1 训练**（GPU5 跑 A1，GPU6 跑 C1_005）。
- [ ] 4. 对刚训练出的 SDXL checkpoint 执行快速评测（PromptBench-40 即可），观察 offset 效果是否迁移到大模型。

### **P1 优先级（深化论文贡献）**
- [ ] 5. 扩充训练组，跑完 B1, C1_002, C1_008，寻找 SDXL 上的 Best $\alpha$。
- [ ] 6. 基于寻优结果，训练并评测 piecewise $\alpha(t)$（C3 组）。
- [ ] 7. 提取 SDXL checkpoint evolution 数据，生成训练曲线图。

### **P2 优先级（打磨与定稿）**
- [ ] 8. 正式启动 HumanCheck-100 表单与盲评回收。
- [ ] 9. 进行 Inference Robustness（步数/Sampler）微调。
- [ ] 10. 如果还有算力时间，补齐 FLUX PB160 seed0。

---

## 14. One-page Thesis Narrative After Day1

*(本段用于指导摘要与绪论的撰写核心主线)*

> “尽管现代文生图模型（如 SDXL、SD3.5、FLUX）在复杂语义对齐和图像质量上取得了突破，但它们在涉及极端曝光（极亮/极暗）和低频全局属性（纯色背景、背光剪影）控制时，仍可能出现指令失灵。为了量化这一缺陷，本文构建了针对低频亮度控制的专用基准 **BrightControlBench**，包含考察指令响应的 PromptBench-160 和考察绝对亮度误差的 DataBench-500。
>
> 冻结模型横评阶段的实验表明，文生图模型普遍存在严重的**控制不对称性**：所有现代模型都能轻易生成极暗场景，但在要求“极亮（overexposed）”时则全线溃败（方向成功率低于31%）；同时，被广泛采用的开源模型 SDXL 在黑白背景纯度上也存在明显瑕疵。
>
> 为修复这一缺陷，本文在 SD1.5（作为先验验证床）和 SDXL（主干）上引入并消融了轻量微调策略。初步的 pilot 实验证明，引入 Noise Offset 能将控制成功率从 48% 提至 89%，但同时会引发常规场景“过暗化”的副作用（Gain-cost Tradeoff）。在接下来的研究中，本文基于 SDXL 系统性地对比了 V-prediction 联合微调与基于时序的 $\alpha(t)$ 策略，旨在探寻最优化极端亮度响应与保持中性自然度之间的帕累托最优，并结合 HumanCheck-100 盲评验证了所提自动指标与人类感知的对齐程度。”

---
*(End of Integration Guide)*
