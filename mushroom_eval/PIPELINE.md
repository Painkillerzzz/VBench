# 蘑菇TUTU视频质量评估 — 全链路技术文档

## 1. 项目概述

**目标：** 对 1951 条首帧引导生成的蘑菇TUTU视频进行全自动质量评估，筛选出含"坏特征"的视频。

**输入：**
- 1951 条 MP4 视频（5秒，720×1280）
- 存储位置：`s3://video-finetune-models-datasets-results/wan-video-training-inference-M12/datasets/v-0331/`
- 本地缓存：`mushroom_data/videos/`

**输出：**
- 每条视频的多维评分 + 二分类标签（good/bad）+ 触发的坏特征列表
- `mushroom_eval_results/classification.json`
- `mushroom_eval_results/good_videos.txt` / `bad_videos.txt`

---

## 2. 坏特征定义

### IP 无关（通用视频质量）
| 编号 | 坏特征 | 说明 |
|------|--------|------|
| F1 | 主体动作缓慢 | 蘑菇TUTU几乎不动 |
| F2 | 出现奇怪/未知物体 | 画面中出现不应存在的畸变物体 |
| F3 | 物理规律混乱 | 物体悬空、错误粘连、穿模 |
| F4 | 帧间跳变 | 相邻帧之间出现不连续的突变 |

### IP 相关（蘑菇TUTU 角色特有）
| 编号 | 坏特征 | 说明 |
|------|--------|------|
| F5 | 动作卡顿僵硬 | 像玩偶而非活泼的小动物 |
| F6 | 不合理的大小突变 | 角色体型突然变大变小 |
| F7 | 运动主体错误 | 生成模型让背景/其他物体运动，蘑菇TUTU僵直不动 |
| F8 | 动作/位移不连贯 | 动作轨迹突然中断或跳跃 |
| F9 | 四肢无法移动/身体僵硬 | 角色全身僵硬，无肢体运动 |
| F10 | 过长时间静止不动 | 大部分时间处于冻结状态 |
| F11 | 衣服/身体时间一致性差 | 外观在帧间发生不自然的变化 |

---

## 3. 评估架构

```
视频 + caption
    │
    ├─── Step 0: Caption 标注 (Gemini Pro) ──────────────────┐
    │                                                         │
    ├─── Tier 1: 信号级指标 (VBench, GPU) ───────────────────┤
    │       ├─ D1: 动态程度 (RAFT 光流)                       │
    │       ├─ D2: 运动平滑度 (AMT 帧插值)                   │
    │       ├─ D3: 时序闪烁 (帧间 MAE, CPU)                  │
    │       ├─ D4: 主体一致性 (DINO 特征相似度)              │
    │       └─ D5: 光流稳定性 (RAFT 加速度/空间方差)         │
    │                                                         │
    ├─── Tier 2: VLM 语义评估 (Gemini API) ─────────────────┤
    │       └─ 9 个 Yes/No 问题 → 综合评分                   │
    │                                                         │
    └─── 融合层: VLM 评分 + Tier1 一票否决 → 二分类标签 ─────┘
```

---

## 4. Step 0: Caption 标注

**目的：** 为每条视频生成高质量中文描述，供 VLM 评估时作为上下文参考。

**方法：**
- 模型：`gemini-3.1-pro-preview`
- Prompt：专业视频标注员角色，要求描述镜头语言、角色动态（头部/五官/肢体）、物理规律、场景细节
- 约束：150-250字，不描述角色物理外观（颜色/材质等），不用猜测性词汇

**实现：** `run_captioner.py` → 100 并发 ThreadPoolExecutor → 输出 `captions.json` + 逐条 `.txt`

**运行结果：** 1951/1951 完成，12 分钟，0 失败

**文件：**
- 代码：`run_captioner.py`, `captioning/prompts.py`
- 输出：`mushroom_data/captions/captions.json`

---

## 5. Tier 1: 信号级指标

### D1: 动态程度 (Dynamic Degree)

**目的：** 检测 F1（主体动作缓慢）、F10（过长时间静止不动）

**方法：**
- 使用 RAFT 光流模型计算相邻帧的光流场
- 取光流幅值最大的前 5% 像素的均值作为帧对运动量
- 输出三个值：
  - `mean_flow`：所有帧对的平均运动量（越大 = 越活跃）
  - `static_ratio`：运动量低于阈值的帧对占比（越大 = 越静止）
  - `is_dynamic`：布尔值，VBench 兼容的二值判断

**复用：** `vbench/dynamic_degree.py` 中的 RAFT 模型，改造为输出连续值

**VLM 审核：** Pearson r = +0.711 (n=20)，强相关 ✓

---

### D2: 运动平滑度 (Motion Smoothness)

**目的：** 检测 F4（帧间跳变）、F5（动作卡顿僵硬）、F8（动作不连贯）

**方法：**
- 使用 AMT (Arbitrary Motion Trajectories) 帧插值模型
- 对相邻帧做中间帧插值，比较插值帧与实际中间帧的 MAE
- MAE 越小 = 运动越平滑（帧间过渡可预测）
- 输出 `motion_smoothness`：归一化到 0-1（越高越好）

**复用：** `vbench/motion_smoothness.py` 的 `MotionSmoothness` 类

**VLM 审核：** Pearson r = +0.584 (n=20)，中等相关 ✓

---

### D3: 时序闪烁 (Temporal Flickering)

**目的：** 检测 F4（帧间跳变）、F11（时间一致性差）

**方法：**
- 计算相邻帧的像素级平均绝对误差 (MAE)
- 不需要模型，纯 CPU 计算
- 高 MAE = 帧间变化大 = 闪烁/跳变
- 输出 `temporal_flickering`：(255 - mean_MAE) / 255，归一化到 0-1（越高越稳定）

**复用：** `vbench/temporal_flickering.py` 的 `cal_score()`

**VLM 审核：** Pearson r = +0.828 (n=20)，高相关 ✓✓

---

### D4: 主体一致性 (Subject Consistency)

**目的：** 检测 F6（不合理的大小突变）、F11（衣服/身体时间一致性差）

**方法：**
- 使用 DINO ViT-B16 提取每帧的视觉特征
- 计算每帧与首帧的余弦相似度 + 相邻帧间的余弦相似度
- 取平均作为视频级一致性分数
- 输出 `subject_consistency`：0-1（越高越一致）

**复用：** `vbench/subject_consistency.py` 的 DINO 特征提取

**VLM 审核：** 本批视频大部分一致性好（VLM 全给 10 分），但 DINO 分数在 0.60-0.99 有区分度，可检测极端低值

**全量结果统计：**
- Mean: 0.9347
- P5: 0.8025
- 低于 0.85: 187 条
- 低于 0.90: 343 条

---

### D5: 光流稳定性 (Flow Stability)

**目的：** 检测 F3（物理规律混乱）、F8（动作/位移不连贯）

**方法：**
- 复用 D1 的 RAFT 光流场，进一步计算：
  - `flow_acceleration`：相邻帧对光流幅值的最大变化量（越大 = 运动越不连贯）
  - `flow_spatial_var`：光流场空间方差均值（越大 = 局部运动越不一致，可能有撕裂/粘连）

**新增模块，基于 RAFT**

**VLM 审核：** Pearson r = +0.712 (n=20)，强相关 ✓

---

## 6. Tier 2: VLM 语义评估

**目的：** 覆盖信号指标无法捕捉的语义级缺陷，特别是 F5（僵硬）、F7（运动主体错误）

**方法：**
- 模型：`gemini-3.1-pro-preview`（视频理解 + yes/no 判断）
- 将视频字节 + caption + 9 个 yes/no 问题一起发给 Gemini
- 每个问题对应一个坏特征维度

**9 个评估问题：**

| 问题 | 对应坏特征 | 方向 |
|------|-----------|------|
| Q1: 是否有明显肢体运动？ | F9 四肢无法移动 | yes=好 |
| Q2: 是否像活泼小动物？ | F5 动作僵硬 | yes=好 |
| Q3: 是否大部分时间静止？ | F10 长时间静止 | yes=坏 |
| Q4: 运动主体是否正确？ | F7 运动主体错误 | yes=好 |
| Q5: 是否出现奇怪物体？ | F2 未知物体 | yes=坏 |
| Q6: 是否符合物理规律？ | F3 物理混乱 | yes=好 |
| Q7: 体型大小是否一致？ | F6 大小突变 | yes=好 |
| Q8: 外观是否一致？ | F11 时间一致性差 | yes=好 |
| Q9: 运动是否连贯？ | F8 动作不连贯 | yes=好 |

**评分规则：**
- 正向问题 yes=1分，no=0分
- 反向问题 yes=0分，no=1分
- 总分 = 得分之和 / 已回答问题数（0-1）

**关键配置：**
- `max_output_tokens=8192`（Pro 模型有内部 thinking chain 约 1000 tokens，需留足空间）
- 100 并发异步请求 + 自适应降速（遇 rate limit 自动减半）
- 断点续传 + 失败自动重试

**实现：** `mushroom_eval/vlm_evaluator.py`

---

## 7. 融合层

**目的：** 将 VLM 评分 + Tier 1 指标合并为最终二分类

**方法：**

### 7.1 VLM 主评分
- 以 VLM 总分的第 P25 分位数作为阈值
- 低于阈值 → bad

### 7.2 Tier 1 一票否决 (Veto Rules)
即使 VLM 评分不低，以下极端值也直接判 bad：
- `subject_consistency < 0.85` → 外观严重漂移
- `static_ratio >= 0.95` → 几乎完全静止
- `flow_acceleration > 阈值` → 运动严重不连贯

### 7.3 坏特征提取
从 VLM 的逐问题回答中提取具体触发的坏特征类型

**实现：** `mushroom_eval/fusion.py`, `mushroom_eval/run_classify.py`

---

## 8. VLM 模型对比

已跑 Flash-Lite 和 Pro-Preview 两个模型的全量评估。

### Flash-Lite (`gemini-3.1-flash-lite-preview`)
- 速度：8.2 条/秒，4 分钟完成全量
- 解析率：100%（9/9 问题全部成功）
- 适合：快速筛选、逐维度打分审核

### Pro-Preview (`gemini-3.1-pro-preview`)
- 速度：2.4 条/秒，~14 分钟完成全量
- 解析率：需设 `max_output_tokens=8192`（Pro 有内部 thinking chain 约 1000 tokens）
- 适合：高精度 yes/no 判断

### 两模型 Q1-Q7 一致率
| 问题 | 一致率 |
|------|--------|
| Q7 大小一致性 | 93.5% |
| Q5 奇怪物体 | 86.4% |
| Q6 物理规律 | 81.1% |
| Q1 肢体运动 | 77.8% |
| Q3 长时间静止 | 76.6% |
| Q2 自然活泼 | 75.9% |
| Q4 运动主体 | 74.9% |

静态特征（大小、物体）一致率高，运动判断差异较大。

---

## 9. Tier 1 可靠性验证

对 20 条随机采样视频，同时计算 Tier 1 数值指标和 VLM（Flash-Lite）逐维度 1-10 打分，计算 Pearson 相关系数。

| Tier 1 指标 | VLM 维度 | Pearson r | 结论 |
|-------------|----------|-----------|------|
| D1 mean_flow | 运动活跃度 | +0.711 | ✓ 强相关，可靠 |
| D2 motion_smoothness | 运动平滑度 | +0.584 | ✓ 中等相关，可用 |
| D3 temporal_flickering | 时序稳定性 | +0.828 | ✓✓ 高相关，非常可靠 |
| D4 subject_consistency | 外观一致性 | NaN | 本批视频区分度不足 |
| D5 -flow_acceleration | 运动连贯性 | +0.712 | ✓ 强相关，可靠 |

---

## 10. 文件结构

```
mushroom_eval/
├── __init__.py
├── config.py              # 配置 + 9 个评估问题定义
├── vlm_evaluator.py       # Gemini 异步批量评估（断点续传 + 自适应并发）
├── tier1_metrics.py       # D1-D5 信号指标（复用 VBench）
├── fusion.py              # 融合决策 + 分类 + 模型对比
├── run_vlm.py             # VLM 评估 CLI
├── run_tier1.py           # Tier 1 评估 CLI
├── run_classify.py        # 分类 CLI
├── verify_tier1.py        # Tier 1 验证 v1（JSON 格式）
├── verify_tier1_v2.py     # Tier 1 验证 v2（逐维度打分）
└── PIPELINE.md            # 本文档

mushroom_eval_results/
├── vlm_results.json                 # Pro-preview 全量 VLM 评分
├── vlm_results_flash_lite.json      # Flash-lite 全量 VLM 评分
├── vlm_comparison.json              # 两模型对比
├── D1_dynamic_degree.json           # D1 全量（运行中）
├── D2_motion_smoothness.json        # D2 全量（运行中）
├── D3_temporal_flickering.json      # D3 全量（已完成）
├── D4_subject_consistency.json      # D4 全量（已完成）
├── D5_flow_stability.json           # D5 全量（运行中）
├── classification.json              # 最终分类结果
├── good_videos.txt                  # 好视频列表
├── bad_videos.txt                   # 坏视频列表
└── verification_v4/                 # Tier 1 VLM 审核结果
```

---

## 11. 运行命令

```bash
# 环境
VBENCH_PYTHON=/home/xiangyuz22/miniconda3/envs/vbench/bin/python
export GEMINI_API_KEY="..."

# Step 0: Caption 标注
python run_captioner.py --video_dir mushroom_data/videos --output_dir mushroom_data/captions --num_workers 100

# Tier 2: VLM 评估（Pro-preview）
python -m mushroom_eval.run_vlm --video_dir mushroom_data/videos \
    --caption_file mushroom_data/captions/captions.json \
    --model gemini-3.1-pro-preview --max_concurrent 100

# Tier 1: 信号指标（需 vbench 环境 + GPU）
$VBENCH_PYTHON -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D1 D5  # RAFT
$VBENCH_PYTHON -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D2     # AMT
$VBENCH_PYTHON -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D3     # CPU
$VBENCH_PYTHON -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D4     # DINO

# 融合分类
python -m mushroom_eval.run_classify \
    --vlm_results mushroom_eval_results/vlm_results.json \
    --tier1_results mushroom_eval_results/tier1_results.json \
    --percentile 25 --veto_consistency 0.85

# 模型对比
python -m mushroom_eval.run_classify \
    --vlm_results mushroom_eval_results/vlm_results.json \
    --vlm_results_b mushroom_eval_results/vlm_results_flash_lite.json \
    --compare

# Tier 1 可靠性验证
$VBENCH_PYTHON -m mushroom_eval.verify_tier1_v2 --video_dir mushroom_data/videos \
    --sample_size 20 --model gemini-3.1-flash-lite-preview
```

---

## 12. 已知问题与注意事项

1. **Pro-preview 的 max_output_tokens 必须设 8192+**：Pro 模型有内部 thinking chain（~1000 tokens），`max_output_tokens` 是 thinking + 输出的总限额。设太小会导致实际输出被截断。

2. **Pro-preview 不适合逐维度 1-10 打分**：倾向全给 1 分或返回 None。Yes/No 问答可靠。逐维度打分应使用 Flash-Lite。

3. **D4 在本批视频上区分度有限**：大部分视频 DINO 分在 0.92-0.99，但仍可检测极端低值（<0.85 有 187 条）。

4. **D1 无法区分主体运动和背景运动**：光流度量全局运动，当背景大幅运动但主体静止时会给高分。VLM 的 Q4（运动主体正确性）弥补此缺陷。

5. **阈值需要根据业务需求调整**：当前 P25 阈值判坏 ~50%，可通过调整 percentile 或 veto 阈值控制。
