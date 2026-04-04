# 蘑菇TUTU视频质量评估 — 技术文档

## 1. 项目概述

**目标：** 对首帧引导生成的蘑菇TUTU视频进行自动质量评估，筛选出含坏特征的视频（good/bad 二分类）。

**输入：**
- MP4 视频（5秒，720×1280）
- 人类专家标注的 badcase_list.txt（808 条 bad，用于训练 SVM 分类器）

**输出：**
- 每条视频的 good/bad 标签 + bad 概率 + 各维度分数 + 具体问题描述
- `classification_final.json` / `classification_final.csv`

---

## 2. 方法

### 架构

```
视频.mp4
    │
    ├─── Gemini Pro VLM 评分 (1-10 综合分) ───┐
    │                                          │
    ├─── D2: 运动平滑度 (VBench AMT)     ─────┤→ 4维特征 → SVM → good/bad
    ├─── D3: 时序稳定性 (VBench MAE)     ─────┤
    └─── D4: 外观一致性 (VBench DINO)    ─────┘
```

### 特征（4维）

| 特征 | 来源 | 说明 |
|------|------|------|
| vlm_score | Gemini gemini-3.1-pro-preview | 综合质量 1-10 分 + 具体问题列表 |
| D2_motion_smoothness | VBench AMT 帧插值 | 运动流畅度 0-1 |
| D3_temporal_flickering | VBench 帧间 MAE | 帧间稳定性 0-1 |
| D4_subject_consistency | VBench DINO ViT | 主体外观一致性 0-1 |

### 分类器

- SVM (RBF kernel, C=0.1, class_weight={0:1, 1:2.0})
- StandardScaler 归一化
- 70% 训练 / 30% 测试

### VLM Prompt

关键设计：告诉模型"安静坐着不算缺陷"，只检测真正的生成失败。

检查 6 类缺陷：
1. MOTION SUBJECT ERROR — 角色完全冻结而只有背景在动
2. STIFF MOTION — 角色运动时像机器人般僵硬（安静或缓慢不算）
3. APPEARANCE DRIFT — 颜色/纹理/大小在视频中发生变化
4. FRAME JUMPS — 突然的不连续/瞬移
5. VISUAL ARTIFACTS — 畸变物体或视觉伪影
6. PHYSICS VIOLATIONS — 物理规律违反

输出格式：`{"score": N, "issues": ["...", ...]}`

### 被排除的指标

以下指标经实验验证对分类无贡献，已从最终方案中移除：

| 指标 | 排除原因 |
|------|----------|
| D1_mean_flow | 不区分主体/背景运动，bad 视频反而值更高 |
| D1_static_ratio | 同上，方向反了 |
| D5_flow_acceleration | 对分类无增量贡献 |
| D5_flow_spatial_var | 对分类无增量贡献 |
| VLM 9Q 分问题评分 | flash-lite 的 1-10 分几乎无区分力，pro 的 9Q 也被 direct score 包含 |
| D2/D4 enhanced (min/P5/std) | SVM 已从 mean 学到信号，增加变体无增量 |

---

## 3. 性能

**Test set（30%, 584 条，未参与训练/调参）：**

| 指标 | 值 |
|------|-----|
| F1 | 0.634 |
| Precision | 0.534 |
| Recall | 0.781 |

**校准过程中尝试过的方案对比：**

| 方案 | F1 | 说明 |
|------|-----|------|
| VLM yes/no 手动阈值 | 0.480 | 最初方案 |
| VLM 1-10 flash-lite 手动阈值 | 0.449 | flash-lite 区分力不足 |
| 权重校准 (加权 yes/no) | 0.628 | 在全集上调权 |
| SVM 16维全特征 | 0.652 | 5折CV，包含冗余特征 |
| **SVM 4维 (V2+D2+D3+D4)** | **0.634** | **最终方案，严格 train/test split** |

---

## 4. 文件结构

```
mushroom_eval/
├── classify_final.py      # 最终流水线（训练 + 推理）
├── config.py              # VLM 配置
├── vlm_evaluator.py       # Gemini 异步批量评估
├── tier1_metrics.py       # D2/D3/D4 信号指标
├── run_vlm.py             # VLM 评估 CLI
├── run_tier1.py           # Tier1 评估 CLI
├── PIPELINE.md            # 本文档
└── METRICS.md             # 指标详细说明

mushroom_eval_results/
├── final/
│   ├── model.pkl                  # 训练好的 SVM 模型
│   ├── classification_final.json  # 全量分类结果
│   ├── classification_final.csv   # CSV 格式
│   ├── bad_videos.txt
│   ├── good_videos.txt
│   ├── vlm_scores.json            # VLM 评分缓存
│   └── tier1_scores.json          # Tier1 评分缓存
├── D2_motion_smoothness.json      # D2 全量原始分数
├── D3_temporal_flickering.json    # D3 全量原始分数
├── D4_subject_consistency.json    # D4 全量原始分数
└── badcase_list.txt               # 人类专家标注
```

---

## 5. 使用方式

### 训练（需要人类标注）

```bash
export GEMINI_API_KEY="..."
python -m mushroom_eval.classify_final train \
    --video_dir mushroom_data/videos \
    --badcase_list mushroom_eval_results/badcase_list.txt \
    --output_dir mushroom_eval_results/final
```

### 推理（新视频）

```bash
export GEMINI_API_KEY="..."
python -m mushroom_eval.classify_final infer \
    --video_dir /path/to/new/videos \
    --model_path mushroom_eval_results/final/model.pkl \
    --output_dir /path/to/output
```

推理需要：
- Gemini API（gemini-3.1-pro-preview）
- VBench 环境（D2/D3/D4 需要 GPU）
- 已训练的 model.pkl

### 输出格式

每条视频一行 JSON/CSV：
```json
{
  "video": "123.mp4",
  "label": "bad",
  "bad_probability": 0.823,
  "vlm_score": 2,
  "vlm_issues": "Character stays mostly frozen; Stiff motion",
  "D2_motion_smoothness": 0.9876,
  "D3_temporal_flickering": 0.9654,
  "D4_subject_consistency": 0.8912
}
```
