# 蘑菇TUTU视频质量评估 — 技术文档

## 1. 目标

对首帧引导生成的蘑菇TUTU视频进行自动分类（good/bad），**优先保证精度**（precision），低 recall 可接受 — 挑出的 bad 视频要尽量可靠。

---

## 2. 方法（高精度规则）

### 规则

```
bad  ⇔  VLM_score <= 3  AND  (D3_temporal_flickering < 0.970  OR  D2_motion_smoothness < 0.989)
```

即三个条件同时满足才判 bad：
1. Gemini Pro 综合打分 ≤ 3（视频有严重质量问题）
2. **且** 时序闪烁或运动平滑度至少有一个指标显著低于正常水平

### 为什么是规则不是分类器

经过 SVM / XGBoost / LogReg 等多种分类器调参，在 4-40 维特征上 F1 上限 ≈ 0.65，且 precision 仅 0.53。改用**多信号共识规则**后 precision 提升至 0.67+，因为：

1. **VLM 和 VBench 在极端区间各自可靠**：VLM 给 ≤3 分的视频中 56% 是真 bad；D2/D3 在极端低值区 bad rate 也高
2. **两个独立信号源同时告警时可信度更高**：两者都说有问题时 precision 到 70%+
3. **分类器会把两个低区分力特征平均，反而稀释信号**

### 性能

| 评估集 | Precision | Recall | 判 Bad 数 |
|--------|-----------|--------|-----------|
| Train (30%, 583 条) | 0.644 | - | - |
| Test (70%, 1363 条) | **0.686** | 0.127 | 102 |
| 全集 (1946 条) | **0.673** | 0.123 | 147 |

Train→Test 无明显过拟合（P 从 0.644→0.686 反而更高）。

---

## 3. 架构

```
视频.mp4
    │
    ├─── Gemini gemini-3.1-pro-preview VLM 综合评分 (1-10 + 问题列表)
    │
    ├─── VBench D2 motion_smoothness (AMT 帧插值误差)
    │
    ├─── VBench D3 temporal_flickering (帧间 MAE)
    │
    └─── 规则判定 → good/bad + 具体触发原因
```

### 特征说明

| 特征 | 来源 | 值域 | 作用 |
|------|------|------|------|
| vlm_score | Gemini Pro | 1-10 | 语义级综合判断 |
| D2_motion_smoothness | AMT 帧插值 | 0-1 | 运动是否平滑 |
| D3_temporal_flickering | 帧间 MAE | 0-1 | 帧间是否稳定 |
| D4_subject_consistency | DINO ViT | 0-1 | 外观是否一致（参考保留，规则不用） |

---

## 4. VLM Prompt（V2 calibrated）

关键设计：明确告知模型"角色安静姿态是正常的"，避免把静止视频误判为缺陷。

```
You are evaluating AI-generated videos of a mushroom character called 蘑菇TUTU.

IMPORTANT: These are 5-second clips. Some videos intentionally show the character
in calm/still poses — this is NORMAL and NOT a defect. Only flag ACTUAL generation failures:

1. MOTION SUBJECT ERROR: Character frozen while only background/camera moves
2. STIFF MOTION: Mechanical/robotic motion with no natural body deformation
3. APPEARANCE DRIFT: Color/texture/shape/size visibly morphing
4. FRAME JUMPS: Sudden discontinuities or teleportation
5. VISUAL ARTIFACTS: Distorted objects, melting shapes
6. PHYSICS VIOLATIONS: Impossible floating, incorrect merging

Score 1-10 (1=severe, 10=perfect). Remember: calm/small movements are NOT defects.

Output ONLY JSON: {"score": N, "issues": ["brief description", ...]}
```

配置：`max_output_tokens=8192`（Pro 有内部 thinking chain），`response_mime_type="application/json"`

---

## 5. 文件结构

```
mushroom_eval/
├── classify_highprec.py       # 最终方案：规则分类器
├── classify_final.py          # 早期 SVM 方案（保留参考）
├── config.py                  # VLM 配置
├── vlm_evaluator.py           # Gemini 异步批量评估
├── tier1_metrics.py           # VBench D1-D5 实现
├── run_vlm.py                 # VLM 评估 CLI
├── run_tier1.py               # VBench 评估 CLI
├── PIPELINE.md                # 本文档
└── METRICS.md                 # 指标详细说明

mushroom_eval_results/
├── badcase_list.txt                       # 人类专家标注（808 条 bad）
├── D2_motion_smoothness.json              # VBench D2 全量
├── D3_temporal_flickering.json            # VBench D3 全量
├── D4_subject_consistency.json            # VBench D4 全量
└── final/
    ├── vlm_scores.json                    # Gemini Pro 评分缓存
    ├── vbench_scores.json                 # VBench 评分缓存
    ├── classification_final.json          # 最终分类（规则）
    ├── classification_final.csv
    ├── bad_videos.txt                     # 147 条高置信 bad
    └── good_videos.txt
```

---

## 6. 使用方式

### 前置条件
1. Gemini Pro VLM 评分（`vlm_scores.json`，JSON: `{video: {score: 1-10, issues: [...]}}`）
2. VBench 评分（`vbench_scores.json`，JSON: `{video: {motion_smoothness, temporal_flickering, subject_consistency}}`）

### 运行规则分类

```bash
python -m mushroom_eval.classify_highprec \
    --vlm_scores mushroom_eval_results/final/vlm_scores.json \
    --vbench_scores mushroom_eval_results/final/vbench_scores.json \
    --badcase_list mushroom_eval_results/badcase_list.txt \
    --output_dir mushroom_eval_results/final
```

输出：
- `classification_final.json/csv` — 每条视频的 label + bad_reasons + 分数
- `bad_videos.txt` / `good_videos.txt` — 视频列表

### 从头生成 VLM/VBench 评分

VLM: `GEMINI_API_KEY=xxx python -m mushroom_eval.run_vlm --video_dir ... --model gemini-3.1-pro-preview`

VBench: `python -m mushroom_eval.run_tier1 --video_dir ... --metrics D2 D3 D4`（需 GPU + vbench conda env）

---

## 7. 实验历程与关键发现

### 尝试过的方案

| 方案 | Precision | 说明 |
|------|-----------|------|
| VLM yes/no 手动阈值 | 0.42 | 最初方案 |
| VLM 1-10 flash-lite | 0.52 | 换 Pro 模型 |
| 权重校准 加权规则 | 0.54 | 在全集上调权 |
| SVM 16维全特征 5折CV | 0.55 | 分类器调参 |
| XGB 40维特征工程 | 0.58 | 特征组合 |
| **VBench 纯规则** | **0.79** | 24 条（数量少）|
| **VLM+VBench 规则** | **0.67** | **147 条 ← 最终采用** |

### 关键发现

1. **Pro VLM 和人类标准有系统性偏差**
   - VLM score=10 的视频中仍有 28% 是人类标的 bad（漏检 345 条）
   - VLM score=2 的视频中只有 56% 是真 bad（误报严重）
   - VLM 对"动作位移不连贯"(81% 漏)、"主体动作缓慢"(81% 漏)这些细节敏感度低

2. **VBench 信号指标在极端区域最可靠**
   - D2/D3/D4 的均值差异极小（<0.005），被分类器稀释
   - 但最差 5-10% 区间的 bad rate 显著高于整体

3. **多信号共识优于分类器**
   - 两个独立来源（VLM 语义 + VBench 信号）同时告警时 precision 最高
   - 分类器学不出这种"共识"逻辑，因为每个单信号都太弱

4. **帧率不是瓶颈**
   - 测试 fps=1/5/10，区分力几乎一样
   - 问题是模型**不知道什么算 bad**，不是"看不清"

5. **F1 全局上限 ≈ 0.65**
   - 达到此上限后 precision 和 recall 严格 trade-off
   - 优先 precision 时 recall 必然跌到 0.12 以下
