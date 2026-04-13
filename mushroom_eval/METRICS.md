# 评估指标说明

## 最终规则使用的 3 个信号

| 信号 | 类型 | 值域 | 规则阈值 | 说明 |
|------|------|------|----------|------|
| vlm_score | VLM 综合评分 | 1-10 | `<= 3` | Gemini Pro 看完视频打的综合质量分 |
| D2_motion_smoothness | VBench Tier1 | 0-1 | `< 0.989` | AMT 帧插值运动平滑度 |
| D3_temporal_flickering | VBench Tier1 | 0-1 | `< 0.970` | 帧间像素 MAE 时序稳定性 |

**最终规则：** `bad ⇔ vlm_score <= 3 AND (D3 < 0.970 OR D2 < 0.989)`

---

## vlm_score — Gemini Pro 综合质量评分

- **模型：** `gemini-3.1-pro-preview`
- **输入：** 原始 MP4（无预处理）+ 结构化评估 prompt
- **输出：** `{"score": 1-10, "issues": ["..."]}`
- **评分含义：**
  - 1-3：严重缺陷（motion subject error / stiff motion / appearance drift / frame jumps / artifacts / physics violation）
  - 4-5：明显缺陷
  - 6-7：轻微问题
  - 8-10：质量好

- **关键配置：** `max_output_tokens=8192`, `response_mime_type="application/json"`, `temperature=0.0`
- **局限：**
  - 与人类标注存在系统偏差（score=10 仍有 28% 是人类 bad）
  - 对"动作位移不连贯"、"主体动作缓慢"等细节不敏感
  - 因此规则只信任最极端的 ≤3 区间

---

## D2_motion_smoothness — 运动平滑度

- **模型：** AMT (Arbitrary Motion Trajectories) 帧插值
- **计算：** 插值中间帧 vs 真实中间帧的像素级 MAE；分数 = (255 - mean_MAE) / 255
- **值域：** 0-1，越高越平滑
- **阈值 < 0.989：** 对应人类 bad 分布的底部，能捕捉明显的运动不连续

---

## D3_temporal_flickering — 时序稳定性

- **模型：** 无（纯数学，CPU 即可）
- **计算：** 相邻帧像素级 MAE 均值；分数 = (255 - mean_MAE) / 255
- **值域：** 0-1，越高越稳定
- **阈值 < 0.970：** 过滤有明显闪烁/帧跳变的视频

---

## D4_subject_consistency — 外观一致性（规则未用）

- **模型：** DINO ViT-B16
- **计算：** 每帧与首帧的余弦相似度 + 相邻帧相似度的平均
- **在规则中被排除的原因：** 经 ablation 验证，加入 D4 没有带来 precision 提升，且 D2+D3 已覆盖大部分外观漂移信号

---

## 被排除的指标

| 指标 | 排除原因 |
|------|----------|
| D1_mean_flow | 全局光流不区分主体/背景运动，bad 视频反而值更高 |
| D1_static_ratio | 同上 |
| D5_flow_acceleration | 对 precision 无增量贡献 |
| D5_flow_spatial_var | 对 precision 无增量贡献 |
| VLM 9 个分项问题 (Q1-Q9) | 分项 yes/no 的区分力远弱于综合 1-10 打分 |
| VLM flash-lite 1-10 | bad/good 均分差异极小（< 0.3），pro-preview 更好 |

---

## 错误分析（全集 147 条 Bad 预测）

| 类型 | 数量 | 占比 |
|------|------|------|
| True Positive (TP) | 99 | 67.3% |
| False Positive (FP) | 48 | 32.7% |
| False Negative (FN) | 707 | - |

48 条 FP 的典型模式：
- VLM 对轻微不自然的视频给了 2-3 分，但人类认为可接受
- VBench 数值在边缘（D3 正好 0.96-0.97 区间）

707 条 FN（漏检 bad）主要是：
- VLM 给了 5+ 分（语义上看起来 OK）
- VBench 数值在正常范围（细微缺陷不触发信号）
- 这些视频的缺陷是**人类看得出但自动方法看不出**的（如轻微动作僵硬、小幅外观漂移）
