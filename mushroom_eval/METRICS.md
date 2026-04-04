# 评估指标说明文档

## 最终方案使用的 4 个指标

| 指标 | 类型 | 值域 | 方向 | 说明 |
|------|------|------|------|------|
| vlm_score | VLM | 1-10 | 高=好 | Gemini Pro 综合质量评分 |
| D2_motion_smoothness | Tier1 | 0-1 | 高=好 | AMT 帧插值运动平滑度 |
| D3_temporal_flickering | Tier1 | 0-1 | 高=好 | 帧间像素 MAE 时序稳定性 |
| D4_subject_consistency | Tier1 | 0-1 | 高=好 | DINO ViT 主体外观一致性 |

---

## vlm_score — Gemini Pro 综合质量评分

- **模型：** gemini-3.1-pro-preview
- **输入：** 原始 MP4 视频 + 评估 prompt
- **输出：** 1-10 整数分 + 具体问题列表（JSON）
- **评分标准：**
  - 1-3 分：有严重缺陷
  - 4-5 分：有明显缺陷
  - 6-7 分：有轻微问题
  - 8-10 分：质量好

**检查的 6 类缺陷：**

| 缺陷 | 对应人类坏特征 |
|------|---------------|
| MOTION SUBJECT ERROR | IP/运动主体错误 |
| STIFF MOTION | IP/动作僵硬, IP/动作卡顿 |
| APPEARANCE DRIFT | IP/外观漂移, IP/不合理的大小突变 |
| FRAME JUMPS | IP无关/帧间跳变, IP/动作位移不连贯 |
| VISUAL ARTIFACTS | IP无关/出现奇怪未知物体 |
| PHYSICS VIOLATIONS | IP无关/物理规律混乱 |

**关键设计：** prompt 明确告知模型"角色安静坐着不算缺陷"，减少误报。

**配置：** `max_output_tokens=8192`（Pro 有内部 thinking chain ~1000 tokens，需留足空间）

---

## D2_motion_smoothness — 运动平滑度

- **模型：** AMT (Arbitrary Motion Trajectories) 帧插值
- **计算方法：**
  1. 对每对相邻帧做中间帧插值
  2. 将插值帧与实际中间帧做像素级 MAE 比较
  3. MAE 越小 = 运动越可预测/越平滑
  4. 最终分数 = (255 - mean_MAE) / 255
- **对应坏特征：** IP/动作卡顿, IP无关/帧间跳变
- **来源：** VBench `vbench/motion_smoothness.py`

---

## D3_temporal_flickering — 时序稳定性

- **模型：** 无（纯数学计算，CPU 即可）
- **计算方法：**
  1. 计算所有相邻帧对的像素级 MAE
  2. 取均值
  3. 分数 = (255 - mean_MAE) / 255
- **对应坏特征：** IP无关/帧间跳变, IP/外观漂移 衣服身体的时间一致性差
- **来源：** VBench `vbench/temporal_flickering.py`

---

## D4_subject_consistency — 主体外观一致性

- **模型：** DINO ViT-B16
- **计算方法：**
  1. 用 DINO 提取每帧的全局视觉特征向量
  2. 计算每帧与首帧的余弦相似度 + 相邻帧间余弦相似度
  3. 取平均值
- **对应坏特征：** IP/不合理的大小突变, IP/外观漂移 衣服身体的时间一致性差
- **来源：** VBench `vbench/subject_consistency.py`

---

## 已排除的指标

| 指标 | 排除原因 |
|------|----------|
| D1_mean_flow (RAFT 光流) | 不区分主体/背景运动，bad 视频值反而更高 |
| D1_static_ratio | 同上 |
| D5_flow_acceleration | 对分类无增量贡献 |
| D5_flow_spatial_var | 对分类无增量贡献 |
| VLM Q1-Q9 分项评分 | 单项区分力极弱（bad/good 均分差 < 0.4），被综合评分包含 |
