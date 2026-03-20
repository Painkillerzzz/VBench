#!/usr/bin/env python3
"""
从 VBench 自定义评估结果 JSON 中综合各维度得分，将视频二分为 good / bad，
并生成若干可视化图表和结果文件。

输入文件示例（results_..._eval_results.json）结构：
{
    "subject_consistency": [
        0.94,  # 维度整体分数（本脚本中不使用）
        [
            {"video_path": ".../10025.mp4", "video_results": 0.89},
            ...
        ]
    ],
    "background_consistency": [...],
    ...
}

综合评分与分段思路（无监督、尽量区分）：
1. 对每个维度，取该维度下每个视频的 score（video_results），收集为矩阵 X[v, d]。
2. 对每个维度做 z-score 标准化（减均值除以标准差），使不同维度处于可比较尺度。
3. 对每个视频，将各维度 z-score 简单平均，得到综合得分 composite_score（等权重）。
4. 将综合得分做 min-max 归一化到 [0, 1]，得到 composite_norm。
5. 在 composite_norm 上按分位数将所有视频划分为 9 段（九分位）：
   - 取下 2 段作为 bad（最差约 2/9），上 2 段作为 good（最好约 2/9），中间 5 段记为 neutral。

输出：
1. CSV: video_scores.csv，包含每个视频在各个维度上的分数、综合分数、归一化分数、标签。
2. 文本：good_videos.txt, bad_videos.txt。
3. HTML：good_videos.html, bad_videos.html（用 <video> 标签方便本地浏览）。
4. 图像：
   - 每个维度的分布直方图 dim_<name>_hist.png
   - 综合分数直方图 + 阈值线 composite_score_hist.png
   - 两个主要维度的散点图 scatter_main_dims.png（颜色区分 good / bad）。
5. JSON：analysis_summary.json（阈值、各维度统计信息、good/bad 数量等）。
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_eval_results(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_score_matrix(
    data: Dict[str, List],
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    将原始 JSON 解析为：
    - video_paths: 按固定顺序的所有视频路径列表
    - dimensions: 维度名称列表
    - scores: shape = [num_videos, num_dims] 的矩阵，可能含 np.nan
    """
    dimensions = list(data.keys())

    # 收集每个维度下所有视频分数
    dim_to_vid_scores: Dict[str, Dict[str, float]] = {}
    all_videos = set()
    for dim, value in data.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"维度 {dim} 的数据格式异常，应为 [overall, list_of_videos]")
        _, video_list = value
        m: Dict[str, float] = {}
        for item in video_list:
            vid = item["video_path"]
            score = float(item["video_results"])
            m[vid] = score
            all_videos.add(vid)
        dim_to_vid_scores[dim] = m

    video_paths = sorted(all_videos)
    num_videos = len(video_paths)
    num_dims = len(dimensions)

    scores = np.full((num_videos, num_dims), np.nan, dtype=float)
    for j, dim in enumerate(dimensions):
        m = dim_to_vid_scores[dim]
        for i, vid in enumerate(video_paths):
            if vid in m:
                scores[i, j] = m[vid]

    return video_paths, dimensions, scores


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """
    对每一列做 z-score 标准化，忽略 NaN：
    z = (x - mean) / std
    如果某列 std 为 0，则整列置为 0。
    """
    x = x.copy()
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = ~np.isnan(col)
        if not np.any(mask):
            x[:, j] = 0.0
            continue
        if stds[j] == 0 or math.isclose(stds[j], 0.0):
            x[mask, j] = 0.0
        else:
            x[mask, j] = (col[mask] - means[j]) / stds[j]
    return x


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if math.isclose(xmax, xmin):
        # 所有分数相同，返回全 0.5 以避免除零
        return np.full_like(x, 0.5)
    return (x - xmin) / (xmax - xmin)


def otsu_threshold(values: np.ndarray, num_bins: int = 256) -> float:
    """
    在 [0, 1] 区间上的一维数据上使用 Otsu 方法寻找最大类间方差阈值。
    要求输入已经预先缩放到 [0, 1]。
    返回阈值（实数）。
    """
    values = np.clip(values, 0.0, 1.0)
    hist, bin_edges = np.histogram(values, bins=num_bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5

    prob = hist / total
    omega = np.cumsum(prob)  # 累积权重
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(prob * bin_centers)
    mu_total = mu[-1]

    # 类间方差：σ_b^2 = (mu_T * omega - mu)^2 / (omega * (1 - omega))
    sigma_b2 = np.zeros_like(omega)
    for i in range(num_bins):
        w0 = omega[i]
        w1 = 1.0 - w0
        if w0 <= 0.0 or w1 <= 0.0:
            sigma_b2[i] = 0.0
            continue
        mu0 = mu[i] / w0
        mu1 = (mu_total - mu[i]) / w1
        sigma_b2[i] = w0 * w1 * (mu0 - mu1) ** 2

    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_dashboard_html(
    out_path: Path,
    video_paths: List[str],
    dimensions: List[str],
    scores: np.ndarray,
    composite_norm: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> None:
    """
    使用 Chart.js 生成一个交互式 dashboard.html：
    - 每个维度一个柱状图，x 轴为视频索引，y 轴为该维度得分，颜色区分 good/bad。
    - 综合分数柱状图，并画出阈值线。
    - 主维度（若存在 aesthetic_quality / imaging_quality）散点图，可视化区分度。
    """
    import json as _json

    # 为可视化准备简单的数据结构
    video_labels = [Path(v).name for v in video_paths]
    scores_list = {dim: scores[:, i].tolist() for i, dim in enumerate(dimensions)}
    composite_list = composite_norm.tolist()
    label_list = labels.tolist()

    # 选择主维度索引用于散点图
    main_idx = []
    preferred = ["aesthetic_quality", "imaging_quality"]
    for name in preferred:
        if name in dimensions:
            main_idx.append(dimensions.index(name))
    if len(main_idx) < 2 and len(dimensions) >= 2:
        main_idx = [0, 1]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Video quality analysis dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; }}
    .chart-container {{ width: 800px; margin: 20px auto; }}
  </style>
</head>
<body>
  <h1>Video quality analysis dashboard</h1>
  <p>阈值 (Otsu, 归一化综合分数): {threshold:.4f}</p>

  <div class="chart-container">
    <h2>Composite score (normalized)</h2>
    <canvas id="compositeChart"></canvas>
  </div>
"""

    for dim in dimensions:
        html += f"""
  <div class="chart-container">
    <h2>Dimension: {dim}</h2>
    <canvas id="dim_{dim}"></canvas>
  </div>
"""

    if len(main_idx) == 2:
        html += """
  <div class="chart-container">
    <h2>Main dimensions scatter (good vs bad)</h2>
    <canvas id="scatterChart"></canvas>
  </div>
"""

    html += f"""
  <script>
    const videoLabels = {_json.dumps(video_labels)};
    const dimensions = {_json.dumps(dimensions)};
    const scores = {_json.dumps(scores_list)};
    const composite = {_json.dumps(composite_list)};
    const labels = {_json.dumps(label_list)};
    const threshold = {threshold:.6f};

    function buildColors() {{
      return labels.map(l => l === "good"
        ? "rgba(75, 192, 192, 0.7)"
        : "rgba(255, 99, 132, 0.7)");
    }}

    // Composite bar chart with threshold line (as第二条数据集)
    const ctxComp = document.getElementById('compositeChart').getContext('2d');
    new Chart(ctxComp, {{
      type: 'bar',
      data: {{
        labels: videoLabels,
        datasets: [
          {{
            label: 'Composite score (normalized)',
            data: composite,
            backgroundColor: buildColors(),
          }},
          {{
            label: 'Threshold',
            type: 'line',
            data: composite.map(() => threshold),
            borderColor: 'rgba(0, 0, 0, 0.8)',
            borderWidth: 2,
            fill: false,
            pointRadius: 0,
          }}
        ]
      }},
      options: {{
        responsive: true,
        scales: {{
          y: {{
            beginAtZero: true,
            max: 1.0
          }}
        }}
      }}
    }});

    // Per-dimension charts
    dimensions.forEach(dim => {{
      const ctx = document.getElementById('dim_' + dim).getContext('2d');
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: videoLabels,
          datasets: [{{
            label: dim,
            data: scores[dim],
            backgroundColor: buildColors(),
          }}]
        }},
        options: {{
          responsive: true,
          scales: {{
            y: {{ beginAtZero: true }}
          }}
        }}
      }});
    }});
"""

    if len(main_idx) == 2:
        dim_x = dimensions[main_idx[0]]
        dim_y = dimensions[main_idx[1]]
        html += f"""
    // Scatter for main dimensions
    (function() {{
      const ctx = document.getElementById('scatterChart').getContext('2d');
      const dataPoints = composite.map((_, i) => {{
        return {{
          x: scores["{dim_x}"][i],
          y: scores["{dim_y}"][i],
          label: videoLabels[i],
          group: labels[i],
        }};
      }});
      new Chart(ctx, {{
        type: 'scatter',
        data: {{
          datasets: [
            {{
              label: 'good',
              data: dataPoints.filter(p => p.group === 'good'),
              backgroundColor: 'rgba(75, 192, 192, 0.7)',
            }},
            {{
              label: 'bad',
              data: dataPoints.filter(p => p.group === 'bad'),
              backgroundColor: 'rgba(255, 99, 132, 0.7)',
            }},
          ]
        }},
        options: {{
          responsive: true,
          plugins: {{
            tooltip: {{
              callbacks: {{
                label: function(context) {{
                  const p = context.raw;
                  return p.label + ' (' + p.x.toFixed(3) + ', ' + p.y.toFixed(3) + ')';
                }}
              }}
            }}
          }},
          scales: {{
            x: {{ title: {{ display: true, text: '{dim_x}' }} }},
            y: {{ title: {{ display: true, text: '{dim_y}' }} }}
          }}
        }}
      }});
    }})();
"""

    html += """
  </script>
</body>
</html>
"""

    with out_path.open("w", encoding="utf-8") as f:
        f.write(html)


def write_video_lists(
    video_paths: List[str],
    labels: np.ndarray,
    out_dir: Path,
) -> None:
    files = {
        "good": out_dir / "good_videos.txt",
        "bad": out_dir / "bad_videos.txt",
        "neutral": out_dir / "neutral_videos.txt",
    }
    handles = {
        k: path.open("w", encoding="utf-8") for k, path in files.items()
    }
    try:
        for vid, lab in zip(video_paths, labels):
            lab = lab if lab in handles else "neutral"
            handles[lab].write(vid + "\n")
    finally:
        for f in handles.values():
            f.close()


def write_video_html(
    video_paths: List[str],
    labels: np.ndarray,
    out_dir: Path,
) -> None:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for vid, lab in zip(video_paths, labels):
        grouped[lab].append(vid)

    for lab, vids in grouped.items():
        html_path = out_dir / f"{lab}_videos.html"
        with html_path.open("w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>")
            f.write(f"{lab} videos</title></head><body>\n")
            f.write(f"<h1>{lab} videos (n={len(vids)})</h1>\n")
            for v in vids:
                # 使用相对路径，避免依赖服务器根目录
                try:
                    rel = os.path.relpath(v, start=out_dir)
                except ValueError:
                    rel = v
                f.write("<div style='margin-bottom:20px;'>\n")
                f.write(f"<p>{rel}</p>\n")
                f.write(
                    f"<video src='{rel}' controls width='320' preload='metadata'></video>\n"
                )
                f.write("</div>\n")
            f.write("</body></html>\n")


def save_summary_json(
    out_path: Path,
    dimensions: List[str],
    scores: np.ndarray,
    composite_score: np.ndarray,
    composite_norm: np.ndarray,
    thresholds: Dict[str, float],
    labels: np.ndarray,
) -> None:
    summary = {
        "num_videos": int(len(composite_score)),
        "dimensions": dimensions,
        "thresholds": thresholds,
        "num_good": int(np.sum(labels == "good")),
        "num_bad": int(np.sum(labels == "bad")),
        "num_neutral": int(np.sum(labels == "neutral")),
        "dimension_stats": {},
        "composite_stats": {
            "mean": float(np.mean(composite_score)),
            "std": float(np.std(composite_score)),
            "min": float(np.min(composite_score)),
            "max": float(np.max(composite_score)),
        },
    }
    for j, dim in enumerate(dimensions):
        col = scores[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        summary["dimension_stats"][dim] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_video_scores_csv(
    out_path: Path,
    video_paths: List[str],
    dimensions: List[str],
    scores: np.ndarray,
    composite_score: np.ndarray,
    composite_norm: np.ndarray,
    labels: np.ndarray,
) -> None:
    import csv

    header = ["video_path"] + dimensions + ["composite_score", "composite_norm", "label"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, vid in enumerate(video_paths):
            row = [vid]
            for j in range(len(dimensions)):
                val = scores[i, j]
                row.append("" if np.isnan(val) else float(val))
            row.append(float(composite_score[i]))
            row.append(float(composite_norm[i]))
            row.append(labels[i])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="综合多维度得分，将视频划分为 good/neutral/bad，并生成可视化结果。"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="评估结果 JSON 文件路径（如 results_..._eval_results.json）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_results",
        help="输出目录（默认：analysis_results）",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    data = load_eval_results(eval_path)
    video_paths, dimensions, scores = build_score_matrix(data)

    # 对缺失值，用维度均值填充
    for j in range(scores.shape[1]):
        col = scores[:, j]
        mask = np.isnan(col)
        if np.any(mask):
            mean_val = np.nanmean(col)
            col[mask] = mean_val
            scores[:, j] = col

    # 1) 维度 z-score 标准化并计算综合分数
    scores_z = zscore_normalize(scores)
    composite_score = np.mean(scores_z, axis=1)
    composite_norm = minmax_normalize(composite_score)

    # 2) 按 9 段分位数划分，取头尾两段为 good / bad
    # q[i] 是 composite_norm 的 i/9 分位点（i=0..9）
    quantiles = np.quantile(composite_norm, [i / 9.0 for i in range(10)])
    low_thr = float(quantiles[2])   # 第 2/9 位置，以下为 bad
    high_thr = float(quantiles[7])  # 第 7/9 位置，以上为 good
    labels = np.full_like(composite_norm, fill_value="neutral", dtype=object)
    labels[composite_norm < low_thr] = "bad"
    labels[composite_norm > high_thr] = "good"

    thresholds = {
        "low_2_9": low_thr,
        "high_7_9": high_thr,
        "all_quantiles_0_9": [float(q) for q in quantiles.tolist()],
    }

    # 3) 输出列表与 HTML 预览（视频可视化）
    write_video_lists(video_paths, labels, out_dir)
    write_video_html(video_paths, labels, out_dir)

    # 4) 保存 summary JSON 与明细 CSV
    save_summary_json(
        out_dir / "analysis_summary.json",
        dimensions,
        scores,
        composite_score,
        composite_norm,
        thresholds,
        labels,
    )
    save_video_scores_csv(
        out_dir / "video_scores.csv",
        video_paths,
        dimensions,
        scores,
        composite_score,
        composite_norm,
        labels,
    )

    # 5) 生成交互式 dashboard（各维度得分与综合分数可视化）
    write_dashboard_html(
        out_dir / "dashboard.html",
        video_paths,
        dimensions,
        scores,
        composite_norm,
        labels,
        high_thr,
    )

    print(f"分析完成，共 {len(video_paths)} 个视频。")
    print(
        f"good: {int(np.sum(labels == 'good'))}, "
        f"bad: {int(np.sum(labels == 'bad'))}, "
        f"neutral: {int(np.sum(labels == 'neutral'))}"
    )
    print(
        f"九分位阈值: low_2_9={low_thr:.4f}, high_7_9={high_thr:.4f}"
    )
    print(f"可视化与结果已保存在目录: {out_dir}")


if __name__ == "__main__":
    main()

