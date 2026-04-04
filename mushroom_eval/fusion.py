"""Fusion layer: combine VLM + Tier 1 scores into final binary classification.

Supports:
  - VLM-only classification (when tier1 not yet available)
  - VLM + Tier1 combined classification
  - Flexible thresholding: percentile-based or fixed
  - Score comparison between flash-lite and pro-preview
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Bad feature mapping ──────────────────────────────────────────────────

# Maps VLM question IDs to bad feature descriptions
VLM_BAD_FEATURE_MAP = {
    "Q1_limb_motion": "四肢无法移动/身体僵硬",
    "Q2_natural_motion": "动作卡顿僵硬（非活泼自然）",
    "Q3_mostly_still": "过长时间静止不动",
    "Q4_correct_subject": "运动主体错误",
    "Q5_strange_objects": "出现奇怪/未知物体",
    "Q6_physics": "物理规律混乱",
    "Q7_size_consistency": "不合理的大小突变",
    "Q8_appearance_consistency": "衣服/身体时间一致性差",
    "Q9_motion_continuity": "动作/位移不连贯",
}


def classify_vlm_only(
    vlm_results: list[dict[str, Any]],
    threshold: float | None = None,
    percentile: float = 20.0,
) -> list[dict[str, Any]]:
    """Classify videos using VLM scores only.

    Args:
        vlm_results: List of VLM result dicts (from vlm_evaluator)
        threshold: Fixed threshold (0-1). If None, use percentile.
        percentile: Bottom N% are classified as bad (default: 20th percentile)

    Returns:
        List of classification dicts with label, scores, bad features
    """
    valid = [r for r in vlm_results if r.get("status") == "success" and r["vlm_score"] >= 0]

    if threshold is None:
        scores = sorted([r["vlm_score"] for r in valid])
        threshold = float(np.percentile(scores, percentile))
        logger.info("Auto threshold (P%.0f): %.3f", percentile, threshold)

    classified = []
    for r in vlm_results:
        score = r.get("vlm_score", -1.0)
        status = r.get("status", "unknown")

        if status != "success" or score < 0:
            label = "error"
            bad_features = ["评估失败"]
        elif score <= threshold:
            label = "bad"
            bad_features = _extract_bad_features(r.get("vlm_details", {}))
        else:
            label = "good"
            bad_features = []

        classified.append({
            "video_path": r["video_path"],
            "video": Path(r["video_path"]).name,
            "label": label,
            "vlm_score": score,
            "vlm_details": r.get("vlm_details", {}),
            "bad_features": bad_features,
        })

    return classified


def _extract_bad_features(vlm_details: dict[str, float]) -> list[str]:
    """Extract triggered bad features from VLM per-question scores."""
    bad = []
    for q_id, score in vlm_details.items():
        if score == 0.0 and q_id in VLM_BAD_FEATURE_MAP:
            bad.append(VLM_BAD_FEATURE_MAP[q_id])
    return bad


def classify_combined(
    vlm_results: list[dict[str, Any]],
    tier1_results: list[dict[str, Any]] | None = None,
    vlm_threshold: float | None = None,
    vlm_percentile: float = 20.0,
    tier1_veto_rules: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Classify using VLM + optional Tier 1 metrics.

    Tier 1 veto rules: if any tier1 metric crosses the veto threshold,
    the video is classified as bad regardless of VLM score.

    Args:
        vlm_results: VLM result list
        tier1_results: Optional tier1 result list (merged format)
        vlm_threshold: Fixed VLM threshold
        vlm_percentile: Percentile for auto-threshold
        tier1_veto_rules: Dict of metric_name -> veto_threshold
            e.g. {"static_ratio": 0.95, "subject_consistency_below": 0.90}
    """
    # Start with VLM classification
    classified = classify_vlm_only(vlm_results, vlm_threshold, vlm_percentile)

    if tier1_results is None:
        return classified

    # Index tier1 by video path
    t1_by_path = {r["video_path"]: r for r in tier1_results}

    # Default veto rules
    if tier1_veto_rules is None:
        tier1_veto_rules = {}

    for entry in classified:
        vp = entry["video_path"]
        t1 = t1_by_path.get(vp, {})

        if not t1:
            continue

        # Apply tier1 veto rules
        vetoed = False
        veto_reasons = []

        # Static ratio > threshold → almost no movement
        if "static_ratio" in tier1_veto_rules:
            sr = t1.get("static_ratio", 0)
            if sr is not None and sr >= tier1_veto_rules["static_ratio"]:
                vetoed = True
                veto_reasons.append(f"主体动作缓慢(static_ratio={sr:.2f})")

        # Subject consistency below threshold → appearance drift
        if "subject_consistency_below" in tier1_veto_rules:
            sc = t1.get("subject_consistency", 1.0)
            if sc is not None and sc < tier1_veto_rules["subject_consistency_below"]:
                vetoed = True
                veto_reasons.append(f"外观一致性差(DINO={sc:.4f})")

        # Flow acceleration above threshold → motion discontinuity
        if "flow_acceleration_above" in tier1_veto_rules:
            fa = t1.get("flow_acceleration", 0)
            if fa is not None and fa > tier1_veto_rules["flow_acceleration_above"]:
                vetoed = True
                veto_reasons.append(f"运动不连贯(accel={fa:.2f})")

        if vetoed and entry["label"] != "bad":
            entry["label"] = "bad"
            entry["bad_features"].extend(veto_reasons)
            entry["tier1_vetoed"] = True

        # Attach tier1 scores
        entry["tier1"] = {k: v for k, v in t1.items() if k != "video_path"}

    return classified


def compare_vlm_models(
    results_a: list[dict], label_a: str,
    results_b: list[dict], label_b: str,
) -> dict[str, Any]:
    """Compare two VLM model results (e.g., flash-lite vs pro-preview)."""
    # Index by video path
    a_by_path = {r["video_path"]: r for r in results_a if r.get("status") == "success"}
    b_by_path = {r["video_path"]: r for r in results_b if r.get("status") == "success"}

    common = set(a_by_path.keys()) & set(b_by_path.keys())
    logger.info("Comparing %s vs %s: %d common videos", label_a, label_b, len(common))

    diffs = []
    agreements = 0
    for vp in common:
        sa = a_by_path[vp]["vlm_score"]
        sb = b_by_path[vp]["vlm_score"]
        diff = abs(sa - sb)
        diffs.append(diff)
        if diff < 0.15:  # within ~1 question difference
            agreements += 1

    diffs_arr = np.array(diffs) if diffs else np.array([0])

    # Per-question agreement
    q_agree = {}
    for vp in common:
        ans_a = a_by_path[vp].get("vlm_answers", {})
        ans_b = b_by_path[vp].get("vlm_answers", {})
        for q_id in ans_a:
            if q_id not in q_agree:
                q_agree[q_id] = {"agree": 0, "total": 0}
            q_agree[q_id]["total"] += 1
            if ans_a.get(q_id) == ans_b.get(q_id):
                q_agree[q_id]["agree"] += 1

    comparison = {
        "models": [label_a, label_b],
        "common_videos": len(common),
        "score_agreement_rate": float(agreements / len(common)) if common else 0,
        "score_diff_mean": float(diffs_arr.mean()),
        "score_diff_median": float(np.median(diffs_arr)),
        "score_diff_max": float(diffs_arr.max()),
        "per_question_agreement": {
            q_id: round(v["agree"] / v["total"], 3) if v["total"] > 0 else 0
            for q_id, v in q_agree.items()
        },
    }
    return comparison


def save_classification(
    classified: list[dict],
    output_dir: str,
    filename: str = "classification.json",
) -> None:
    """Save classification results and print summary."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)

    good = [c for c in classified if c["label"] == "good"]
    bad = [c for c in classified if c["label"] == "bad"]
    error = [c for c in classified if c["label"] == "error"]

    logger.info("Classification saved to %s", path)
    print(f"\n{'='*50}")
    print(f"Classification Summary")
    print(f"{'='*50}")
    print(f"  Total:  {len(classified)}")
    print(f"  Good:   {len(good)} ({100*len(good)/len(classified):.1f}%)")
    print(f"  Bad:    {len(bad)} ({100*len(bad)/len(classified):.1f}%)")
    if error:
        print(f"  Error:  {len(error)}")

    # Bad feature frequency
    if bad:
        from collections import Counter
        feat_counts = Counter()
        for c in bad:
            for feat in c.get("bad_features", []):
                feat_counts[feat] += 1
        print(f"\n  Top bad features:")
        for feat, count in feat_counts.most_common(10):
            print(f"    {feat}: {count} ({100*count/len(bad):.0f}%)")
    print(f"{'='*50}\n")

    # Also save bad video list (convenient)
    bad_list_path = os.path.join(output_dir, "bad_videos.txt")
    with open(bad_list_path, "w") as f:
        for c in bad:
            f.write(c["video"] + "\n")

    good_list_path = os.path.join(output_dir, "good_videos.txt")
    with open(good_list_path, "w") as f:
        for c in good:
            f.write(c["video"] + "\n")

    logger.info("Bad list: %s, Good list: %s", bad_list_path, good_list_path)
