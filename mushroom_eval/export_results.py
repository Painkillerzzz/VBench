"""Export all evaluation results into a single structured JSON and CSV.

Usage:
    python -m mushroom_eval.export_results \
        --vlm_results mushroom_eval_results/vlm_results_flash_lite.json \
        --caption_file mushroom_data/captions/captions.json \
        --output_dir mushroom_eval_results \
        --percentile 25 --veto_consistency 0.85
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Bad feature mapping from VLM question IDs
VLM_BAD_FEATURE_MAP = {
    "Q1_limb_motion": "四肢无法移动/身体僵硬",
    "Q2_natural_motion": "动作卡顿僵硬",
    "Q3_mostly_still": "过长时间静止不动",
    "Q4_correct_subject": "运动主体错误",
    "Q5_strange_objects": "出现奇怪/未知物体",
    "Q6_physics": "物理规律混乱",
    "Q7_size_consistency": "不合理的大小突变",
    "Q8_appearance_consistency": "衣服/身体时间一致性差",
    "Q9_motion_continuity": "动作/位移不连贯",
}


def load_metric_file(path: str) -> dict[str, dict]:
    """Load a metric JSON file, return dict keyed by video filename."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for r in data:
        name = Path(r["video_path"]).name
        result[name] = {k: v for k, v in r.items() if k != "video_path"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all results to JSON/CSV")
    parser.add_argument("--vlm_results", required=True)
    parser.add_argument("--caption_file", default="")
    parser.add_argument("--result_dir", default="mushroom_eval_results")
    parser.add_argument("--output_dir", default="mushroom_eval_results")
    parser.add_argument("--percentile", type=float, default=25.0)
    parser.add_argument("--veto_consistency", type=float, default=0.85)
    parser.add_argument("--veto_static", type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VLM results
    with open(args.vlm_results) as f:
        vlm_data = json.load(f)
    vlm_by_name = {}
    for r in vlm_data:
        name = Path(r["video_path"]).name
        vlm_by_name[name] = r
    logger.info("Loaded %d VLM results", len(vlm_by_name))

    # Load captions
    captions: dict[str, str] = {}
    if args.caption_file and os.path.exists(args.caption_file):
        with open(args.caption_file) as f:
            cap_data = json.load(f)
        if isinstance(cap_data, list):
            for item in cap_data:
                name = Path(item.get("video_path", "")).name
                if name and item.get("caption"):
                    captions[name] = item["caption"]
        elif isinstance(cap_data, dict):
            captions = {Path(k).name: v for k, v in cap_data.items()}
        logger.info("Loaded %d captions", len(captions))

    # Load Tier 1 metrics
    d1 = load_metric_file(os.path.join(args.result_dir, "D1_dynamic_degree.json"))
    d2 = load_metric_file(os.path.join(args.result_dir, "D2_motion_smoothness.json"))
    d3 = load_metric_file(os.path.join(args.result_dir, "D3_temporal_flickering.json"))
    d4 = load_metric_file(os.path.join(args.result_dir, "D4_subject_consistency.json"))
    d5 = load_metric_file(os.path.join(args.result_dir, "D5_flow_stability.json"))
    logger.info(
        "Tier1 coverage: D1=%d D2=%d D3=%d D4=%d D5=%d",
        len(d1), len(d2), len(d3), len(d4), len(d5),
    )

    # Compute VLM threshold
    import numpy as np
    valid_scores = sorted([
        r["vlm_score"] for r in vlm_data
        if r.get("status") == "success" and r["vlm_score"] >= 0
    ])
    vlm_threshold = float(np.percentile(valid_scores, args.percentile))
    logger.info("VLM threshold (P%.0f): %.3f", args.percentile, vlm_threshold)

    # Build unified records
    records = []
    for name, vlm in vlm_by_name.items():
        video_path = vlm["video_path"]
        vlm_score = vlm.get("vlm_score", -1.0)
        vlm_answers = vlm.get("vlm_answers", {})
        vlm_details = vlm.get("vlm_details", {})

        # Tier 1 scores
        t1_d1 = d1.get(name, {})
        t1_d2 = d2.get(name, {})
        t1_d3 = d3.get(name, {})
        t1_d4 = d4.get(name, {})
        t1_d5 = d5.get(name, {})

        # Classification logic
        bad_reasons: list[str] = []

        # VLM-based bad features
        if vlm_score >= 0 and vlm_score <= vlm_threshold:
            for q_id, score in vlm_details.items():
                if score == 0.0 and q_id in VLM_BAD_FEATURE_MAP:
                    bad_reasons.append(VLM_BAD_FEATURE_MAP[q_id])

        # Tier1 veto: subject_consistency
        sc = t1_d4.get("subject_consistency")
        if sc is not None and sc >= 0 and sc < args.veto_consistency:
            bad_reasons.append("外观一致性差")

        # Tier1 veto: static_ratio
        sr = t1_d1.get("static_ratio")
        if sr is not None and sr >= 0 and sr >= args.veto_static:
            bad_reasons.append("主体几乎静止")

        # Final label
        if vlm.get("status") != "success" or vlm_score < 0:
            label = "error"
            bad_reasons = ["VLM评估失败"]
        elif bad_reasons:
            label = "bad"
        else:
            label = "good"

        record = {
            "video": name,
            "video_path": video_path,
            "label": label,
            "caption": captions.get(name, ""),
            # VLM scores (0=bad, 1=good per question; vlm_score is the mean)
            "vlm_score": vlm_score,
            "vlm_Q1_limb_motion": vlm_details.get("Q1_limb_motion"),
            "vlm_Q2_natural_motion": vlm_details.get("Q2_natural_motion"),
            "vlm_Q3_mostly_still": vlm_details.get("Q3_mostly_still"),
            "vlm_Q4_correct_subject": vlm_details.get("Q4_correct_subject"),
            "vlm_Q5_strange_objects": vlm_details.get("Q5_strange_objects"),
            "vlm_Q6_physics": vlm_details.get("Q6_physics"),
            "vlm_Q7_size_consistency": vlm_details.get("Q7_size_consistency"),
            "vlm_Q8_appearance_consistency": vlm_details.get("Q8_appearance_consistency"),
            "vlm_Q9_motion_continuity": vlm_details.get("Q9_motion_continuity"),
            # Tier1 D1
            "D1_mean_flow": t1_d1.get("mean_flow"),
            "D1_static_ratio": t1_d1.get("static_ratio"),
            "D1_is_dynamic": t1_d1.get("is_dynamic"),
            # Tier1 D2
            "D2_motion_smoothness": t1_d2.get("motion_smoothness"),
            # Tier1 D3
            "D3_temporal_flickering": t1_d3.get("temporal_flickering"),
            # Tier1 D4
            "D4_subject_consistency": t1_d4.get("subject_consistency"),
            # Tier1 D5
            "D5_flow_acceleration": t1_d5.get("flow_acceleration"),
            "D5_flow_spatial_var": t1_d5.get("flow_spatial_var"),
            # Bad reasons
            "bad_reasons": "; ".join(bad_reasons) if bad_reasons else "",
        }
        records.append(record)

    # Sort by video name (numeric)
    records.sort(key=lambda r: int(r["video"].replace(".mp4", "")) if r["video"].replace(".mp4", "").isdigit() else 0)

    # Save JSON
    json_path = os.path.join(args.output_dir, "evaluation_full.json")
    with open(json_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON: %s", json_path)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "evaluation_full.csv")
    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    logger.info("Saved CSV: %s", csv_path)

    # Summary
    good = [r for r in records if r["label"] == "good"]
    bad = [r for r in records if r["label"] == "bad"]
    error = [r for r in records if r["label"] == "error"]

    print(f"\n{'='*60}")
    print(f"  Total:  {len(records)}")
    print(f"  Good:   {len(good)} ({100*len(good)/len(records):.1f}%)")
    print(f"  Bad:    {len(bad)} ({100*len(bad)/len(records):.1f}%)")
    if error:
        print(f"  Error:  {len(error)}")
    print(f"  VLM threshold (P{args.percentile:.0f}): {vlm_threshold:.3f}")
    print(f"  Veto: DINO<{args.veto_consistency}, static>={args.veto_static}")

    # Bad reason frequency
    from collections import Counter
    reason_counts = Counter()
    for r in bad:
        for reason in r["bad_reasons"].split("; "):
            if reason:
                reason_counts[reason] += 1
    if reason_counts:
        print(f"\n  坏特征分布 (在 {len(bad)} 条 bad 中):")
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count} ({100*count/len(bad):.0f}%)")

    # Tier1 coverage
    t1_fields = ["D1_mean_flow", "D2_motion_smoothness", "D3_temporal_flickering",
                 "D4_subject_consistency", "D5_flow_acceleration"]
    print(f"\n  Tier1 覆盖率:")
    for field in t1_fields:
        covered = sum(1 for r in records if r.get(field) is not None)
        print(f"    {field}: {covered}/{len(records)}")

    print(f"\n  输出文件:")
    print(f"    {json_path}")
    print(f"    {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
