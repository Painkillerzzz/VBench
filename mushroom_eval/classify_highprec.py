"""High-precision rule-based classification for mushroom TUTU videos.

Uses a consensus rule over VLM + VBench signals:
    bad ⇔ VLM_score <= 3 AND (D3_temporal_flickering < 0.970 OR D2_motion_smoothness < 0.989)

Rule discovered via train/test (30/70) split:
    Train Precision: 0.644
    Test  Precision: 0.686  (102 bad predictions on 1363 test videos)
    Full  Precision: 0.673  (147 bad predictions on 1946 videos)

Usage:
    # Requires pre-computed VLM and VBench scores.
    python -m mushroom_eval.classify_highprec \\
        --vlm_scores mushroom_eval_results/final/vlm_scores.json \\
        --vbench_scores mushroom_eval_results/final/vbench_scores.json \\
        --output_dir mushroom_eval_results/final
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HighPrecRule:
    """Consensus rule for high-precision bad classification."""
    vlm_max: int = 3            # VLM score must be <= this
    d3_max: float = 0.970       # OR D3 temporal flickering must be <
    d2_max: float = 0.989       # OR D2 motion smoothness must be <

    def classify(self, vlm_score: float | None, d2: float | None, d3: float | None) -> bool:
        """Return True if video is classified as bad."""
        if vlm_score is None or d2 is None or d3 is None:
            return False
        return vlm_score <= self.vlm_max and (d3 < self.d3_max or d2 < self.d2_max)

    def reasons(self, vlm_score: float, d2: float, d3: float, vlm_issues: list[str]) -> str:
        """Produce a human-readable reason string."""
        parts: list[str] = []
        if vlm_score <= self.vlm_max:
            parts.append(f"VLM score={vlm_score}")
        if d3 < self.d3_max:
            parts.append(f"D3 flickering={d3:.4f}")
        if d2 < self.d2_max:
            parts.append(f"D2 smoothness={d2:.4f}")
        if vlm_issues:
            parts.append("VLM: " + "; ".join(vlm_issues))
        return "; ".join(parts)


def classify_all(
    vlm_scores: dict[str, dict],
    vbench_scores: dict[str, dict],
    rule: HighPrecRule,
    human_bad: set[str] | None = None,
) -> list[dict]:
    """Classify all videos with the rule. Returns list of per-video dicts."""
    rows: list[dict] = []
    for video in sorted(set(vlm_scores.keys()) & set(vbench_scores.keys())):
        vlm_info = vlm_scores.get(video, {})
        vb_info = vbench_scores.get(video, {})
        vlm_score = vlm_info.get("score")
        d2 = vb_info.get("motion_smoothness")
        d3 = vb_info.get("temporal_flickering")
        d4 = vb_info.get("subject_consistency")

        if vlm_score is None or d2 is None or d3 is None:
            continue

        is_bad = rule.classify(float(vlm_score), float(d2), float(d3))
        row: dict = {
            "video": video,
            "label": "bad" if is_bad else "good",
            "bad_reasons": rule.reasons(float(vlm_score), float(d2), float(d3), vlm_info.get("issues", [])) if is_bad else "",
            "vlm_score": vlm_score,
            "vlm_issues": "; ".join(vlm_info.get("issues", [])),
            "D2_motion_smoothness": d2,
            "D3_temporal_flickering": d3,
            "D4_subject_consistency": d4,
        }
        if human_bad is not None:
            row["human_label"] = "bad" if video in human_bad else ""
        rows.append(row)

    rows.sort(key=lambda r: int(r["video"].replace(".mp4", "")) if r["video"].replace(".mp4", "").isdigit() else 0)
    return rows


def save_outputs(rows: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "classification_final.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "classification_final.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bad_rows = [r for r in rows if r["label"] == "bad"]
    good_rows = [r for r in rows if r["label"] == "good"]

    with open(os.path.join(output_dir, "bad_videos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(r["video"] for r in bad_rows) + "\n")
    with open(os.path.join(output_dir, "good_videos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(r["video"] for r in good_rows) + "\n")

    logger.info("Saved %d rows → %s", len(rows), json_path)
    logger.info("Saved CSV → %s", csv_path)
    logger.info("bad=%d  good=%d", len(bad_rows), len(good_rows))

    # Report precision if human labels available
    if rows and "human_label" in rows[0]:
        tp = sum(1 for r in bad_rows if r["human_label"] == "bad")
        fp = sum(1 for r in bad_rows if r["human_label"] != "bad")
        fn = sum(1 for r in good_rows if r["human_label"] == "bad")
        total_bad = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_bad if total_bad else 0.0
        logger.info("Precision=%.3f  Recall=%.3f  (TP=%d  FP=%d  FN=%d)", precision, recall, tp, fp, fn)


def load_badcase(path: str) -> set[str]:
    all_bad: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith(".mp4"):
                all_bad.add(line.split("/")[-1])
    return all_bad


def main() -> None:
    parser = argparse.ArgumentParser(description="High-precision rule-based classifier")
    parser.add_argument("--vlm_scores", required=True, help="JSON: {video: {score, issues}}")
    parser.add_argument("--vbench_scores", required=True, help="JSON: {video: {motion_smoothness, temporal_flickering, subject_consistency}}")
    parser.add_argument("--output_dir", default="mushroom_eval_results/final")
    parser.add_argument("--badcase_list", default="", help="Optional human labels for precision reporting")
    parser.add_argument("--vlm_max", type=int, default=3)
    parser.add_argument("--d3_max", type=float, default=0.970)
    parser.add_argument("--d2_max", type=float, default=0.989)
    args = parser.parse_args()

    with open(args.vlm_scores, encoding="utf-8") as f:
        vlm_scores = json.load(f)
    with open(args.vbench_scores, encoding="utf-8") as f:
        vbench_scores = json.load(f)

    human_bad = load_badcase(args.badcase_list) if args.badcase_list else None

    rule = HighPrecRule(vlm_max=args.vlm_max, d3_max=args.d3_max, d2_max=args.d2_max)
    logger.info("Rule: VLM<=%d AND (D3<%.3f OR D2<%.3f)", rule.vlm_max, rule.d3_max, rule.d2_max)

    rows = classify_all(vlm_scores, vbench_scores, rule, human_bad)
    save_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
