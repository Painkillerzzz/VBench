"""Verify Tier 1 metrics: ask VLM one question per dimension, score 1-10.

Instead of one big JSON response, ask 5 separate simple questions
to avoid parsing failures.

Usage:
    GEMINI_API_KEY=xxx python -m mushroom_eval.verify_tier1_v2 \
        --video_dir mushroom_data/videos \
        --sample_size 20
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Each question maps to a tier1 metric. Answer is just a number 1-10.
DIMENSION_QUESTIONS = [
    {
        "id": "motion_activity",
        "tier1_key": "mean_flow",
        "question": "Watch this video of 蘑菇TUTU. Rate the CHARACTER's motion activity level on a scale from 1 to 10, where 1 means completely still/frozen and 10 means very active with rich movements. Reply with ONLY a single integer between 1 and 10, nothing else.",
    },
    {
        "id": "motion_smoothness",
        "tier1_key": "motion_smoothness",
        "question": "Watch this video of 蘑菇TUTU. Rate the smoothness and fluidity of motion on a scale from 1 to 10, where 1 means severe stuttering/frame jumps and 10 means perfectly smooth natural motion. Reply with ONLY a single integer between 1 and 10, nothing else.",
    },
    {
        "id": "temporal_consistency",
        "tier1_key": "temporal_flickering",
        "question": "Watch this video of 蘑菇TUTU. Rate the frame-to-frame temporal stability on a scale from 1 to 10, where 1 means severe flickering/color shifts between frames and 10 means perfectly stable. Reply with ONLY a single integer between 1 and 10, nothing else.",
    },
    {
        "id": "appearance_consistency",
        "tier1_key": "subject_consistency",
        "question": "Watch this video of 蘑菇TUTU. Rate how consistent the character's appearance (shape, color, size, texture) remains throughout the video on a scale from 1 to 10, where 1 means extreme changes/morphing and 10 means perfectly consistent. Reply with ONLY a single integer between 1 and 10, nothing else.",
    },
    {
        "id": "motion_continuity",
        "tier1_key": "flow_acceleration",
        "question": "Watch this video of 蘑菇TUTU. Rate the continuity and naturalness of motion trajectories on a scale from 1 to 10, where 1 means chaotic motion with sudden teleportation/direction changes and 10 means perfectly smooth natural trajectories. Reply with ONLY a single integer between 1 and 10, nothing else.",
    },
]


def ask_one_dimension(
    client: genai.Client,
    model: str,
    video_bytes: bytes,
    question: str,
) -> int | None:
    """Ask VLM a single dimension question, return score 1-10 or None."""
    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[video_part, question],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16),
        )
        text = response.text.strip()
        match = re.search(r'\b(\d{1,2})\b', text)
        if match:
            val = int(match.group(1))
            if 1 <= val <= 10:
                return val
        logger.warning("Could not parse score from: %s", text[:50])
        return None
    except Exception as e:
        logger.warning("VLM error: %s", str(e)[:100])
        return None


def run_verification(
    video_list: list[str],
    tier1_by_path: dict[str, dict],
    client: genai.Client,
    model: str,
    output_dir: str,
) -> list[dict]:
    """Run per-dimension VLM verification."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, vp in enumerate(video_list):
        logger.info("Verifying %d/%d: %s", i + 1, len(video_list), Path(vp).name)
        video_bytes = Path(vp).read_bytes()
        t1 = tier1_by_path.get(vp, {})

        vlm_scores = {}
        for dim in DIMENSION_QUESTIONS:
            score = ask_one_dimension(client, model, video_bytes, dim["question"])
            vlm_scores[dim["id"]] = score

        results.append({
            "video": Path(vp).name,
            "video_path": vp,
            "tier1": {k: v for k, v in t1.items() if k != "video_path"},
            "vlm_scores": vlm_scores,
        })

    # Save
    out_path = os.path.join(output_dir, "tier1_verification_v2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print comparison table
    print(f"\n{'='*130}")
    print(f"{'Video':>12s} | {'D1:Flow':>8s} {'VLM-Act':>7s} | {'D2:Smooth':>9s} {'VLM-Smo':>7s} | {'D3:Flicker':>10s} {'VLM-Tem':>7s} | {'D4:Consist':>10s} {'VLM-App':>7s} | {'D5:Accel':>8s} {'VLM-Con':>7s}")
    print(f"{'-'*130}")
    for e in results:
        t = e["tier1"]
        v = e["vlm_scores"]
        print(
            f"{e['video']:>12s} | "
            f"{t.get('mean_flow', 0) or 0:8.1f} {str(v.get('motion_activity', '-')):>7s} | "
            f"{t.get('motion_smoothness', 0) or 0:9.4f} {str(v.get('motion_smoothness', '-')):>7s} | "
            f"{t.get('temporal_flickering', 0) or 0:10.4f} {str(v.get('temporal_consistency', '-')):>7s} | "
            f"{t.get('subject_consistency', 0) or 0:10.4f} {str(v.get('appearance_consistency', '-')):>7s} | "
            f"{t.get('flow_acceleration', 0) or 0:8.2f} {str(v.get('motion_continuity', '-')):>7s}"
        )
    print(f"{'='*130}")

    # Compute correlation for each dimension
    print(f"\n--- 相关性分析 ---")
    for dim in DIMENSION_QUESTIONS:
        t1_key = dim["tier1_key"]
        vlm_key = dim["id"]
        pairs = []
        for e in results:
            t1_val = e["tier1"].get(t1_key)
            vlm_val = e["vlm_scores"].get(vlm_key)
            if t1_val is not None and vlm_val is not None and t1_val >= 0:
                pairs.append((float(t1_val), float(vlm_val)))
        if len(pairs) >= 5:
            import numpy as np
            t1_arr = np.array([p[0] for p in pairs])
            vlm_arr = np.array([p[1] for p in pairs])
            # For flow_acceleration, higher = worse, so negate for correlation
            if t1_key == "flow_acceleration":
                corr = np.corrcoef(-t1_arr, vlm_arr)[0, 1]
            # For temporal_flickering/motion_smoothness/subject_consistency, higher = better
            else:
                corr = np.corrcoef(t1_arr, vlm_arr)[0, 1]
            print(f"  {vlm_key:.<30s} Pearson r = {corr:+.3f}  (n={len(pairs)})")
        else:
            print(f"  {vlm_key:.<30s} 数据不足 (n={len(pairs)})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Tier 1 with per-dimension VLM questions")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="mushroom_eval_results/verification_v2")
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--tier1_dir", default=None, help="Dir with D1-D5 json files (for stratified sampling)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)

    # Sample videos
    all_videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    videos = random.sample(all_videos, min(args.sample_size, len(all_videos)))
    logger.info("Sampled %d videos", len(videos))

    # Run Tier 1 on sampled videos
    logger.info("Running Tier 1 metrics on sampled videos...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mushroom_eval.tier1_metrics import run_tier1
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_tier1(videos, device=device, output_dir=args.output_dir)

    # Load merged results
    with open(os.path.join(args.output_dir, "tier1_results.json")) as f:
        tier1_data = json.load(f)
    tier1_by_path = {r["video_path"]: r for r in tier1_data}

    # Run VLM verification
    client = genai.Client(api_key=api_key, http_options={"timeout": 120000})
    run_verification(videos, tier1_by_path, client, args.model, args.output_dir)


if __name__ == "__main__":
    main()
