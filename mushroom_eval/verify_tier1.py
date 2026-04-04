"""Verify Tier 1 metric reliability by cross-checking with VLM (Gemini Pro).

Workflow:
1. Sample N videos (diverse scores from tier1 results, or random)
2. Compute Tier 1 metrics on them
3. Ask VLM to watch each video and independently rate the same dimensions
4. Compare numerical scores vs VLM judgments

Usage:
    GEMINI_API_KEY=xxx python -m mushroom_eval.verify_tier1 \
        --video_dir mushroom_data/videos \
        --sample_size 15 \
        --output_dir mushroom_eval_results/verification
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFY_PROMPT = """你是一个专业的视频质量评估专家。请仔细观看这段AI生成的视频（主角是一个叫蘑菇TUTU的毛绒蘑菇精灵），然后对以下5个维度分别打分（1-10分）并给出简短理由。

**评分维度：**

1. **运动活跃度** (1=完全静止/极缓慢, 10=动作丰富活跃)
   - 主体是否有明显的运动？运动幅度大还是小？

2. **运动平滑度** (1=严重卡顿/跳变, 10=非常流畅自然)
   - 动作是否连贯？是否有突然的跳变或不自然的卡顿？

3. **时间一致性** (1=严重闪烁/突变, 10=帧间非常稳定)
   - 画面是否有闪烁？颜色/纹理是否在帧间突变？

4. **外观一致性** (1=主体外观严重漂移/突变, 10=始终一致)
   - 蘑菇TUTU的外形、颜色、大小是否在视频中保持一致？

5. **运动连贯性** (1=运动轨迹混乱/突然变向, 10=运动轨迹自然连贯)
   - 运动方向和速度变化是否自然？是否有不合理的突然加速或方向改变？

请严格按以下JSON格式回答（不要添加其他内容）：
```json
{
  "motion_activity": {"score": N, "reason": "..."},
  "motion_smoothness": {"score": N, "reason": "..."},
  "temporal_consistency": {"score": N, "reason": "..."},
  "appearance_consistency": {"score": N, "reason": "..."},
  "motion_continuity": {"score": N, "reason": "..."}
}
```"""


def vlm_verify_video(client: genai.Client, video_path: str, model: str) -> dict[str, Any]:
    """Ask VLM to independently rate a video on tier1-aligned dimensions."""
    video_bytes = Path(video_path).read_bytes()
    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

    try:
        response = client.models.generate_content(
            model=model,
            contents=[video_part, VERIFY_PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        raw = response.text

        # Extract JSON from response (handle markdown code blocks)
        json_str = raw
        if "```json" in raw:
            json_str = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            json_str = raw.split("```")[1].split("```")[0]

        # Try strict JSON parse first
        try:
            parsed = json.loads(json_str.strip())
        except json.JSONDecodeError:
            # Fallback: extract scores with regex
            import re
            parsed = {}
            for dim in ["motion_activity", "motion_smoothness", "temporal_consistency",
                        "appearance_consistency", "motion_continuity"]:
                match = re.search(rf'"{dim}".*?"score":\s*(\d+)', json_str, re.DOTALL)
                if match:
                    parsed[dim] = {"score": int(match.group(1)), "reason": "(regex extracted)"}
            if not parsed:
                logger.warning("VLM verify: no scores extracted for %s", video_path)

        return {"video_path": video_path, "vlm_ratings": parsed, "raw": raw, "status": "success" if parsed else "parse_failed"}
    except Exception as e:
        logger.warning("VLM verify failed for %s: %s", video_path, str(e)[:200])
        return {"video_path": video_path, "vlm_ratings": {}, "raw": str(e), "status": "failed"}


def sample_videos(video_dir: str, tier1_path: str | None, n: int) -> list[str]:
    """Sample videos — stratified by tier1 scores if available, random otherwise."""
    all_videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    if tier1_path and os.path.exists(tier1_path):
        # Stratified: pick from different score ranges
        with open(tier1_path) as f:
            tier1 = json.load(f)
        # Sort by mean_flow (dynamic degree) if available
        scored = [(r["video_path"], r.get("mean_flow", 0)) for r in tier1 if r.get("mean_flow", -1) >= 0]
        if scored:
            scored.sort(key=lambda x: x[1])
            # Pick evenly from low/mid/high
            step = max(1, len(scored) // n)
            sampled = [scored[i * step][0] for i in range(min(n, len(scored)))]
            logger.info("Stratified sample: %d videos by mean_flow", len(sampled))
            return sampled[:n]

    # Random sample
    sampled = random.sample(all_videos, min(n, len(all_videos)))
    logger.info("Random sample: %d videos", len(sampled))
    return sampled


def run_verification(
    video_list: list[str],
    tier1_results: dict[str, dict],
    client: genai.Client,
    model: str,
    output_dir: str,
) -> list[dict]:
    """Run VLM verification and compare with Tier 1 metrics."""
    os.makedirs(output_dir, exist_ok=True)
    verification = []

    for i, vp in enumerate(video_list):
        logger.info("Verifying %d/%d: %s", i + 1, len(video_list), Path(vp).name)

        # Get tier1 metrics for this video
        t1 = tier1_results.get(vp, {})

        # Get VLM ratings
        vlm = vlm_verify_video(client, vp, model)

        # Build comparison
        entry = {
            "video": Path(vp).name,
            "tier1": {
                "mean_flow": t1.get("mean_flow"),
                "static_ratio": t1.get("static_ratio"),
                "is_dynamic": t1.get("is_dynamic"),
                "motion_smoothness": t1.get("motion_smoothness"),
                "temporal_flickering": t1.get("temporal_flickering"),
                "subject_consistency": t1.get("subject_consistency"),
                "flow_acceleration": t1.get("flow_acceleration"),
                "flow_spatial_var": t1.get("flow_spatial_var"),
            },
            "vlm_ratings": vlm.get("vlm_ratings", {}),
            "vlm_raw": vlm.get("raw", ""),
            "vlm_status": vlm.get("status", ""),
        }
        verification.append(entry)

    # Save
    out_path = os.path.join(output_dir, "tier1_verification.json")
    with open(out_path, "w") as f:
        json.dump(verification, f, ensure_ascii=False, indent=2)
    logger.info("Verification saved to %s", out_path)

    # Print comparison table
    print(f"\n{'='*110}")
    print(f"{'Video':>12s} | {'Flow':>6s} {'Dyn':>3s} | {'Smooth':>6s} | {'Flicker':>7s} | {'Consist':>7s} | {'Accel':>6s} || {'VLM-Act':>7s} {'VLM-Smo':>7s} {'VLM-Tem':>7s} {'VLM-App':>7s} {'VLM-Con':>7s}")
    print(f"{'-'*110}")
    for e in verification:
        t = e["tier1"]
        v = e.get("vlm_ratings", {})
        print(
            f"{e['video']:>12s} | "
            f"{t.get('mean_flow', 0) or 0:6.1f} {'Y' if t.get('is_dynamic') else 'N':>3s} | "
            f"{t.get('motion_smoothness', 0) or 0:6.4f} | "
            f"{t.get('temporal_flickering', 0) or 0:7.4f} | "
            f"{t.get('subject_consistency', 0) or 0:7.4f} | "
            f"{t.get('flow_acceleration', 0) or 0:6.2f} || "
            f"{v.get('motion_activity', {}).get('score', '-'):>7} "
            f"{v.get('motion_smoothness', {}).get('score', '-'):>7} "
            f"{v.get('temporal_consistency', {}).get('score', '-'):>7} "
            f"{v.get('appearance_consistency', {}).get('score', '-'):>7} "
            f"{v.get('motion_continuity', {}).get('score', '-'):>7}"
        )
    print(f"{'='*110}")

    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Tier 1 metrics with VLM")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="mushroom_eval_results/verification")
    parser.add_argument("--tier1_file", default=None, help="Path to D1_dynamic_degree.json for stratified sampling")
    parser.add_argument("--sample_size", type=int, default=15)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)

    # Sample videos
    videos = sample_videos(args.video_dir, args.tier1_file, args.sample_size)

    # Run Tier 1 on sampled videos
    logger.info("Running Tier 1 metrics on %d sampled videos...", len(videos))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mushroom_eval.tier1_metrics import run_tier1

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    run_tier1(videos, device=device, output_dir=args.output_dir)

    # Load tier1 results
    tier1_merged_path = os.path.join(args.output_dir, "tier1_results.json")
    with open(tier1_merged_path) as f:
        tier1_data = json.load(f)
    tier1_by_path = {r["video_path"]: r for r in tier1_data}

    # Run VLM verification
    client = genai.Client(api_key=api_key, http_options={"timeout": 120000})
    run_verification(videos, tier1_by_path, client, args.model, args.output_dir)


if __name__ == "__main__":
    main()
