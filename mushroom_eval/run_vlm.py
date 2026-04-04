"""Run VLM evaluation on mushroom character videos.

Usage:
    # Evaluate videos in a local directory:
    python -m mushroom_eval.run_vlm --video_dir /path/to/videos --output_dir results/

    # With captions file (JSON: {"video_name.mp4": "caption text", ...}):
    python -m mushroom_eval.run_vlm --video_dir /path/to/videos --caption_file captions.json

    # With S3 sync first:
    aws s3 sync s3://bucket/path /local/path --exclude "*" --include "*.mp4"
    python -m mushroom_eval.run_vlm --video_dir /local/path

    # Adjust concurrency:
    python -m mushroom_eval.run_vlm --video_dir /path --max_concurrent 20
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

from .config import EvalConfig, GeminiConfig
from .vlm_evaluator import evaluate_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_captions(caption_file: str) -> dict[str, str]:
    """Load captions from JSON or JSONL file.

    Supports:
        - JSON dict: {"video_name.mp4": "caption", ...}
        - JSON list: [{"video_path": "path/name.mp4", "caption": "text"}, ...]
        - JSONL: one JSON object per line
        - Also auto-discovers per-video .txt files in the same directory
    """
    if not caption_file or not os.path.exists(caption_file):
        return {}

    path = Path(caption_file)
    captions: dict[str, str] = {}

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                name = obj.get("video") or obj.get("name") or obj.get("video_name") or obj.get("video_path", "")
                text = obj.get("caption") or obj.get("prompt") or obj.get("text", "")
                if name and text:
                    # Ensure key is "X.mp4" format
                    fname = Path(name).stem + ".mp4"
                    captions[fname] = text
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            captions = {Path(k).stem + ".mp4": v for k, v in data.items()}
        elif isinstance(data, list):
            for obj in data:
                name = obj.get("video") or obj.get("name") or obj.get("video_name") or obj.get("video_path", "")
                text = obj.get("caption") or obj.get("prompt") or obj.get("text", "")
                if name and text:
                    fname = Path(name).stem + ".mp4"
                    captions[fname] = text

    # Also try loading per-video .txt files from the same directory
    txt_dir = path.parent
    for txt_file in txt_dir.glob("*.txt"):
        if txt_file.stem.isdigit():
            fname = txt_file.stem + ".mp4"
            if fname not in captions:
                captions[fname] = txt_file.read_text(encoding="utf-8").strip()

    logger.info("Loaded %d captions from %s", len(captions), caption_file)
    return captions


def find_videos(video_dir: str) -> list[str]:
    """Find all mp4 videos in directory (non-recursive)."""
    patterns = ["*.mp4", "*.MP4"]
    videos = []
    for pattern in patterns:
        videos.extend(glob.glob(os.path.join(video_dir, pattern)))
    videos = sorted(set(videos))
    logger.info("Found %d videos in %s", len(videos), video_dir)
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini VLM evaluation on mushroom character videos"
    )
    parser.add_argument(
        "--video_dir", required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--caption_file", default="",
        help="JSON/JSONL file mapping video names to captions",
    )
    parser.add_argument(
        "--output_dir", default="mushroom_eval_results",
        help="Directory for output results (default: mushroom_eval_results)",
    )
    parser.add_argument(
        "--model", default="gemini-3.1-flash-lite-preview",
        help="Gemini model name",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=100,
        help="Max concurrent API requests (default: 100, auto-reduces on rate limit)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Checkpoint save interval (default: 50)",
    )
    parser.add_argument(
        "--max_videos", type=int, default=0,
        help="Max videos to evaluate (0 = all, useful for testing)",
    )
    args = parser.parse_args()

    # Validate
    if not os.path.isdir(args.video_dir):
        logger.error("Video directory not found: %s", args.video_dir)
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        sys.exit(1)

    # Build config
    gemini_config = GeminiConfig(
        model=args.model,
        max_concurrent=args.max_concurrent,
    )
    eval_config = EvalConfig(
        gemini=gemini_config,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        caption_file=args.caption_file,
        batch_size=args.batch_size,
    )

    # Load data
    videos = find_videos(args.video_dir)
    if not videos:
        logger.error("No videos found in %s", args.video_dir)
        sys.exit(1)

    if args.max_videos > 0:
        videos = videos[: args.max_videos]
        logger.info("Limited to first %d videos", args.max_videos)

    captions = load_captions(args.caption_file)

    # Run evaluation
    results = evaluate_videos(videos, captions, eval_config)

    # Quick summary
    valid = [r for r in results if r["vlm_score"] >= 0]
    if valid:
        scores = sorted(r["vlm_score"] for r in valid)
        n = len(scores)
        print(f"\n{'='*60}")
        print(f"VLM Evaluation Complete: {n} videos")
        print(f"{'='*60}")
        print(f"  Mean score:   {sum(scores)/n:.3f}")
        print(f"  Median score: {scores[n//2]:.3f}")
        print(f"  P10 score:    {scores[int(n*0.1)]:.3f}")
        print(f"  P25 score:    {scores[int(n*0.25)]:.3f}")
        print(f"  Min score:    {scores[0]:.3f}")
        print(f"  Max score:    {scores[-1]:.3f}")

        # Suggested bad threshold (bottom 20%)
        threshold = scores[int(n * 0.2)]
        bad_count = sum(1 for s in scores if s <= threshold)
        print(f"\n  Suggested threshold (P20): {threshold:.3f}")
        print(f"  Videos below threshold:    {bad_count}/{n}")
        print(f"\n  Results saved to: {eval_config.output_dir}/vlm_results.json")


if __name__ == "__main__":
    main()
