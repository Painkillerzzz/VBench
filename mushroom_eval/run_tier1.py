"""Run Tier 1 signal-based metrics on mushroom character videos.

Usage:
    # Run all metrics:
    python -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos

    # Run only specific metrics (fast first):
    python -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D3
    python -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D1 D5
    python -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --metrics D2 D4

    # Limit for testing:
    python -m mushroom_eval.run_tier1 --video_dir mushroom_data/videos --max_videos 10
"""

import argparse
import glob
import logging
import os
import sys

from .tier1_metrics import run_tier1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier 1 metrics on mushroom videos")
    parser.add_argument("--video_dir", required=True, help="Directory with .mp4 files")
    parser.add_argument("--output_dir", default="mushroom_eval_results", help="Output directory")
    parser.add_argument("--device", default="cuda", help="torch device (cuda/cpu)")
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Which metrics to run (D1 D2 D3 D4 D5). Default: all",
    )
    parser.add_argument("--max_videos", type=int, default=0, help="Limit (0=all)")
    args = parser.parse_args()

    if not os.path.isdir(args.video_dir):
        logger.error("Video directory not found: %s", args.video_dir)
        sys.exit(1)

    # Find videos
    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    logger.info("Found %d videos", len(videos))

    if args.max_videos > 0:
        videos = videos[: args.max_videos]
        logger.info("Limited to %d videos", len(videos))

    if not videos:
        logger.error("No videos found")
        sys.exit(1)

    run_tier1(
        video_list=videos,
        device=args.device,
        output_dir=args.output_dir,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
