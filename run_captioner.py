"""Run GeminiCaptioner on all v-0331 videos with checkpoint/resume support.

Usage:
    GEMINI_API_KEY=xxx python run_captioner.py --video_dir mushroom_data/videos --num_workers 100
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
from google import genai
from google.genai import types

from captioning.prompts import CAPTION_PROMPT_VIDEO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def caption_one_video(client: genai.Client, model: str, video_path: str) -> str:
    """Caption a single video using Gemini."""
    response = client.models.generate_content(
        model=model,
        contents=types.Content(
            parts=[
                types.Part(text=CAPTION_PROMPT_VIDEO),
                types.Part(
                    inline_data=types.Blob(
                        data=open(video_path, "rb").read(),
                        mime_type="video/mp4",
                    ),
                    video_metadata=types.VideoMetadata(fps=5),
                ),
            ]
        ),
    )
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption mushroom videos with Gemini")
    parser.add_argument("--video_dir", required=True, help="Directory with .mp4 files")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: same as video_dir)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Gemini model")
    parser.add_argument("--num_workers", type=int, default=100, help="Parallel workers")
    parser.add_argument("--max_videos", type=int, default=0, help="Limit (0=all)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir) if args.output_dir else video_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all videos
    video_paths = sorted(video_dir.glob("*.mp4"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    logger.info("Found %d videos in %s", len(video_paths), video_dir)

    if args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]

    # Load checkpoint
    checkpoint_path = output_dir / "captions.json"
    completed: dict[str, str] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            existing = json.load(f)
        for item in existing:
            if item.get("caption"):
                vname = Path(item["video_path"]).name
                completed[vname] = item["caption"]
        logger.info("Loaded %d existing captions from checkpoint", len(completed))

    # Filter remaining
    remaining = [p for p in video_paths if p.name not in completed]
    logger.info("Total: %d, done: %d, remaining: %d", len(video_paths), len(completed), len(remaining))

    if not remaining:
        logger.info("All videos already captioned!")
        return

    # Create client
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 120000},
    )

    # Run parallel captioning
    results: dict[str, str] = dict(completed)  # start with existing
    failed: list[str] = []
    save_interval = 50

    def _run_one(vpath: Path) -> tuple[str, str]:
        caption = caption_one_video(client, args.model, str(vpath))
        return vpath.name, caption

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_run_one, vp): vp for vp in remaining}
        done_count = 0

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Captioning"):
            vp = futures[future]
            try:
                name, caption = future.result()
                results[name] = caption
                done_count += 1

                # Also save individual txt
                txt_path = output_dir / (vp.stem + ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

            except Exception as e:
                logger.warning("Failed %s: %s", vp.name, str(e)[:200])
                failed.append(str(vp))
                done_count += 1

            # Periodic checkpoint
            if done_count % save_interval == 0:
                _save_checkpoint(results, video_paths, checkpoint_path)
                logger.info(
                    "Checkpoint: %d/%d done, %d failed",
                    len(results), len(video_paths), len(failed),
                )

    # Final save
    _save_checkpoint(results, video_paths, checkpoint_path)

    # Retry failed
    if failed:
        logger.info("=== RETRY PASS: %d failed videos ===", len(failed))
        retry_paths = [Path(p) for p in failed]
        failed2 = []
        with ThreadPoolExecutor(max_workers=max(10, args.num_workers // 4)) as executor:
            futures = {executor.submit(_run_one, vp): vp for vp in retry_paths}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Retry"):
                vp = futures[future]
                try:
                    name, caption = future.result()
                    results[name] = caption
                    txt_path = output_dir / (vp.stem + ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                except Exception as e:
                    logger.error("Retry failed %s: %s", vp.name, str(e)[:200])
                    failed2.append(str(vp))

        _save_checkpoint(results, video_paths, checkpoint_path)
        if failed2:
            failed_path = output_dir / "caption_failed.json"
            with open(failed_path, "w") as f:
                json.dump(failed2, f, indent=2)
            logger.warning("%d videos still failed. See %s", len(failed2), failed_path)

    logger.info("Done! %d/%d captioned. Results: %s", len(results), len(video_paths), checkpoint_path)


def _save_checkpoint(
    results: dict[str, str],
    video_paths: list[Path],
    checkpoint_path: Path,
) -> None:
    """Save captions.json atomically."""
    data = []
    for vp in video_paths:
        caption = results.get(vp.name)
        data.append({
            "video_path": str(vp),
            "caption": caption,
        })
    tmp = str(checkpoint_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(checkpoint_path))


if __name__ == "__main__":
    main()
