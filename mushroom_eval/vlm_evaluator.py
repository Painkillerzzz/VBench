"""Gemini-based VLM evaluator for mushroom character videos.

Features:
- True async parallelism via google-genai async client (client.aio)
- Configurable concurrency (default 100)
- Adaptive rate limit handling with exponential backoff
- Automatic retry with failure tracking
- Checkpoint/resume support
- Final retry pass for all failed videos
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import (
    EvalConfig,
    GeminiConfig,
    VLM_QUESTIONS,
    VLM_QUESTION_PROMPT,
    VLM_SYSTEM_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_vlm_response(text: str) -> dict[str, str]:
    """Parse VLM response into question_id -> yes/no mapping."""
    answers: dict[str, str] = {}
    pattern = re.compile(r"Q?(\d+)[:\.\s]+\s*(yes|no)", re.IGNORECASE)
    for match in pattern.finditer(text):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(VLM_QUESTIONS):
            q_id = VLM_QUESTIONS[idx][0]
            answers[q_id] = match.group(2).lower()
    return answers


def _compute_vlm_score(answers: dict[str, str]) -> tuple[float, dict[str, float]]:
    """Compute overall VLM score and per-question scores."""
    per_question: dict[str, float] = {}
    total = 0.0
    answered = 0

    for q_id, _text, is_positive in VLM_QUESTIONS:
        if q_id in answers:
            ans = answers[q_id]
            if is_positive:
                score = 1.0 if ans == "yes" else 0.0
            else:
                score = 0.0 if ans == "yes" else 1.0
            per_question[q_id] = score
            total += score
            answered += 1
        else:
            per_question[q_id] = -1.0  # unanswered

    overall = total / answered if answered > 0 else 0.0
    return overall, per_question


def _build_prompt(caption: str = "") -> str:
    """Build the full prompt string."""
    caption_context = (
        f'\nThe video was generated with this prompt: "{caption}"'
        if caption else ""
    )
    return (
        f"{VLM_SYSTEM_PROMPT}{caption_context}\n\n"
        f"Please answer the following questions about this video:\n"
        f"{VLM_QUESTION_PROMPT}"
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        "429", "rate limit", "resource exhausted", "quota",
        "too many requests", "rate_limit",
    ])


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit, server error, timeout)."""
    msg = str(exc).lower()
    return _is_rate_limit_error(exc) or any(keyword in msg for keyword in [
        "500", "502", "503", "504", "internal", "unavailable",
        "timeout", "deadline", "connection",
    ])


class AdaptiveConcurrencyController:
    """Dynamically adjusts concurrency based on rate limit feedback."""

    def __init__(self, initial: int = 100, min_concurrent: int = 5):
        self.current = initial
        self.min_concurrent = min_concurrent
        self._semaphore = asyncio.Semaphore(initial)
        self._lock = asyncio.Lock()
        self.rate_limit_hits = 0
        self.total_requests = 0
        self.total_success = 0

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    async def on_rate_limit(self) -> None:
        """Called when a rate limit error is detected. Reduces concurrency."""
        async with self._lock:
            self.rate_limit_hits += 1
            new_val = max(self.min_concurrent, self.current // 2)
            if new_val < self.current:
                logger.warning(
                    "Rate limit hit (#%d). Reducing concurrency: %d -> %d",
                    self.rate_limit_hits, self.current, new_val,
                )
                self.current = new_val
                # Recreate semaphore with lower limit
                self._semaphore = asyncio.Semaphore(new_val)

    def on_success(self) -> None:
        self.total_success += 1


async def _evaluate_single_async(
    controller: AdaptiveConcurrencyController,
    async_client: Any,  # genai.Client.aio
    video_path: str,
    config: GeminiConfig,
    caption: str = "",
) -> dict[str, Any]:
    """Evaluate a single video with adaptive concurrency and retry."""
    prompt = _build_prompt(caption)

    # Read video bytes once
    video_bytes = Path(video_path).read_bytes()
    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

    last_error = ""
    for attempt in range(config.max_retries):
        await controller.acquire()
        try:
            response = await async_client.models.generate_content(
                model=config.model,
                contents=[video_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
            raw_text = response.text
            answers = _parse_vlm_response(raw_text)

            # Validate: we should get at least 5 out of 9 answers
            if len(answers) < 5:
                logger.warning(
                    "Incomplete response for %s (%d/%d answers). Raw: %s",
                    Path(video_path).name, len(answers), len(VLM_QUESTIONS),
                    raw_text[:200],
                )
                # Still use what we got if >= 3 answers
                if len(answers) < 3:
                    last_error = f"too few answers ({len(answers)})"
                    controller.release()
                    await asyncio.sleep(config.retry_delay)
                    continue

            overall_score, per_question = _compute_vlm_score(answers)
            controller.on_success()
            controller.release()

            return {
                "video_path": video_path,
                "vlm_score": overall_score,
                "vlm_details": per_question,
                "vlm_answers": answers,
                "vlm_raw": raw_text,
                "status": "success",
            }

        except Exception as e:
            controller.release()
            last_error = str(e)

            if _is_rate_limit_error(e):
                await controller.on_rate_limit()
                wait_time = config.retry_delay * (2 ** attempt) + 5.0
                logger.warning(
                    "Rate limit for %s (attempt %d/%d). Waiting %.1fs",
                    Path(video_path).name, attempt + 1, config.max_retries,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            elif _is_retryable_error(e):
                wait_time = config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Retryable error for %s (attempt %d/%d): %s. Waiting %.1fs",
                    Path(video_path).name, attempt + 1, config.max_retries,
                    str(e)[:100], wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "Non-retryable error for %s: %s",
                    Path(video_path).name, str(e)[:200],
                )
                break

    logger.error("All retries exhausted for %s: %s", Path(video_path).name, last_error[:200])
    return {
        "video_path": video_path,
        "vlm_score": -1.0,
        "vlm_details": {},
        "vlm_answers": {},
        "vlm_raw": f"ERROR: {last_error}",
        "status": "failed",
    }


def _save_checkpoint(
    results: list[dict[str, Any]],
    checkpoint_path: str,
    total: int,
) -> None:
    """Save checkpoint atomically."""
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)

    success = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    logger.info(
        "Checkpoint: %d/%d done (success=%d, failed=%d)",
        len(results), total, success, failed,
    )


async def evaluate_batch_async(
    video_list: list[str],
    captions: dict[str, str],
    config: GeminiConfig,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 100,
) -> list[dict[str, Any]]:
    """Evaluate videos with true async parallelism and checkpointing."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)
    controller = AdaptiveConcurrencyController(
        initial=config.max_concurrent, min_concurrent=5,
    )

    # Load existing checkpoint
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            existing = json.load(f)
        for r in existing:
            # Only skip if previously succeeded
            if r.get("status") == "success":
                completed[r["video_path"]] = r
        logger.info("Loaded %d successful results from checkpoint", len(completed))

    remaining = [v for v in video_list if v not in completed]
    logger.info(
        "Total: %d, already done: %d, remaining: %d",
        len(video_list), len(completed), len(remaining),
    )

    results = list(completed.values())
    start_time = time.time()

    # Process all remaining videos concurrently (semaphore controls parallelism)
    # But checkpoint in batches
    for batch_start in range(0, len(remaining), checkpoint_interval):
        batch = remaining[batch_start: batch_start + checkpoint_interval]
        tasks = [
            _evaluate_single_async(
                controller, client.aio, vp, config,
                captions.get(Path(vp).name, ""),
            )
            for vp in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in batch_results:
            if isinstance(r, Exception):
                logger.error("Unexpected gather exception: %s", r)
                continue
            results.append(r)

        if checkpoint_path:
            _save_checkpoint(results, checkpoint_path, len(video_list))

        elapsed = time.time() - start_time
        done = len(results)
        if done > 0:
            rate = done / elapsed
            eta = (len(video_list) - done) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f/s, ETA: %.0fs). Concurrency: %d",
                done, len(video_list), rate, eta, controller.current,
            )

    # === RETRY PASS: retry all failed videos ===
    failed_results = [r for r in results if r.get("status") == "failed"]
    if failed_results:
        logger.info(
            "=== RETRY PASS: %d failed videos ===", len(failed_results),
        )
        # Remove failed from results
        results = [r for r in results if r.get("status") != "failed"]
        failed_paths = [r["video_path"] for r in failed_results]

        # Reset controller with lower concurrency for retry
        retry_controller = AdaptiveConcurrencyController(
            initial=max(10, config.max_concurrent // 4), min_concurrent=3,
        )
        retry_config = GeminiConfig(
            model=config.model,
            max_concurrent=max(10, config.max_concurrent // 4),
            max_retries=config.max_retries + 2,  # more retries
            retry_delay=config.retry_delay * 2,   # longer backoff
        )

        retry_tasks = [
            _evaluate_single_async(
                retry_controller, client.aio, vp, retry_config,
                captions.get(Path(vp).name, ""),
            )
            for vp in failed_paths
        ]
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

        for r in retry_results:
            if isinstance(r, Exception):
                logger.error("Retry gather exception: %s", r)
                continue
            results.append(r)

        if checkpoint_path:
            _save_checkpoint(results, checkpoint_path, len(video_list))

        final_failed = sum(1 for r in results if r.get("status") == "failed")
        logger.info(
            "After retry: %d still failed out of %d total",
            final_failed, len(video_list),
        )

    return results


def evaluate_videos(
    video_list: list[str],
    captions: dict[str, str],
    config: EvalConfig,
) -> list[dict[str, Any]]:
    """Main entry point: evaluate videos using Gemini VLM."""
    checkpoint_path = os.path.join(config.output_dir, "vlm_checkpoint.json")
    os.makedirs(config.output_dir, exist_ok=True)

    results = asyncio.run(
        evaluate_batch_async(
            video_list=video_list,
            captions=captions,
            config=config.gemini,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=config.batch_size,
        )
    )

    # Save final results
    output_path = os.path.join(config.output_dir, "vlm_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("VLM results saved to %s", output_path)

    # Print summary
    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    if success:
        scores = [r["vlm_score"] for r in success]
        logger.info(
            "VLM Summary: %d success, %d failed, mean=%.3f, min=%.3f, max=%.3f",
            len(success), len(failed),
            sum(scores) / len(scores), min(scores), max(scores),
        )

    # Save failed list for manual inspection
    if failed:
        failed_path = os.path.join(config.output_dir, "vlm_failed.json")
        with open(failed_path, "w") as f:
            json.dump(
                [{"video_path": r["video_path"], "error": r["vlm_raw"]}
                 for r in failed],
                f, ensure_ascii=False, indent=2,
            )
        logger.warning(
            "%d videos failed. See %s", len(failed), failed_path,
        )

    return results
