"""Tier 1: Signal-based metrics reusing VBench implementations.

Metrics:
  D1 - dynamic_degree: Optical flow magnitude (RAFT) → detects static/slow videos
  D2 - motion_smoothness: Frame interpolation error (AMT) → detects jerky motion
  D3 - temporal_flickering: Frame-to-frame MAE → detects frame jumps
  D4 - subject_consistency: DINO feature similarity → detects appearance drift
  D5 - flow_stability: Flow acceleration & spatial variance → detects motion discontinuity
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

# Add VBench to path
VBENCH_ROOT = str(Path(__file__).resolve().parent.parent)
if VBENCH_ROOT not in sys.path:
    sys.path.insert(0, VBENCH_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/.cache/vbench"))


# ── D1: Dynamic Degree (continuous score, not binary) ────────────────────

def _load_raft(device: str) -> Any:
    """Load RAFT optical flow model."""
    from vbench.third_party.RAFT.core.raft import RAFT

    model_path = f"{CACHE_DIR}/raft_model/models/raft-things.pth"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"RAFT model not found at {model_path}. "
            "Run VBench init_submodules(['dynamic_degree']) first to download it."
        )
    args = edict(model=model_path, small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)
    ckpt = torch.load(args.model, map_location="cpu")
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt)
    model.to(device)
    model.eval()
    return model


def _read_frames_for_flow(video_path: str, device: str, target_fps: int = 8) -> list[torch.Tensor]:
    """Read video frames, subsample to target_fps, return as tensors on device."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / target_fps))
    frames = []
    idx = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if idx % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frames.append(tensor[None].to(device))
        idx += 1
    video.release()
    return frames


def compute_dynamic_degree_continuous(
    raft_model: Any,
    video_list: list[str],
    device: str,
) -> list[dict[str, Any]]:
    """D1: Compute continuous dynamic degree score (mean flow magnitude).

    Returns per-video:
      - mean_flow: average optical flow magnitude (higher = more motion)
      - static_ratio: fraction of frame pairs with low flow (higher = more static)
      - is_dynamic: bool (VBench-compatible binary)
    """
    from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

    results = []
    for video_path in tqdm(video_list, desc="D1:dynamic_degree"):
        try:
            frames = _read_frames_for_flow(video_path, device)
            if len(frames) < 2:
                results.append({"video_path": video_path, "mean_flow": 0.0, "static_ratio": 1.0, "is_dynamic": False})
                continue

            scale = min(list(frames[0].shape)[-2:])
            threshold = 6.0 * (scale / 256.0)

            flow_magnitudes = []
            with torch.no_grad():
                for img1, img2 in zip(frames[:-1], frames[1:]):
                    padder = InputPadder(img1.shape)
                    img1_p, img2_p = padder.pad(img1, img2)
                    _, flow_up = raft_model(img1_p, img2_p, iters=20, test_mode=True)
                    # Compute magnitude of top 5% pixels
                    flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    rad = np.sqrt(flo[:, :, 0] ** 2 + flo[:, :, 1] ** 2)
                    h, w = rad.shape
                    cut = int(h * w * 0.05)
                    top5 = np.mean(np.sort(rad.flatten())[-cut:]) if cut > 0 else 0.0
                    flow_magnitudes.append(top5)

            mean_flow = float(np.mean(flow_magnitudes))
            static_ratio = float(np.mean([1.0 if m < threshold else 0.0 for m in flow_magnitudes]))
            count_num = round(4 * (len(frames) / 16.0))
            is_dynamic = sum(1 for m in flow_magnitudes if m > threshold) >= count_num

            results.append({
                "video_path": video_path,
                "mean_flow": mean_flow,
                "static_ratio": static_ratio,
                "is_dynamic": is_dynamic,
            })
        except Exception as e:
            logger.warning("D1 failed for %s: %s", video_path, e)
            results.append({"video_path": video_path, "mean_flow": -1.0, "static_ratio": -1.0, "is_dynamic": False})

    return results


# ── D2: Motion Smoothness ────────────────────────────────────────────────

def _load_amt(device: str) -> Any:
    """Load AMT frame interpolation model."""
    from vbench.motion_smoothness import MotionSmoothness

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    vbench_dir = os.path.join(VBENCH_ROOT, "vbench")
    config = os.path.join(vbench_dir, "third_party", "amt", "cfgs", "AMT-S.yaml")
    ckpt = f"{CACHE_DIR}/amt_model/amt-s.pth"

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"AMT model not found at {ckpt}. "
            "Run VBench init_submodules(['motion_smoothness']) first."
        )
    return MotionSmoothness(config, ckpt, device)


def compute_motion_smoothness_scores(
    amt_model: Any,
    video_list: list[str],
) -> list[dict[str, Any]]:
    """D2: Compute motion smoothness (0-1, higher = smoother)."""
    results = []
    for video_path in tqdm(video_list, desc="D2:motion_smoothness"):
        try:
            score = amt_model.motion_score(video_path)
            results.append({"video_path": video_path, "motion_smoothness": float(score)})
        except Exception as e:
            logger.warning("D2 failed for %s: %s", video_path, e)
            results.append({"video_path": video_path, "motion_smoothness": -1.0})
    return results


# ── D3: Temporal Flickering ──────────────────────────────────────────────

def compute_temporal_flickering_scores(
    video_list: list[str],
) -> list[dict[str, Any]]:
    """D3: Compute temporal flickering (0-1, higher = less flickering). No model needed."""
    from vbench.temporal_flickering import cal_score

    results = []
    for video_path in tqdm(video_list, desc="D3:temporal_flickering"):
        try:
            score = cal_score(video_path)
            results.append({"video_path": video_path, "temporal_flickering": float(score)})
        except Exception as e:
            logger.warning("D3 failed for %s: %s", video_path, e)
            results.append({"video_path": video_path, "temporal_flickering": -1.0})
    return results


# ── D4: Subject Consistency ──────────────────────────────────────────────

def _load_dino(device: str) -> Any:
    """Load DINO ViT-B16 model."""
    try:
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device)
    except Exception:
        # Fallback to local if github is unreachable
        local_path = f"{CACHE_DIR}/dino_model/facebookresearch_dino_main/"
        if os.path.isdir(local_path):
            model = torch.hub.load(
                local_path,
                "dino_vitb16",
                source="local",
            ).to(device)
        else:
            raise
    model.eval()
    return model


def compute_subject_consistency_scores(
    dino_model: Any,
    video_list: list[str],
    device: str,
) -> list[dict[str, Any]]:
    """D4: Compute subject consistency (0-1, higher = more consistent)."""
    from vbench.subject_consistency import subject_consistency

    results_raw = []
    video_results = []
    for video_path in tqdm(video_list, desc="D4:subject_consistency"):
        try:
            # Call the core function for single video
            _, single_results = subject_consistency(dino_model, [video_path], device, read_frame=False)
            score = single_results[0]["video_results"]
            video_results.append({"video_path": video_path, "subject_consistency": float(score)})
        except Exception as e:
            logger.warning("D4 failed for %s: %s", video_path, e)
            video_results.append({"video_path": video_path, "subject_consistency": -1.0})

    return video_results


# ── D5: Flow Stability (NEW) ────────────────────────────────────────────

def compute_flow_stability(
    raft_model: Any,
    video_list: list[str],
    device: str,
) -> list[dict[str, Any]]:
    """D5: Compute flow-based stability metrics.

    - flow_acceleration: max change in flow magnitude between consecutive pairs
      (high = sudden motion change = discontinuity)
    - flow_spatial_var: mean spatial variance of flow field
      (high = incoherent motion = physics issues)
    """
    from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

    results = []
    for video_path in tqdm(video_list, desc="D5:flow_stability"):
        try:
            frames = _read_frames_for_flow(video_path, device)
            if len(frames) < 3:
                results.append({
                    "video_path": video_path,
                    "flow_acceleration": 0.0,
                    "flow_spatial_var": 0.0,
                })
                continue

            flow_mags = []
            flow_vars = []
            with torch.no_grad():
                for img1, img2 in zip(frames[:-1], frames[1:]):
                    padder = InputPadder(img1.shape)
                    img1_p, img2_p = padder.pad(img1, img2)
                    _, flow_up = raft_model(img1_p, img2_p, iters=20, test_mode=True)
                    flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    rad = np.sqrt(flo[:, :, 0] ** 2 + flo[:, :, 1] ** 2)
                    flow_mags.append(float(np.mean(rad)))
                    flow_vars.append(float(np.var(rad)))

            # Acceleration = max absolute difference in consecutive flow magnitudes
            accels = [abs(flow_mags[i + 1] - flow_mags[i]) for i in range(len(flow_mags) - 1)]
            flow_acceleration = float(max(accels)) if accels else 0.0
            flow_spatial_var = float(np.mean(flow_vars))

            results.append({
                "video_path": video_path,
                "flow_acceleration": flow_acceleration,
                "flow_spatial_var": flow_spatial_var,
            })
        except Exception as e:
            logger.warning("D5 failed for %s: %s", video_path, e)
            results.append({
                "video_path": video_path,
                "flow_acceleration": -1.0,
                "flow_spatial_var": -1.0,
            })

    return results


# ── Orchestrator ─────────────────────────────────────────────────────────

def run_tier1(
    video_list: list[str],
    device: str = "cuda",
    output_dir: str = "mushroom_eval_results",
    metrics: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run all Tier 1 metrics and save results.

    Args:
        video_list: List of video paths
        device: torch device
        output_dir: Where to save results
        metrics: Which metrics to run. Default: all.
            Options: "D1", "D2", "D3", "D4", "D5"

    Returns:
        Dict mapping metric name to list of per-video results
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = metrics or ["D1", "D2", "D3", "D4", "D5"]
    all_results: dict[str, list[dict[str, Any]]] = {}

    # D1 + D5 share RAFT model
    raft_model = None
    if "D1" in all_metrics or "D5" in all_metrics:
        logger.info("Loading RAFT model...")
        raft_model = _load_raft(device)

    if "D1" in all_metrics:
        logger.info("Computing D1: dynamic_degree...")
        d1 = compute_dynamic_degree_continuous(raft_model, video_list, device)
        all_results["D1_dynamic_degree"] = d1
        _save_metric(d1, output_dir, "D1_dynamic_degree.json")

    if "D5" in all_metrics:
        logger.info("Computing D5: flow_stability...")
        d5 = compute_flow_stability(raft_model, video_list, device)
        all_results["D5_flow_stability"] = d5
        _save_metric(d5, output_dir, "D5_flow_stability.json")

    # Free RAFT
    if raft_model is not None:
        del raft_model
        torch.cuda.empty_cache()

    if "D3" in all_metrics:
        logger.info("Computing D3: temporal_flickering (CPU)...")
        d3 = compute_temporal_flickering_scores(video_list)
        all_results["D3_temporal_flickering"] = d3
        _save_metric(d3, output_dir, "D3_temporal_flickering.json")

    if "D2" in all_metrics:
        logger.info("Loading AMT model...")
        amt_model = _load_amt(device)
        logger.info("Computing D2: motion_smoothness...")
        d2 = compute_motion_smoothness_scores(amt_model, video_list)
        all_results["D2_motion_smoothness"] = d2
        _save_metric(d2, output_dir, "D2_motion_smoothness.json")
        del amt_model
        torch.cuda.empty_cache()

    if "D4" in all_metrics:
        logger.info("Loading DINO model...")
        dino_model = _load_dino(device)
        logger.info("Computing D4: subject_consistency...")
        d4 = compute_subject_consistency_scores(dino_model, video_list, device)
        all_results["D4_subject_consistency"] = d4
        _save_metric(d4, output_dir, "D4_subject_consistency.json")
        del dino_model
        torch.cuda.empty_cache()

    # Also load any previously saved individual metric files for full merge
    metric_files = {
        "D1_dynamic_degree": "D1_dynamic_degree.json",
        "D2_motion_smoothness": "D2_motion_smoothness.json",
        "D3_temporal_flickering": "D3_temporal_flickering.json",
        "D4_subject_consistency": "D4_subject_consistency.json",
        "D5_flow_stability": "D5_flow_stability.json",
    }
    for key, fname in metric_files.items():
        if key not in all_results:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    all_results[key] = json.load(f)
                logger.info("Loaded existing %s for merge", fname)

    # Merge all into a single per-video table
    merged = _merge_results(all_results, video_list)
    merged_path = os.path.join(output_dir, "tier1_results.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info("Tier 1 results saved to %s", merged_path)

    return all_results


def _save_metric(results: list[dict], output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s (%d videos)", filename, len(results))


def _merge_results(
    all_results: dict[str, list[dict[str, Any]]],
    video_list: list[str],
) -> list[dict[str, Any]]:
    """Merge per-metric results into a single per-video table."""
    # Index by video_path
    merged: dict[str, dict[str, Any]] = {v: {"video_path": v} for v in video_list}

    for metric_name, results in all_results.items():
        for r in results:
            vp = r["video_path"]
            if vp in merged:
                for k, v in r.items():
                    if k != "video_path":
                        merged[vp][k] = v

    return list(merged.values())
