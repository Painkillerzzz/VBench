"""Final classification pipeline: V2 direct score + D2/D3/D4 → SVM → good/bad.

Training mode:
    python -m mushroom_eval.classify_final train \
        --video_dir mushroom_data/videos \
        --badcase_list mushroom_eval_results/badcase_list.txt \
        --output_dir mushroom_eval_results/final

Inference mode (new videos):
    python -m mushroom_eval.classify_final infer \
        --video_dir /path/to/new/videos \
        --model_path mushroom_eval_results/final/model.pkl \
        --output_dir mushroom_eval_results/final
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VBENCH_PYTHON = "/home/xiangyuz22/miniconda3/envs/vbench/bin/python"

VLM_PROMPT = (
    "You are evaluating AI-generated videos of a mushroom character called 蘑菇TUTU.\n\n"
    "IMPORTANT: These are 5-second clips. Some videos intentionally show the character "
    "in calm/still poses (sitting, watching, resting) — this is NORMAL and NOT a defect. "
    "Only flag issues that represent actual generation failures.\n\n"
    "Check for these ACTUAL defects:\n"
    "1. MOTION SUBJECT ERROR: Character is completely frozen like a statue while ONLY background/camera moves\n"
    "2. STIFF MOTION: When the character DOES move, motion looks mechanical/robotic with no natural body deformation "
    "(a character being intentionally still or moving slowly is NOT stiff)\n"
    "3. APPEARANCE DRIFT: Character's color, texture, shape, or size visibly changes/morphs during video\n"
    "4. FRAME JUMPS: Sudden discontinuities where the character teleports or the scene abruptly changes\n"
    "5. VISUAL ARTIFACTS: Strange distorted objects, melting shapes, or unrecognizable blobs appear\n"
    "6. PHYSICS VIOLATIONS: Objects float impossibly, merge incorrectly, or defy gravity\n\n"
    "Score 1-10:\n"
    "- 1-3: Severe defects\n- 4-5: Noticeable defects\n- 6-7: Minor issues\n- 8-10: Good quality\n\n"
    "Remember: A character sitting quietly or making small movements is NOT a defect.\n\n"
    'Output ONLY JSON: {"score": N, "issues": ["brief description", ...]}'
)

FEATURE_NAMES = ["vlm_score", "D2_motion_smoothness", "D3_temporal_flickering", "D4_subject_consistency"]


# ── VLM scoring ──────────────────────────────────────────────────────────

async def _score_one(client, model, video_path, semaphore):
    async with semaphore:
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=Path(video_path).read_bytes(), mime_type="video/mp4"),
                    VLM_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=8192,
                    response_mime_type="application/json",
                ),
            )
            m = re.search(r'\{.*\}', resp.text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                return Path(video_path).name, obj.get("score"), obj.get("issues", [])
        except Exception as e:
            logger.warning("VLM error %s: %s", Path(video_path).name, str(e)[:100])
        return Path(video_path).name, None, []


def run_vlm_scoring(video_dir: str, videos: list[str], model: str = "gemini-3.1-pro-preview",
                     max_concurrent: int = 100) -> dict[str, dict]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"timeout": 120000})

    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        tasks = [_score_one(client, model, os.path.join(video_dir, v), sem) for v in videos]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    return {name: {"score": score, "issues": issues} for name, score, issues in results}


# ── Tier1 (D2, D3, D4) ──────────────────────────────────────────────────

def run_tier1(video_dir: str, videos: list[str], output_dir: str) -> dict[str, dict]:
    """Run D2, D3, D4 via VBench env subprocess or direct import."""
    import subprocess
    import glob

    # Write video list to temp file
    list_path = os.path.join(output_dir, "_video_list.json")
    full_paths = [os.path.join(video_dir, v) for v in videos]
    with open(list_path, "w") as f:
        json.dump(full_paths, f)

    # Try direct import (if running in vbench env)
    try:
        from mushroom_eval.tier1_metrics import run_tier1 as _run_tier1
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _run_tier1(full_paths, device=device, output_dir=output_dir, metrics=["D2", "D3", "D4"])
    except ImportError:
        # Fallback: run via vbench python
        logger.info("Running Tier1 via %s", VBENCH_PYTHON)
        cmd = [VBENCH_PYTHON, "-m", "mushroom_eval.run_tier1",
               "--video_dir", video_dir, "--metrics", "D2", "D3", "D4",
               "--output_dir", output_dir]
        subprocess.run(cmd, check=True)

    # Load results
    result = {}
    for metric_file, keys in [
        ("D2_motion_smoothness.json", ["motion_smoothness"]),
        ("D3_temporal_flickering.json", ["temporal_flickering"]),
        ("D4_subject_consistency.json", ["subject_consistency"]),
    ]:
        fpath = os.path.join(output_dir, metric_file)
        if os.path.exists(fpath):
            with open(fpath) as f:
                for r in json.load(f):
                    name = Path(r["video_path"]).name
                    if name not in result:
                        result[name] = {}
                    for k in keys:
                        result[name][k] = r.get(k) or r.get(f"D{metric_file[1]}_{k}")
    return result


# ── Feature extraction ───────────────────────────────────────────────────

def extract_features(vlm_scores: dict, tier1_scores: dict, videos: list[str]) -> tuple[np.ndarray, list[str]]:
    X, valid_videos = [], []
    for v in videos:
        vlm = vlm_scores.get(v, {}).get("score")
        t1 = tier1_scores.get(v, {})
        d2 = t1.get("motion_smoothness")
        d3 = t1.get("temporal_flickering")
        d4 = t1.get("subject_consistency")
        if vlm is not None and d2 is not None and d3 is not None and d4 is not None:
            X.append([float(vlm), float(d2), float(d3), float(d4)])
            valid_videos.append(v)
    return np.array(X), valid_videos


# ── Train ────────────────────────────────────────────────────────────────

def cmd_train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    videos = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    logger.info("Videos: %d", len(videos))

    # Load or compute VLM scores
    vlm_cache = os.path.join(args.output_dir, "vlm_scores.json")
    if os.path.exists(vlm_cache):
        with open(vlm_cache) as f:
            vlm_scores = json.load(f)
        logger.info("Loaded VLM scores from cache (%d)", len(vlm_scores))
    else:
        logger.info("Running VLM scoring...")
        vlm_scores = run_vlm_scoring(args.video_dir, videos, max_concurrent=args.max_concurrent)
        with open(vlm_cache, "w") as f:
            json.dump(vlm_scores, f, ensure_ascii=False, indent=2)

    # Load or compute Tier1
    tier1_cache = os.path.join(args.output_dir, "tier1_scores.json")
    if os.path.exists(tier1_cache):
        with open(tier1_cache) as f:
            tier1_scores = json.load(f)
        logger.info("Loaded Tier1 scores from cache (%d)", len(tier1_scores))
    else:
        logger.info("Running Tier1 (D2, D3, D4)...")
        tier1_scores = run_tier1(args.video_dir, videos, args.output_dir)
        with open(tier1_cache, "w") as f:
            json.dump(tier1_scores, f, ensure_ascii=False, indent=2)

    # Labels
    all_bad = set()
    with open(args.badcase_list) as f:
        for line in f:
            line = line.strip()
            if line.endswith(".mp4"):
                all_bad.add(line.split("/")[-1])

    # Extract features
    X, valid_videos = extract_features(vlm_scores, tier1_scores, videos)
    y = np.array([1 if v in all_bad else 0 for v in valid_videos])
    logger.info("Features: %d videos, %d dims, bad=%d", len(X), X.shape[1], y.sum())

    # Train/test split
    X_tr, X_te, y_tr, y_te, v_tr, v_te = train_test_split(
        X, y, valid_videos, test_size=0.3, random_state=42, stratify=y)

    # Train with best params (C=0.1, bw=2.0 from tuning)
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=0.1, class_weight={0: 1, 1: 2.0}, probability=True))])
    model.fit(X_tr, y_tr)

    # Evaluate
    y_pred = model.predict(X_te)
    print(f"\n{'='*50}")
    print(f"  Test set: {len(X_te)} videos")
    print(f"  F1={f1_score(y_te,y_pred):.3f}  P={precision_score(y_te,y_pred):.3f}  R={recall_score(y_te,y_pred):.3f}")
    print(f"{'='*50}")
    print(classification_report(y_te, y_pred, target_names=["good", "bad"]))

    # Retrain on ALL data for deployment
    model_full = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=0.1, class_weight={0: 1, 1: 2.0}, probability=True))])
    model_full.fit(X, y)

    # Save model
    model_path = os.path.join(args.output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_full, f)
    logger.info("Model saved to %s", model_path)

    # Save classification for all videos
    _save_results(model_full, X, valid_videos, vlm_scores, tier1_scores, args.output_dir)


# ── Infer ────────────────────────────────────────────────────────────────

def cmd_infer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    videos = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    logger.info("Videos: %d", len(videos))

    # Load model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", args.model_path)

    # VLM scoring
    logger.info("Running VLM scoring...")
    vlm_scores = run_vlm_scoring(args.video_dir, videos, max_concurrent=args.max_concurrent)
    with open(os.path.join(args.output_dir, "vlm_scores.json"), "w") as f:
        json.dump(vlm_scores, f, ensure_ascii=False, indent=2)

    # Tier1
    logger.info("Running Tier1...")
    tier1_scores = run_tier1(args.video_dir, videos, args.output_dir)
    with open(os.path.join(args.output_dir, "tier1_scores.json"), "w") as f:
        json.dump(tier1_scores, f, ensure_ascii=False, indent=2)

    # Extract and predict
    X, valid_videos = extract_features(vlm_scores, tier1_scores, videos)
    _save_results(model, X, valid_videos, vlm_scores, tier1_scores, args.output_dir)


# ── Save results ─────────────────────────────────────────────────────────

def _save_results(model, X, valid_videos, vlm_scores, tier1_scores, output_dir):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X))

    rows = []
    for i, v in enumerate(valid_videos):
        vlm = vlm_scores.get(v, {})
        t1 = tier1_scores.get(v, {})
        rows.append({
            "video": v,
            "label": "bad" if y_pred[i] == 1 else "good",
            "bad_probability": round(float(y_prob[i]), 3),
            "vlm_score": vlm.get("score"),
            "vlm_issues": "; ".join(vlm.get("issues", [])),
            "D2_motion_smoothness": t1.get("motion_smoothness"),
            "D3_temporal_flickering": t1.get("temporal_flickering"),
            "D4_subject_consistency": t1.get("subject_consistency"),
        })

    rows.sort(key=lambda r: int(r["video"].replace(".mp4", "")) if r["video"].replace(".mp4", "").isdigit() else 0)

    # JSON
    json_path = os.path.join(output_dir, "classification_final.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = os.path.join(output_dir, "classification_final.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Lists
    bad_list = [r["video"] for r in rows if r["label"] == "bad"]
    good_list = [r["video"] for r in rows if r["label"] == "good"]
    with open(os.path.join(output_dir, "bad_videos.txt"), "w") as f:
        f.write("\n".join(bad_list) + "\n")
    with open(os.path.join(output_dir, "good_videos.txt"), "w") as f:
        f.write("\n".join(good_list) + "\n")

    n = len(rows)
    n_bad = len(bad_list)
    print(f"\n{'='*50}")
    print(f"  Total: {n}  Good: {n-n_bad} ({100*(n-n_bad)/n:.1f}%)  Bad: {n_bad} ({100*n_bad/n:.1f}%)")
    print(f"  Saved: {json_path}")
    print(f"         {csv_path}")
    print(f"{'='*50}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--video_dir", required=True)
    p_train.add_argument("--badcase_list", required=True)
    p_train.add_argument("--output_dir", default="mushroom_eval_results/final")
    p_train.add_argument("--max_concurrent", type=int, default=100)

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--video_dir", required=True)
    p_infer.add_argument("--model_path", required=True)
    p_infer.add_argument("--output_dir", required=True)
    p_infer.add_argument("--max_concurrent", type=int, default=100)

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "infer":
        cmd_infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
