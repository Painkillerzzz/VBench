"""Run final classification on mushroom videos.

Usage:
    # VLM-only classification (percentile-based):
    python -m mushroom_eval.run_classify \
        --vlm_results mushroom_eval_results/vlm_results.json \
        --percentile 20

    # VLM + Tier1 combined:
    python -m mushroom_eval.run_classify \
        --vlm_results mushroom_eval_results/vlm_results.json \
        --tier1_results mushroom_eval_results/tier1_results.json

    # Compare flash-lite vs pro-preview:
    python -m mushroom_eval.run_classify \
        --vlm_results mushroom_eval_results/vlm_results.json \
        --vlm_results_b mushroom_eval_results/vlm_results_flash_lite.json \
        --compare
"""

import argparse
import json
import logging
import os

from .fusion import (
    classify_combined,
    classify_vlm_only,
    compare_vlm_models,
    save_classification,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify mushroom videos")
    parser.add_argument("--vlm_results", required=True, help="VLM results JSON")
    parser.add_argument("--tier1_results", default=None, help="Tier 1 merged results JSON")
    parser.add_argument("--vlm_results_b", default=None, help="Second VLM results for comparison")
    parser.add_argument("--output_dir", default="mushroom_eval_results")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed VLM threshold")
    parser.add_argument("--percentile", type=float, default=20.0, help="Percentile threshold (default: 20)")
    parser.add_argument("--compare", action="store_true", help="Compare two VLM models")
    # Tier1 veto thresholds
    parser.add_argument("--veto_static", type=float, default=None,
                        help="Veto if static_ratio >= this value (e.g. 0.95)")
    parser.add_argument("--veto_consistency", type=float, default=None,
                        help="Veto if subject_consistency < this value (e.g. 0.90)")
    parser.add_argument("--veto_acceleration", type=float, default=None,
                        help="Veto if flow_acceleration > this value (e.g. 50)")
    args = parser.parse_args()

    # Load VLM results
    with open(args.vlm_results) as f:
        vlm_results = json.load(f)
    logger.info("Loaded %d VLM results from %s", len(vlm_results), args.vlm_results)

    # Compare mode
    if args.compare and args.vlm_results_b:
        with open(args.vlm_results_b) as f:
            vlm_results_b = json.load(f)
        label_a = os.path.basename(args.vlm_results).replace(".json", "")
        label_b = os.path.basename(args.vlm_results_b).replace(".json", "")
        comparison = compare_vlm_models(vlm_results, label_a, vlm_results_b, label_b)

        # Print
        print(f"\n{'='*60}")
        print(f"VLM Model Comparison: {label_a} vs {label_b}")
        print(f"{'='*60}")
        print(f"  Common videos:     {comparison['common_videos']}")
        print(f"  Score agreement:   {comparison['score_agreement_rate']:.1%}")
        print(f"  Score diff (mean): {comparison['score_diff_mean']:.3f}")
        print(f"  Score diff (max):  {comparison['score_diff_max']:.3f}")
        print(f"\n  Per-question agreement:")
        for q_id, rate in comparison["per_question_agreement"].items():
            print(f"    {q_id}: {rate:.1%}")
        print(f"{'='*60}\n")

        out_path = os.path.join(args.output_dir, "vlm_comparison.json")
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Comparison saved to %s", out_path)
        return

    # Classification
    tier1_results = None
    if args.tier1_results:
        with open(args.tier1_results) as f:
            tier1_results = json.load(f)
        logger.info("Loaded %d Tier 1 results", len(tier1_results))

    # Build veto rules
    veto_rules = {}
    if args.veto_static is not None:
        veto_rules["static_ratio"] = args.veto_static
    if args.veto_consistency is not None:
        veto_rules["subject_consistency_below"] = args.veto_consistency
    if args.veto_acceleration is not None:
        veto_rules["flow_acceleration_above"] = args.veto_acceleration

    if tier1_results:
        classified = classify_combined(
            vlm_results, tier1_results,
            vlm_threshold=args.threshold,
            vlm_percentile=args.percentile,
            tier1_veto_rules=veto_rules or None,
        )
    else:
        classified = classify_vlm_only(
            vlm_results,
            threshold=args.threshold,
            percentile=args.percentile,
        )

    save_classification(classified, args.output_dir)


if __name__ == "__main__":
    main()
