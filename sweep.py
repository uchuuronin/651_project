#!/usr/bin/env python3
# Run abstention sweep on inference CSV.
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.sweep import AbstractionSweep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_LIM = 50
DEFAULT_MODEL = "1.5b"


def main():
    parser = argparse.ArgumentParser(description="Run abstention sweep on inference CSV.")
    parser.add_argument("--lim", type=int, default=DEFAULT_LIM,help=f"Number of examples (default: {DEFAULT_LIM})")
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model tag for output filename (default: {DEFAULT_MODEL}). "
                        f"Does not change which model llama-server loads, set that when starting llama-server.")
    args = parser.parse_args()

    # Input CSV: results/inference_{dataset}_{lim}_{model}.csv
    inference_file = Path(f"results/inference_{args.dataset}_{args.lim}_{args.model}.csv")
    # Output JSON: results/sweep_{dataset}_{lim}_{model}.json
    output_file = Path(f"results/sweep_{args.dataset}_{args.lim}_{args.model}.json")

    logger.info(f"SWEEP: {args.dataset} | n={args.lim} | model={args.model}")

    if not inference_file.exists():
        logger.error(f"Inference file not found: {inference_file}")
        logger.error(f"Generate it with: python run_inference.py --lim {args.lim} --dataset {args.dataset} --model {args.model}")
        return 1

    logger.info(f"Loading: {inference_file}")
    df = pd.read_csv(inference_file).dropna(subset=["token_prob_first"])
    confidences = df["token_prob_first"].values
    correct = df["correct"].astype(float).values

    logger.info(f"Loaded {len(df)} examples")
    logger.info(f"\tConfidence  mean={confidences.mean():.3f}  "
                f"range=[{confidences.min():.3f}, {confidences.max():.3f}]")
    logger.info(f"\tAccuracy: {correct.mean():.1%} ({int(correct.sum())}/{len(correct)})")

    sweep = AbstractionSweep(loss_functions=["utility", "brier", "cross-entropy"])
    logger.info("Running sweep (51 λ points × 3 loss functions)...")
    sweep.evaluate(confidences, correct)

    sweep.save_results(str(output_file))
    logger.info(f"Saved: {output_file}")

    logger.info("\nIntegrated metrics:")
    for loss_fn, metrics in sweep.compute_integrated_metrics().items():
        logger.info(f"\t{loss_fn}:  "
                    f"mean_acc={metrics['mean_accuracy']:.3f},"
                    f"mean_cov={metrics['mean_coverage']:.3f}, "
                    f"max_f1={metrics['max_f1']:.3f}")

    logger.info("\nOptimal λ by accuracy:")
    optimal = sweep.get_optimal_lambda(metric="accuracy")
    for loss_fn, lam in optimal.items():
        logger.info(f"\t{loss_fn}: {lam:.4f}")

    logger.info("SWEEP COMPLETE")
    return 0


if __name__ == "__main__":
    exit(main())