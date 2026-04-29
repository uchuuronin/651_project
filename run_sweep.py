#!/usr/bin/env python3
# Runs abstention sweep + empirical grid search on inference CSV.
# Produces sweep_{dataset}_{lim}_{model}.json (theoretical threshold behavior) and mae_{dataset}_{lim}_{model}.json (p* vs theoretical MAE table).

import logging
import argparse
from pathlib import Path
import pandas as pd

from src.sweep import AbstractionSweep
from src.grid import run_grid_search, save_grid_results, print_mae_table

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_LIM = 50
DEFAULT_MODEL = "1.5b"
DEFAULT_SIGNAL = "token_prob_first"

def main():
    parser = argparse.ArgumentParser(description="Run abstention sweep + grid search on inference CSV.")
    parser.add_argument("--lim", type=int, default=DEFAULT_LIM,help=f"Number of examples (default: {DEFAULT_LIM})")
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model tag for output filename (default: {DEFAULT_MODEL}). "
                        f"Does not change which model llama-server loads, set that when starting llama-server.")
    parser.add_argument("--signal", type=str, default=DEFAULT_SIGNAL, choices=["token_prob_first", "token_prob_mean"], 
                        help=f"Confidence signal column to use (default: {DEFAULT_SIGNAL})")
    args = parser.parse_args()

    # Input CSV: results/inference_{dataset}_{lim}_{model}.csv
    inference_file = Path(f"results/inference_{args.dataset}_{args.lim}_{args.model}.csv")

    # Output filenames include signal tag when not default
    signal_tag = f"_{args.signal}" if args.signal != DEFAULT_SIGNAL else ""
    sweep_file = Path(f"results/sweep_{args.dataset}_{args.lim}_{args.model}{signal_tag}.json")
    mae_file = Path(f"results/mae_{args.dataset}_{args.lim}_{args.model}{signal_tag}.json")

    logger.info(f"SWEEP + GRID SEARCH: {args.dataset} | n={args.lim} | model={args.model} | signal={args.signal}")

    if not inference_file.exists():
        logger.error(f"Inference file not found: {inference_file}")
        logger.error(f"Generate it with: python run_inference.py --lim {args.lim} --dataset {args.dataset} --model {args.model}")
        return 1

    df = pd.read_csv(inference_file).dropna(subset=[args.signal])
    confidences = df[args.signal].values
    correct = df["correct"].astype(float).values

    logger.info(f"Loaded {len(df)} examples (signal={args.signal})")
    logger.info(f"\tConfidence mean={confidences.mean():.3f} range=[{confidences.min():.3f}, {confidences.max():.3f}]")
    logger.info(f"\tAccuracy: {correct.mean():.1%} ({int(correct.sum())}/{len(correct)})")

    # 1: theoretical threshold sweep (AbstractionSweep from src/sweep.py)
    logger.info("\n01: Theoretical threshold sweep...")
    sweep = AbstractionSweep(loss_functions=["utility", "brier", "cross-entropy"])
    sweep.evaluate(confidences, correct)
    sweep.save_results(str(sweep_file))
    logger.info(f"Saved: {sweep_file}")

    logger.info("\nIntegrated metrics:")
    for loss_fn, metrics in sweep.compute_integrated_metrics().items():
        logger.info(f"\t{loss_fn}:  "
                    f"mean_acc={metrics['mean_accuracy']:.3f},"
                    f"mean_cov={metrics['mean_coverage']:.3f}, "
                    f"max_f1={metrics['max_f1']:.3f}")

    logger.info("\nOptimal λ by accuracy:")
    for loss_fn, lam in sweep.get_optimal_lambda(metric="accuracy").items():
        logger.info(f"\t{loss_fn}: {lam:.4f}")

    # Part 2: grid search for p*(lambda) (run_grid_search from src/grid.py)
    logger.info("\n02: Grid search for p*(λ)...")
    mae_results = run_grid_search(confidences, correct)
    save_grid_results(mae_results, str(mae_file))
    logger.info(f"Saved: {mae_file}")

    print_mae_table(mae_results)

    logger.info("\nSWEEP COMPLETE")
    return 0

if __name__ == "__main__":
    exit(main())