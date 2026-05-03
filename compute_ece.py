#!/usr/bin/env python3
# Compute ECE (Expected Calibration Error) for inference CSV results.

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SIGNALS = ["token_prob_first", "token_prob_mean"]


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    if len(confidences) == 0:
        raise ValueError("No confidence values provided for ECE computation.")

    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    valid = ~np.isnan(conf)
    conf = conf[valid]
    correct = correct[valid]

    if len(conf) == 0:
        raise ValueError("All confidence values are NaN.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    details = []

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_conf = float(conf[mask].mean())
        mean_acc = float(correct[mask].mean())
        gap = abs(mean_conf - mean_acc)
        weight = count / n
        ece += weight * gap
        details.append({
            "bin": b,
            "count": count,
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "gap": gap,
            "weight": weight,
        })

    return float(ece), details


def main():
    parser = argparse.ArgumentParser(description="Compute Expected Calibration Error (ECE) for inference CSV files.")
    parser.add_argument("--lim", type=int, default=1000)
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default="1.5b")
    parser.add_argument("--signal", type=str, default="all",
                        choices=["all", "token_prob_first", "token_prob_mean"])
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of equal-width bins for ECE computation.")
    args = parser.parse_args()

    inference_file = Path(f"results/inference_{args.dataset}_{args.lim}_{args.model}.csv")
    if not inference_file.exists():
        logger.error(f"Inference file not found: {inference_file}")
        return 1

    df = pd.read_csv(inference_file)
    signals = DEFAULT_SIGNALS if args.signal == "all" else [args.signal]

    summary = {"dataset": args.dataset, "model": args.model, "lim": args.lim, "n_bins": args.n_bins, "signals": {}}

    for signal in signals:
        if signal not in df.columns:
            logger.warning(f"Signal column missing from CSV: {signal}")
            continue

        df_clean = df[[signal, "correct"]].dropna(subset=[signal])
        confidences = df_clean[signal].values
        correct = df_clean["correct"].astype(float).values

        if len(confidences) == 0:
            logger.warning(f"No valid examples for signal {signal}.")
            continue

        ece, details = compute_ece(confidences, correct, n_bins=args.n_bins)
        logger.info(f"ECE {signal}: {ece:.4f} over {len(confidences)} examples")

        summary["signals"][signal] = {
            "ece": ece,
            "num_examples": int(len(confidences)),
            "bin_details": details,
        }

    out_file = RESULTS_DIR / f"ece_{args.dataset}_{args.lim}_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved ECE results to {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
