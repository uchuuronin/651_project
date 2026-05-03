#!/usr/bin/env python3
# Temperature scaling analysis for calibration sensitivity.

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.config import RESULTS_DIR
from src.grid import run_grid_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SIGNAL = "token_prob_first"
EPS = 1e-9


def temperature_scale(confidences: np.ndarray, temperature: float) -> np.ndarray:
    conf = np.clip(confidences, EPS, 1.0 - EPS)
    logits = np.log(conf / (1.0 - conf))
    scaled = expit(logits / temperature)
    return scaled


def negative_log_likelihood(log_T: np.ndarray, confidences: np.ndarray, correct: np.ndarray) -> float:
    temperature = float(np.exp(log_T[0]))
    scaled = temperature_scale(confidences, temperature)
    nll = -np.mean(correct * np.log(scaled + EPS) + (1.0 - correct) * np.log(1.0 - scaled + EPS))
    return float(nll)


def fit_temperature(confidences: np.ndarray, correct: np.ndarray, init_temperature: float = 1.0) -> tuple[float, float]:
    log_init = np.log(max(init_temperature, EPS))
    result = minimize(
        fun=negative_log_likelihood,
        x0=np.array([log_init], dtype=float),
        args=(confidences, correct),
        bounds=[(-10.0, 10.0)],
        method="L-BFGS-B",
    )

    if not result.success:
        logger.warning("Temperature scaling optimization did not converge, using T=1.0.")
        return 1.0, float(negative_log_likelihood(np.array([log_init]), confidences, correct))

    temperature = float(np.exp(result.x[0]))
    nll = float(result.fun)
    return temperature, nll


def main():
    parser = argparse.ArgumentParser(description="Temperature scaling for confidence calibration and threshold performance.")
    parser.add_argument("--lim", type=int, default=1000)
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default="1.5b")
    parser.add_argument("--signal", type=str, default=DEFAULT_SIGNAL,
                        choices=["token_prob_first", "token_prob_mean"])
    args = parser.parse_args()

    inference_file = Path(f"results/inference_{args.dataset}_{args.lim}_{args.model}.csv")
    if not inference_file.exists():
        logger.error(f"Inference file not found: {inference_file}")
        return 1

    df = pd.read_csv(inference_file).dropna(subset=[args.signal])
    confidences = df[args.signal].values
    correct = df["correct"].astype(float).values

    if len(confidences) == 0:
        logger.error("No valid examples found after dropping NaN confidences.")
        return 1

    logger.info(f"Temperature scaling: {args.model} on {args.dataset} (n={len(confidences)}, signal={args.signal})")

    temperature, nll = fit_temperature(confidences, correct)
    logger.info(f"Fitted temperature T={temperature:.4f} (NLL={nll:.6f})")

    scaled_confidences = temperature_scale(confidences, temperature)

    logger.info("Running grid search on original confidences...")
    grid_before = run_grid_search(confidences, correct)
    logger.info("Running grid search on temperature-scaled confidences...")
    grid_after = run_grid_search(scaled_confidences, correct)

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "lim": args.lim,
        "signal": args.signal,
        "temperature": temperature,
        "nll": nll,
        "before": {
            "mean_mae_tau_U": grid_before["utility"]["mean_mae_tau_U"],
            "mean_mae_tau_B": grid_before["brier"]["mean_mae_tau_B"],
            "mean_mae_tau_CE": grid_before["cross-entropy"]["mean_mae_tau_CE"],
        },
        "after": {
            "mean_mae_tau_U": grid_after["utility"]["mean_mae_tau_U"],
            "mean_mae_tau_B": grid_after["brier"]["mean_mae_tau_B"],
            "mean_mae_tau_CE": grid_after["cross-entropy"]["mean_mae_tau_CE"],
        },
    }

    logger.info("Mean MAE before scaling:")
    logger.info(f"\tutility: {summary['before']['mean_mae_tau_U']:.4f}")
    logger.info(f"\tbrier:   {summary['before']['mean_mae_tau_B']:.4f}")
    logger.info(f"\tce:      {summary['before']['mean_mae_tau_CE']:.4f}")
    logger.info("Mean MAE after scaling:")
    logger.info(f"\tutility: {summary['after']['mean_mae_tau_U']:.4f}")
    logger.info(f"\tbrier:   {summary['after']['mean_mae_tau_B']:.4f}")
    logger.info(f"\tce:      {summary['after']['mean_mae_tau_CE']:.4f}")

    out_file = RESULTS_DIR / f"temp_scaling_{args.dataset}_{args.lim}_{args.model}_{args.signal}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved temperature scaling results to {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
