# Grid search for empirical optimal threshold p*(lambda) under each loss function.
# We sweep tau in [0,1] to minimize expected loss, finding p* empirically.

import json
import logging
import numpy as np

from src.config import LOSS_FUNCTIONS, LAMBDA_GRID, TAU_GRID
from src.thresholds import tau_U, tau_B, tau_CE
logger = logging.getLogger(__name__)

_THEORETICAL = {"utility": tau_U, "brier": tau_B, "cross-entropy": tau_CE}

def expected_loss(confidences: np.ndarray, correct: np.ndarray,
                  tau: float, lam: float, loss_fn: str) -> float:
    # Expected loss of applying threshold tau under a given loss function to find p*(lambda) via grid search.
    mask = confidences >= tau
    n = len(confidences)
    if mask.sum() == 0:
        return float(lam)

    c = confidences[mask]
    y = correct[mask].astype(float)
    n_abstain = n - mask.sum()

    if loss_fn == "utility":
        return float((-y.sum() + lam * (1 - y).sum() + lam * n_abstain) / n)
    elif loss_fn == "brier":
        return float(np.mean((c - y) ** 2) * mask.sum() / n + lam * n_abstain / n)
    elif loss_fn == "cross-entropy":
        c_clip = np.clip(c, 1e-9, 1 - 1e-9)
        ce = -np.mean(y * np.log(c_clip) + (1 - y) * np.log(1 - c_clip))
        return float(ce * mask.sum() / n + lam * n_abstain / n)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def run_grid_search(confidences: np.ndarray, correct: np.ndarray) -> dict:
    # For each loss function and lambda, find p*(lambda) by grid search over TAU_GRID.
    # Computes MAE between p* and each theoretical formula (tau_U, tau_B, tau_CE).
    results = {}

    for loss_fn in _THEORETICAL:
        logger.info(f"\tGrid search: {loss_fn}...")
        p_stars, t_U, t_B, t_CE, mae_U, mae_B, mae_CE = [], [], [], [], [], [], []

        for lam in LAMBDA_GRID:
            losses = [expected_loss(confidences, correct, tau, lam, loss_fn) for tau in TAU_GRID]
            p_star = float(TAU_GRID[int(np.argmin(losses))])
            u, b, ce = float(tau_U(lam)), float(tau_B(lam)), float(tau_CE(lam))
            p_stars.append(p_star)
            t_U.append(u); t_B.append(b); t_CE.append(ce)
            mae_U.append(abs(p_star - u))
            mae_B.append(abs(p_star - b))
            mae_CE.append(abs(p_star - ce))

        results[loss_fn] = {
            "lambda": LAMBDA_GRID.tolist(),
            "p_star": p_stars,
            "tau_U": t_U, "tau_B": t_B, "tau_CE": t_CE,
            "mae_tau_U": mae_U, "mae_tau_B": mae_B, "mae_tau_CE": mae_CE,
            "mean_mae_tau_U": float(np.mean(mae_U)),
            "mean_mae_tau_B": float(np.mean(mae_B)),
            "mean_mae_tau_CE": float(np.mean(mae_CE)),
        }
    return results

def save_grid_results(results: dict, filepath: str) -> None:
    # Serialize grid search results to JSON.
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved grid search results to {filepath}")

def load_grid_results(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)

def print_mae_table(mae_results: dict) -> None:
    logger.info("MAE: theoretical formula vs empirical p*")
    logger.info(f"\n\t{'Loss Function':<20} {'MAE(τ_U)':>10} {'MAE(τ_B)':>10} {'MAE(τ_CE)':>10}  Best")
    for loss_fn, res in mae_results.items():
        u = res["mean_mae_tau_U"]
        b = res["mean_mae_tau_B"]
        ce = res["mean_mae_tau_CE"]
        scores = {"τ_U": u, "τ_B": b, "τ_CE": ce}
        best = min(scores, key=scores.get)
        logger.info(f"\t{loss_fn:<20} {u:>10.4f} {b:>10.4f} {ce:>10.4f}  {best}")