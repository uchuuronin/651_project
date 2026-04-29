# Utility recovery experiment.
# For each lambda, apply each theoretical threshold post-hoc and measure the actual loss incurred. 
# Compare against grid-search optimal and random.

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import LAMBDA_GRID, TAU_GRID
from src.grid import expected_loss
from src.thresholds import tau_U, tau_B, tau_CE

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_FORMULAS = {"tau_U": tau_U, "tau_B": tau_B, "tau_CE": tau_CE}
_LOSS_FNS = ["utility", "brier", "cross-entropy"]

def compute_recovery(confidences, correct, signal="token_prob_first"):
    """
    For each loss function and each lambda:
    - Compute oracle loss (grid search p*)
    - Compute loss at each theoretical threshold
    - Compute loss at random threshold (mean over TAU_GRID)
    - Report regret = loss(tau) - loss(p*)
    """
    results = {}

    for loss_fn in _LOSS_FNS:
        logger.info(f"\t{loss_fn}...")
        rows = []

        for lam in LAMBDA_GRID:
            # Oracle: grid search
            losses_grid = [expected_loss(confidences, correct, tau, lam, loss_fn) for tau in TAU_GRID]
            oracle_loss = float(np.min(losses_grid))
            p_star = float(TAU_GRID[int(np.argmin(losses_grid))])

            # Random baseline: average loss across all thresholds
            random_loss = float(np.mean(losses_grid))

            # Each theoretical threshold
            row = {"lambda": float(lam), "p_star": p_star, "loss_oracle": oracle_loss, "loss_random": random_loss}

            for name, fn in _FORMULAS.items():
                tau_val = float(fn(lam))
                loss_at_tau = expected_loss(confidences, correct, tau_val, lam, loss_fn)
                row[f"tau_{name}"] = tau_val
                row[f"loss_{name}"] = float(loss_at_tau)
                row[f"regret_{name}"] = float(loss_at_tau - oracle_loss)

            rows.append(row)

        results[loss_fn] = rows

    return results

def summarize(results):
    records = []
    for loss_fn in _LOSS_FNS:
        rows = results[loss_fn]
        regrets = {
            "tau_U": np.mean([r["regret_tau_U"] for r in rows]),
            "tau_B": np.mean([r["regret_tau_B"] for r in rows]),
            "tau_CE": np.mean([r["regret_tau_CE"] for r in rows]),
            "random": np.mean([r["loss_random"] - r["loss_oracle"] for r in rows]),
        }
        recovery = {
            f"recovery_{k}": 1 - v / (regrets["random"] + 1e-12)
            for k, v in regrets.items() if k != "random"
        }
        records.append({"loss": loss_fn, **{f"regret_{k}": v for k, v in regrets.items()}, **recovery})

    df = pd.DataFrame(records).set_index("loss")
    logger.info("\nMean Regret (lower preferred):")
    logger.info("\n" + df.filter(like="regret").to_string(float_format="%.4f"))

    logger.info("\nRecovery Rate (higher preferred):")
    recovery_df = df.filter(like="recovery").applymap(lambda x: f"{x:.1%}")
    logger.info("\n" + recovery_df.to_string())
    return df.to_dict(orient="index")

def main():
    parser = argparse.ArgumentParser(description="Utility recovery experiment.")
    parser.add_argument("--lim", type=int, default=1000)
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default="1.5b")
    parser.add_argument("--signal", type=str, default="token_prob_first", choices=["token_prob_first", "token_prob_mean"])
    args = parser.parse_args()

    inference_file = Path(f"results/inference_{args.dataset}_{args.lim}_{args.model}.csv")
    if not inference_file.exists():
        logger.error(f"Not found: {inference_file}")
        return 1

    signal_tag = f"_{args.signal}" if args.signal != "token_prob_first" else ""
    out_file = Path(f"results/recovery_{args.dataset}_{args.lim}_{args.model}{signal_tag}.json")

    df = pd.read_csv(inference_file).dropna(subset=[args.signal])
    confidences = df[args.signal].values
    correct = df["correct"].astype(float).values
    
    logger.info(f"Utility recovery: {args.model} on {args.dataset} (n={len(df)}, signal={args.signal})")
    results = compute_recovery(confidences, correct, signal=args.signal)
    summary = summarize(results)

    per_lambda = {
        loss_fn: [{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in row.items()} for row in rows]
        for loss_fn, rows in results.items()
    }

    output = {"summary": summary, "per_lambda": per_lambda}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
