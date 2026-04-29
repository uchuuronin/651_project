# Bootstrap 95% confidence intervals for MAE estimates.
# Resamples (confidence, correct) pairs with replacement, re-runs grid search each iteration, produces CI on mean MAE for each (loss_fn, formula) pair.

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

_THEORETICAL = {"utility": tau_U, "brier": tau_B, "cross-entropy": tau_CE}
_FORMULA_NAMES = ["tau_U", "tau_B", "tau_CE"]
_FORMULA_FNS = [tau_U, tau_B, tau_CE]
N_BOOT = 1000
SEED = 42


def grid_search_mae_single_loss(confidences, correct, loss_fn):
    # Run grid search for one loss function, return mean MAE vs each formula.
    mae_per_formula = {name: [] for name in _FORMULA_NAMES}

    for lam in LAMBDA_GRID:
        losses = [expected_loss(confidences, correct, tau, lam, loss_fn) for tau in TAU_GRID]
        p_star = float(TAU_GRID[int(np.argmin(losses))])

        for name, fn in zip(_FORMULA_NAMES, _FORMULA_FNS):
            mae_per_formula[name].append(abs(p_star - float(fn(lam))))

    return {name: float(np.mean(vals)) for name, vals in mae_per_formula.items()}


def run_bootstrap(confidences, correct, n_boot=N_BOOT, seed=SEED):
    # Bootstrap resample and re-run grid search each iteration.
    rng = np.random.default_rng(seed)
    n = len(confidences)

    #results[loss_fn][formula_name] = list of n_boot MAE values
    results = {}
    for loss_fn in _THEORETICAL:
        results[loss_fn] = {name: [] for name in _FORMULA_NAMES}

    for b in range(n_boot):
        if (b + 1) % 100 == 0:
            logger.info(f"  Bootstrap iteration {b+1}/{n_boot}")

        idx = rng.integers(0, n, size=n)
        c_boot = confidences[idx]
        y_boot = correct[idx]

        for loss_fn in _THEORETICAL:
            mae_dict = grid_search_mae_single_loss(c_boot, y_boot, loss_fn)
            for name in _FORMULA_NAMES:
                results[loss_fn][name].append(mae_dict[name])

    return results


def compute_cis(boot_results, alpha=0.05):
    # Compute point estimate + percentile CI from bootstrap samples.
    summary = {}
    for loss_fn, formula_dict in boot_results.items():
        summary[loss_fn] = {}
        for name, samples in formula_dict.items():
            arr = np.array(samples)
            summary[loss_fn][name] = {
                "mean": float(np.mean(arr)),
                "ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
                "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
                "std": float(np.std(arr)),
            }
    return summary

def print_ci_table(summary):
    records = []
    for loss_fn in ["utility", "brier", "cross-entropy"]:
        row = {"loss": loss_fn}
        for name in _FORMULA_NAMES:
            s = summary[loss_fn][name]
            row[name] = f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
        records.append(row)

    df = pd.DataFrame(records).set_index("loss")
    logger.info("\nMAE with 95% Bootstrap CIs:")
    logger.info("\n" + df.to_string())
    
def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for MAE estimates.")
    parser.add_argument("--lim", type=int, default=1000)
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default="1.5b")
    parser.add_argument("--signal", type=str, default="token_prob_first", choices=["token_prob_first", "token_prob_mean"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    inference_file = Path(f"results/inference_{args.dataset}_{args.lim}_{args.model}.csv")
    if not inference_file.exists():
        logger.error(f"Not found: {inference_file}")
        return 1

    signal_tag = f"_{args.signal}" if args.signal != "token_prob_first" else ""
    out_file = Path(f"results/bootstrap_{args.dataset}_{args.lim}_{args.model}{signal_tag}.json")

    df = pd.read_csv(inference_file).dropna(subset=[args.signal])
    confidences = df[args.signal].values
    correct = df["correct"].astype(float).values

    logger.info(f"Bootstrap MAE: {args.model} on {args.dataset} (n={len(df)}, signal={args.signal}, B={args.n_boot})")
    boot_results = run_bootstrap(confidences, correct, n_boot=args.n_boot, seed=args.seed)
    summary = compute_cis(boot_results)
    print_ci_table(summary)

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {out_file}")

    return 0

if __name__ == "__main__":
    exit(main())