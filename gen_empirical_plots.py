#!/usr/bin/env python3
# Generate empirical vs theoretical threshold comparison plots.
# Loads sweep JSON files produced by sweep.py.
import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.config import LOSS_FUNCTIONS
from src.sweep import AbstractionSweep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

def load_sweep(filepath: str) -> AbstractionSweep:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}. Generate it with: python sweep.py (check --lim, --dataset, --model)")
    return AbstractionSweep.load_results(str(path))


def plot_empirical_vs_theoretical(sweep: AbstractionSweep, label: str, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Empirical vs Theoretical Abstention Thresholds\n({label})", fontsize=14, fontweight="bold")

    colors = {"utility": "#1f77b4", "brier": "#ff7f0e", "cross-entropy": "#2ca02c"}
    for idx, loss_fn in enumerate(["utility", "brier", "cross-entropy"]):
        ax = axes[idx]
        results = sweep.results[loss_fn]
        lambdas = results["lambda"]
        accuracy = results["accuracy"]
        coverage = results["coverage"]

        ax.plot(lambdas, accuracy, "o-", label="Empirical Accuracy",color=colors[loss_fn], linewidth=2, markersize=4)

        ax_twin = ax.twinx()
        ax_twin.plot(lambdas, coverage, "s--", label="Coverage", color=colors[loss_fn], alpha=0.5, linewidth=2, markersize=3)

        threshold_fn = LOSS_FUNCTIONS[loss_fn]
        tau_vals = np.array([threshold_fn(lam) for lam in lambdas])
        tau_norm = (tau_vals - tau_vals.min()) / (tau_vals.max() - tau_vals.min() + 1e-9)
        ax_twin.plot(lambdas, tau_norm, "^--", label="Theoretical τ(λ) (normalized)",
                     color="red", alpha=0.6, linewidth=1.5, markersize=4)

        max_idx = np.argmax(accuracy)
        ax.plot(lambdas[max_idx], accuracy[max_idx], "r*", markersize=20, label=f"Optimal (λ={lambdas[max_idx]:.3f})")

        ax.set_xlabel("Cost Weight λ", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11, color=colors[loss_fn])
        ax_twin.set_ylabel("Coverage / Threshold", fontsize=11)
        ax.set_title(loss_fn.replace("-", " ").title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, 0.26)
        ax.set_ylim(-0.05, 1.05)
        ax_twin.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

    plt.tight_layout()
    out = output_dir / "empirical_vs_theoretical.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close()


def plot_pilot_vs_full(pilot_sweep: AbstractionSweep, full_sweep: AbstractionSweep, pilot_label: str, full_label: str, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Scaling: {pilot_label} vs {full_label}", fontsize=14, fontweight="bold")

    colors = {"utility": "#1f77b4", "brier": "#ff7f0e", "cross-entropy": "#2ca02c"}
    for idx, loss_fn in enumerate(["utility", "brier", "cross-entropy"]):
        ax = axes[idx]
        pr = pilot_sweep.results[loss_fn]
        fr = full_sweep.results[loss_fn]
        lambdas = pr["lambda"]

        ax.plot(lambdas, pr["accuracy"], "o-", label=pilot_label, linewidth=2, color=colors[loss_fn], alpha=0.6)
        ax.plot(lambdas, fr["accuracy"], "s-", label=full_label, linewidth=2, color=colors[loss_fn])

        ax_twin = ax.twinx()
        ax_twin.plot(lambdas, fr["coverage"], "--", label="Coverage (full)", color=colors[loss_fn], alpha=0.3, linewidth=1.5)

        ax.set_xlabel("Cost Weight λ", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax_twin.set_ylabel("Coverage", fontsize=11)
        ax.set_title(loss_fn.replace("-", " ").title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

    plt.tight_layout()
    out = output_dir/"pilot_vs_full_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close()


def generate_summary(pilot_sweep: AbstractionSweep, full_sweep: AbstractionSweep, output_file: Path):
    report = {"loss_functions": ["utility", "brier", "cross-entropy"], "comparison": {}}

    for loss_fn in report["loss_functions"]:
        pr = pilot_sweep.results[loss_fn]
        fr = full_sweep.results[loss_fn]
        pi = int(np.argmax(pr["accuracy"]))
        fi = int(np.argmax(fr["accuracy"]))

        report["comparison"][loss_fn] = {
            "pilot": {
                "max_accuracy": float(pr["accuracy"][pi]),
                "optimal_lambda": float(pr["lambda"][pi]),
                "coverage_at_optimal": float(pr["coverage"][pi]),
            },
            "full": {
                "max_accuracy": float(fr["accuracy"][fi]),
                "optimal_lambda": float(fr["lambda"][fi]),
                "coverage_at_optimal": float(fr["coverage"][fi]),
            },
        }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-lim", type=int, default=50)
    parser.add_argument("--full-lim", type=int, default=1000)
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--model", type=str, default="1.5b")
    args = parser.parse_args()

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    pilot_file = f"results/sweep_{args.dataset}_{args.pilot_lim}_{args.model}.json"
    full_file = f"results/sweep_{args.dataset}_{args.full_lim}_{args.model}.json"
    summary_file = output_dir / "empirical_comparison_summary.json"

    logger.info("EMPIRICAL VS THEORETICAL COMPARISON")
    try:
        pilot_sweep = load_sweep(pilot_file)
        full_sweep = load_sweep(full_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    pilot_label = f"Pilot (n={args.pilot_lim})"
    full_label = f"Full (n={args.full_lim})"

    plot_empirical_vs_theoretical(
        full_sweep,
        f"{args.model.upper()}, {args.dataset}, n={args.full_lim}",
        output_dir
    )
    plot_pilot_vs_full(pilot_sweep, full_sweep, pilot_label, full_label, output_dir)
    report = generate_summary(pilot_sweep, full_sweep, summary_file)

    logger.info("\nSummary:")
    for loss_fn, c in report["comparison"].items():
        logger.info(f"{loss_fn}:")
        logger.info(f"\tPilot: {c['pilot']['max_accuracy']:.1%} @ λ={c['pilot']['optimal_lambda']:.3f}")
        logger.info(f"\tFull: {c['full']['max_accuracy']:.1%} @ λ={c['full']['optimal_lambda']:.3f}")

    return 0


if __name__ == "__main__":
    exit(main())