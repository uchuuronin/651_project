#!/usr/bin/env python3
"""
Generate empirical vs theoretical threshold comparison plots.
Task 10: Create final empirical evaluation visualizations.

Compares the empirically observed optimal thresholds from 500-example sweep
against the theoretical predictions for each loss function.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import LOSS_FUNCTIONS
from src.sweep import AbstractionSweep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_both_sweeps(pilot_file: str, full_file: str):
    """
    Load both pilot and full sweep results for comparison.
    
    Args:
        pilot_file: Path to pilot sweep results
        full_file: Path to full sweep results
        
    Returns:
        Tuple of (pilot_sweep, full_sweep) AbstractionSweep instances
    """
    logger.info("Loading sweep results for comparison...")
    
    pilot_path = Path(pilot_file)
    full_path = Path(full_file)
    
    if not pilot_path.exists():
        raise FileNotFoundError(f"Pilot sweep not found: {pilot_path}")
    if not full_path.exists():
        raise FileNotFoundError(f"Full sweep not found: {full_path}")
    
    pilot_sweep = AbstractionSweep.load_results(str(pilot_path))
    full_sweep = AbstractionSweep.load_results(str(full_path))
    
    logger.info(f"Loaded pilot (50 ex) and full (500 ex) sweep results")
    return pilot_sweep, full_sweep


def plot_empirical_vs_theoretical(sweep: AbstractionSweep, output_dir: str = "results"):
    """
    Plot empirical tradeoff curves with theoretical thresholds overlaid.
    
    Args:
        sweep: AbstractionSweep instance with full results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating empirical vs theoretical comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Empirical Verification of Optimal Abstention Thresholds\n(500-Example Full Dataset)",
        fontsize=14,
        fontweight="bold"
    )
    
    colors = {"utility": "#1f77b4", "brier": "#ff7f0e", "cross-entropy": "#2ca02c"}
    
    for idx, loss_fn in enumerate(["utility", "brier", "cross-entropy"]):
        ax = axes[idx]
        results = sweep.results[loss_fn]
        
        lambdas = results["lambda"]
        accuracy = results["accuracy"]
        coverage = results["coverage"]
        
        # Empirical curve
        ax.plot(lambdas, accuracy, "o-", label="Empirical Accuracy", 
                color=colors[loss_fn], linewidth=2, markersize=4)
        
        # Also show coverage
        ax_twin = ax.twinx()
        ax_twin.plot(lambdas, coverage, "s--", label="Coverage", 
                    color=colors[loss_fn], alpha=0.5, linewidth=2, markersize=3)
        
        # Theoretical threshold function
        threshold_fn = LOSS_FUNCTIONS[loss_fn]
        theoretical_thresholds = np.array([threshold_fn(lam) for lam in lambdas])
        
        # Normalize thresholds for visualization (map to same scale as coverage)
        norm_thresholds = (theoretical_thresholds - theoretical_thresholds.min()) / \
                         (theoretical_thresholds.max() - theoretical_thresholds.min())
        
        ax_twin.plot(lambdas, norm_thresholds, "^--", label="Theoretical τ(λ) (normalized)",
                    color="red", alpha=0.6, linewidth=1.5, markersize=4)
        
        # Mark optimal point
        max_acc_idx = np.argmax(accuracy)
        ax.plot(lambdas[max_acc_idx], accuracy[max_acc_idx], "r*", 
               markersize=25, label=f"Optimal (λ={lambdas[max_acc_idx]:.3f})")
        
        ax.set_xlabel("Cost Weight λ", fontsize=11)
        ax.set_ylabel("Empirical Accuracy", fontsize=11, color=colors[loss_fn])
        ax_twin.set_ylabel("Coverage / Threshold", fontsize=11)
        ax.set_title(f"{loss_fn.replace('-', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, 0.26)
        ax.set_ylim(-0.05, 1.05)
        ax_twin.set_ylim(-0.05, 1.05)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    
    plt.tight_layout()
    filepath = output_dir / "empirical_vs_theoretical_full_500.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved to {filepath}")
    plt.close()


def plot_pilot_vs_full_comparison(pilot_sweep: AbstractionSweep, full_sweep: AbstractionSweep, 
                                   output_dir: str = "results"):
    """
    Compare pilot (50-ex) vs full (500-ex) sweep results.
    
    Args:
        pilot_sweep: Pilot sweep results
        full_sweep: Full sweep results  
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating pilot vs full comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Scaling Analysis: Pilot (50 ex) vs Full (500 ex) Dataset",
        fontsize=14,
        fontweight="bold"
    )
    
    loss_functions = ["utility", "brier", "cross-entropy"]
    colors = {"utility": "#1f77b4", "brier": "#ff7f0e", "cross-entropy": "#2ca02c"}
    
    for idx, loss_fn in enumerate(loss_functions):
        ax = axes[idx]
        
        pilot_results = pilot_sweep.results[loss_fn]
        full_results = full_sweep.results[loss_fn]
        
        lambdas = pilot_results["lambda"]
        
        # Plot both
        ax.plot(lambdas, pilot_results["accuracy"], "o-", label="Pilot (50 ex)", 
               linewidth=2, color=colors[loss_fn], alpha=0.6)
        ax.plot(lambdas, full_results["accuracy"], "s-", label="Full (500 ex)", 
               linewidth=2, color=colors[loss_fn], alpha=1.0)
        
        # Coverage as secondary
        ax_twin = ax.twinx()
        ax_twin.plot(lambdas, full_results["coverage"], "--", label="Coverage (full)",
                    color=colors[loss_fn], alpha=0.3, linewidth=1.5)
        
        ax.set_xlabel("Cost Weight λ", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax_twin.set_ylabel("Coverage", fontsize=11, alpha=0.5)
        ax.set_title(f"{loss_fn.replace('-', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    
    plt.tight_layout()
    filepath = output_dir / "pilot_vs_full_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved to {filepath}")
    plt.close()


def generate_summary_report(pilot_sweep: AbstractionSweep, full_sweep: AbstractionSweep,
                           output_file: str = None):
    """
    Generate comprehensive comparison summary report.
    
    Args:
        pilot_sweep: Pilot sweep results
        full_sweep: Full sweep results
        output_file: Optional JSON file to save report
        
    Returns:
        Dictionary with comprehensive comparison
    """
    report = {
        "title": "Empirical vs Theoretical Threshold Verification",
        "dataset_sizes": {
            "pilot": 50,
            "full": 500
        },
        "loss_functions": ["utility", "brier", "cross-entropy"],
        "comparison": {}
    }
    
    for loss_fn in ["utility", "brier", "cross-entropy"]:
        pilot_results = pilot_sweep.results[loss_fn]
        full_results = full_sweep.results[loss_fn]
        
        pilot_max_acc_idx = np.argmax(pilot_results["accuracy"])
        full_max_acc_idx = np.argmax(full_results["accuracy"])
        
        report["comparison"][loss_fn] = {
            "pilot_50": {
                "max_accuracy": float(pilot_results["accuracy"][pilot_max_acc_idx]),
                "optimal_lambda": float(pilot_results["lambda"][pilot_max_acc_idx]),
                "coverage_at_optimal": float(pilot_results["coverage"][pilot_max_acc_idx]),
            },
            "full_500": {
                "max_accuracy": float(full_results["accuracy"][full_max_acc_idx]),
                "optimal_lambda": float(full_results["lambda"][full_max_acc_idx]),
                "coverage_at_optimal": float(full_results["coverage"][full_max_acc_idx]),
            },
            "scaling_analysis": {
                "accuracy_change": float(
                    full_results["accuracy"][full_max_acc_idx] - 
                    pilot_results["accuracy"][pilot_max_acc_idx]
                ),
                "lambda_shift": float(
                    full_results["lambda"][full_max_acc_idx] - 
                    pilot_results["lambda"][pilot_max_acc_idx]
                )
            }
        }
    
    if output_file:
        logger.info(f"Saving summary report to {output_file}")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
    
    return report


def main():
    """Run Task 10 empirical vs theoretical comparison."""
    logger.info("\n" + "=" * 70)
    logger.info("TASK 10: EMPIRICAL VS THEORETICAL COMPARISON")
    logger.info("=" * 70)
    
    # File paths
    pilot_file = "results/sweep_results_pilot_50.json"
    full_file = "results/sweep_results_full_500.json"
    summary_file = "results/empirical_comparison_summary.json"
    
    # Verify files exist
    try:
        pilot_sweep, full_sweep = load_both_sweeps(pilot_file, full_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Both Task 7 (pilot) and Task 9 (full) sweeps must be completed first")
        return 1
    
    # Generate plots
    plot_empirical_vs_theoretical(full_sweep)
    plot_pilot_vs_full_comparison(pilot_sweep, full_sweep)
    
    # Generate summary report
    report = generate_summary_report(pilot_sweep, full_sweep, summary_file)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EMPIRICAL VERIFICATION SUMMARY")
    logger.info("=" * 70)
    
    for loss_fn, comparison in report["comparison"].items():
        logger.info(f"\n{loss_fn.upper()}:")
        logger.info(f"  Pilot (50 ex):  {comparison['pilot_50']['max_accuracy']:.1%} @ λ={comparison['pilot_50']['optimal_lambda']:.4f}")
        logger.info(f"  Full (500 ex):  {comparison['full_500']['max_accuracy']:.1%} @ λ={comparison['full_500']['optimal_lambda']:.4f}")
        diff = comparison["scaling_analysis"]["accuracy_change"]
        change_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
        logger.info(f"  Change:         {change_str} accuracy, λ shift={comparison['scaling_analysis']['lambda_shift']:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TASK 10 COMPLETE: Empirical vs theoretical comparison finished!")
    logger.info("=" * 70)
    logger.info("\nGenerated outputs:")
    logger.info("  - empirical_vs_theoretical_full_500.png")
    logger.info("  - pilot_vs_full_comparison.png")
    logger.info("  - empirical_comparison_summary.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
