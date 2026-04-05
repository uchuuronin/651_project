"""
Comprehensive validation and analysis of pilot sweep results.
Task 7: Sweep validation on 50-example pilot dataset.
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


def load_and_validate_sweep_results(filepath: str) -> AbstractionSweep:
    """
    Load sweep results and perform basic validation.
    
    Args:
        filepath: Path to sweep results JSON file
        
    Returns:
        AbstractionSweep instance with loaded results
    """
    logger.info(f"Loading sweep results from {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Sweep results file not found: {filepath}")
    
    sweep = AbstractionSweep.load_results(filepath)
    
    logger.info(f"Loaded results for loss functions: {sweep.loss_functions}")
    
    # Validation checks
    logger.info("Running validation checks...")
    
    for loss_fn, results in sweep.results.items():
        lambdas = results["lambda"]
        accuracy = results["accuracy"]
        coverage = results["coverage"]
        precision = results["precision"]
        
        logger.info(f"\n{loss_fn}:")
        logger.info(f"  Lambda range: [{lambdas[0]:.4f}, {lambdas[-1]:.4f}]")
        logger.info(f"  Accuracy range: [{accuracy.min():.4f}, {accuracy.max():.4f}]")
        logger.info(f"  Coverage range: [{coverage.min():.4f}, {coverage.max():.4f}]")
        logger.info(f"  Precision range: [{precision.min():.4f}, {precision.max():.4f}]")
        
        # Verify ranges
        assert np.all(accuracy >= 0) and np.all(accuracy <= 1), f"Invalid accuracy for {loss_fn}"
        assert np.all(coverage >= 0) and np.all(coverage <= 1), f"Invalid coverage for {loss_fn}"
        assert np.all(precision >= 0) and np.all(precision <= 1), f"Invalid precision for {loss_fn}"
        
        logger.info(f"  ✓ All metrics in valid ranges [0, 1]")
    
    return sweep


def analyze_abstention_tradeoffs(sweep: AbstractionSweep) -> None:
    """
    Analyze and report on the abstention-accuracy tradeoffs.
    
    Args:
        sweep: Validated AbstractionSweep instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("ABSTENTION-ACCURACY TRADEOFF ANALYSIS")
    logger.info("=" * 70)
    
    for loss_fn, results in sweep.results.items():
        lambdas = results["lambda"]
        accuracy = results["accuracy"]
        coverage = results["coverage"]
        precision = results["precision"]
        num_abstained = results["num_abstained"]
        
        logger.info(f"\n{loss_fn.upper()} Loss Function:")
        logger.info("-" * 70)
        
        # Find key operating points
        max_acc_idx = np.argmax(accuracy)
        min_cov_idx = np.argmin(coverage)
        max_f1_idx = np.argmax(2 * accuracy * coverage / (accuracy + coverage + 1e-9))
        
        logger.info(f"  Maximum Accuracy Operating Point:")
        logger.info(f"    λ = {lambdas[max_acc_idx]:.4f}")
        logger.info(f"    Accuracy = {accuracy[max_acc_idx]:.1%}")
        logger.info(f"    Coverage = {coverage[max_acc_idx]:.1%}")
        logger.info(f"    Questions abstained = {int(num_abstained[max_acc_idx])}")
        
        logger.info(f"\n  Minimum Coverage Operating Point (Highest Abstention):")
        logger.info(f"    λ = {lambdas[min_cov_idx]:.4f}")
        logger.info(f"    Accuracy = {accuracy[min_cov_idx]:.1%}")
        logger.info(f"    Coverage = {coverage[min_cov_idx]:.1%}")
        logger.info(f"    Questions abstained = {int(num_abstained[min_cov_idx])}")
        
        logger.info(f"\n  Optimal F1 Operating Point (Balanced):")
        logger.info(f"    λ = {lambdas[max_f1_idx]:.4f}")
        logger.info(f"    Accuracy = {accuracy[max_f1_idx]:.1%}")
        logger.info(f"    Coverage = {coverage[max_f1_idx]:.1%}")
        logger.info(f"    Questions abstained = {int(num_abstained[max_f1_idx])}")


def compare_loss_functions(sweep: AbstractionSweep) -> None:
    """
    Compare behavior across the three loss functions.
    
    Args:
        sweep: Validated AbstractionSweep instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("LOSS FUNCTION COMPARISON")
    logger.info("=" * 70)
    
    # Get max accuracy for each loss function
    logger.info("\nMax Accuracy Comparison:")
    for loss_fn, results in sweep.results.items():
        max_acc = np.max(results["accuracy"])
        opt_idx = np.argmax(results["accuracy"])
        cov_at_max_acc = results["coverage"][opt_idx]
        lam_at_max_acc = results["lambda"][opt_idx]
        
        logger.info(f"  {loss_fn}: {max_acc:.1%} (λ={lam_at_max_acc:.4f}, coverage={cov_at_max_acc:.1%})")
    
    # Get max coverage for each loss function
    logger.info("\nMax Coverage Comparison:")
    for loss_fn, results in sweep.results.items():
        max_cov = np.max(results["coverage"])
        opt_idx = np.argmax(results["coverage"])
        acc_at_max_cov = results["accuracy"][opt_idx]
        lam_at_max_cov = results["lambda"][opt_idx]
        
        logger.info(f"  {loss_fn}: {max_cov:.1%} (λ={lam_at_max_cov:.4f}, accuracy={acc_at_max_cov:.1%})")


def plot_tradeoff_curves(sweep: AbstractionSweep, output_dir: str = "results") -> None:
    """
    Create coverage vs accuracy tradeoff curves for each loss function.
    
    Args:
        sweep: Validated AbstractionSweep instance
        output_dir: Directory to save plots
    """
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING TRADEOFF CURVE PLOTS")
    logger.info("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Individual tradeoff curves (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Abstention Tradeoff Curves (Pilot 50 Examples)", fontsize=14, fontweight="bold")
    
    colors = {"utility": "#1f77b4", "brier": "#ff7f0e", "cross-entropy": "#2ca02c"}
    
    for idx, (loss_fn, results) in enumerate(sweep.results.items()):
        ax = axes[idx]
        lambdas = results["lambda"]
        accuracy = results["accuracy"]
        coverage = results["coverage"]
        
        ax.plot(coverage, accuracy, "o-", color=colors[loss_fn], linewidth=2, markersize=4)
        
        # Highlight key points
        max_acc_idx = np.argmax(accuracy)
        min_cov_idx = np.argmin(coverage)
        
        ax.plot(
            coverage[max_acc_idx], 
            accuracy[max_acc_idx], 
            "r*", 
            markersize=20, 
            label=f"Max Accuracy (λ={lambdas[max_acc_idx]:.3f})"
        )
        ax.plot(
            coverage[min_cov_idx], 
            accuracy[min_cov_idx], 
            "bs", 
            markersize=10, 
            label=f"Max Abstention (λ={lambdas[min_cov_idx]:.3f})"
        )
        
        ax.set_xlabel("Coverage (Fraction Answered)", fontsize=11)
        ax.set_ylabel("Accuracy (Fraction Correct)", fontsize=11)
        ax.set_title(f"{loss_fn.replace('-', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    
    filepath = output_dir / "sweep_tradeoff_curves_pilot.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved tradeoff curves to {filepath}")
    plt.close()
    
    # Plot 2: Combined comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for loss_fn, results in sweep.results.items():
        coverage = results["coverage"]
        accuracy = results["accuracy"]
        ax.plot(coverage, accuracy, "o-", label=loss_fn, linewidth=2, color=colors[loss_fn])
    
    ax.set_xlabel("Coverage (Fraction Answered)", fontsize=12)
    ax.set_ylabel("Accuracy (Fraction Correct)", fontsize=12)
    ax.set_title("Coverage vs Accuracy: All Loss Functions (Pilot 50 Examples)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    filepath = output_dir / "sweep_comparison_pilot.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved comparison plot to {filepath}")
    plt.close()


def plot_lambda_metrics(sweep: AbstractionSweep, output_dir: str = "results") -> None:
    """
    Plot how metrics vary with lambda for each loss function.
    
    Args:
        sweep: Validated AbstractionSweep instance
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Metrics vs Lambda (Pilot 50 Examples)", fontsize=14, fontweight="bold")
    
    colors = {"utility": "#1f77b4", "brier": "#ff7f0e", "cross-entropy": "#2ca02c"}
    
    # Accuracy vs lambda
    ax = axes[0]
    for loss_fn, results in sweep.results.items():
        ax.plot(results["lambda"], results["accuracy"], "o-", label=loss_fn, color=colors[loss_fn])
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy vs Lambda", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Coverage vs lambda
    ax = axes[1]
    for loss_fn, results in sweep.results.items():
        ax.plot(results["lambda"], results["coverage"], "o-", label=loss_fn, color=colors[loss_fn])
    ax.set_ylabel("Coverage", fontsize=11)
    ax.set_title("Coverage vs Lambda (Fraction Answered)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Precision vs lambda
    ax = axes[2]
    for loss_fn, results in sweep.results.items():
        ax.plot(results["lambda"], results["precision"], "o-", label=loss_fn, color=colors[loss_fn])
    ax.set_xlabel("Lambda (Abstention Cost)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Lambda", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    filepath = output_dir / "sweep_lambda_metrics_pilot.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved lambda metrics plot to {filepath}")
    plt.close()


def validate_against_theory(sweep: AbstractionSweep) -> None:
    """
    Validate sweep results against theoretical threshold functions.
    
    Args:
        sweep: Validated AbstractionSweep instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("THEORETICAL VALIDATION")
    logger.info("=" * 70)
    
    # Load theoretical thresholds
    logger.info("\nComputing theoretical thresholds for observed lambdas...")
    
    for loss_fn, results in sweep.results.items():
        lambdas = results["lambda"]
        threshold_fn = LOSS_FUNCTIONS[loss_fn]
        
        # Compute theoretical thresholds
        theoretical_thresholds = np.array([threshold_fn(lam) for lam in lambdas])
        
        logger.info(f"\n{loss_fn}:")
        logger.info(f"  Threshold range: [{theoretical_thresholds.min():.4f}, {theoretical_thresholds.max():.4f}]")
        
        # Sample a few lambdas to show threshold values
        sample_indices = [0, len(lambdas) // 4, len(lambdas) // 2, 3 * len(lambdas) // 4, -1]
        for idx in sample_indices:
            lam = lambdas[idx]
            tau = theoretical_thresholds[idx]
            acc = results["accuracy"][idx]
            cov = results["coverage"][idx]
            logger.info(f"    λ={lam:.4f}: τ={tau:.4f} → Accuracy={acc:.1%}, Coverage={cov:.1%}")


def generate_summary_report(sweep: AbstractionSweep, output_file: str = None) -> dict:
    """
    Generate a summary report of sweep validation results.
    
    Args:
        sweep: Validated AbstractionSweep instance
        output_file: Optional JSON file to save summary
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "dataset": "TriviaQA (50-example pilot)",
        "loss_functions": sweep.loss_functions,
        "lambda_range": [0.0, 0.25],
        "num_lambda_points": 51,
        "results": {}
    }
    
    for loss_fn, results in sweep.results.items():
        lambdas = results["lambda"]
        accuracy = results["accuracy"]
        coverage = results["coverage"]
        precision = results["precision"]
        
        # Compute integrated metrics
        auc_accuracy = np.trapz(accuracy, x=lambdas)
        auc_coverage = np.trapz(coverage, x=lambdas)
        
        # F1-like metric
        f1_scores = 2 * accuracy * coverage / (accuracy + coverage + 1e-9)
        max_f1 = np.max(f1_scores)
        
        summary["results"][loss_fn] = {
            "max_accuracy": float(np.max(accuracy)),
            "min_accuracy": float(np.min(accuracy)),
            "mean_accuracy": float(np.mean(accuracy)),
            "max_coverage": float(np.max(coverage)),
            "min_coverage": float(np.min(coverage)),
            "mean_coverage": float(np.mean(coverage)),
            "auc_accuracy": float(auc_accuracy),
            "auc_coverage": float(auc_coverage),
            "max_f1": float(max_f1),
            "optimal_lambda_accuracy": float(lambdas[np.argmax(accuracy)]),
            "optimal_lambda_f1": float(lambdas[np.argmax(f1_scores)]),
        }
    
    if output_file:
        logger.info(f"\nSaving summary report to {output_file}")
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    return summary


def main():
    """Run complete sweep validation and analysis."""
    logger.info("\n" + "=" * 70)
    logger.info("TASK 7: SWEEP VALIDATION ON 50-EXAMPLE PILOT")
    logger.info("=" * 70)
    
    # Load and validate
    sweep_file = "results/sweep_results_pilot_50.json"
    sweep = load_and_validate_sweep_results(sweep_file)
    
    # Analyze tradeoffs
    analyze_abstention_tradeoffs(sweep)
    compare_loss_functions(sweep)
    validate_against_theory(sweep)
    
    # Generate plots
    plot_tradeoff_curves(sweep)
    plot_lambda_metrics(sweep)
    
    # Generate summary report
    summary = generate_summary_report(sweep, "results/sweep_validation_summary.json")
    
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nGenerated outputs:")
    logger.info("  - sweep_tradeoff_curves_pilot.png (individual loss function curves)")
    logger.info("  - sweep_comparison_pilot.png (comparative plot)")
    logger.info("  - sweep_lambda_metrics_pilot.png (metrics vs lambda)")
    logger.info("  - sweep_validation_summary.json (summary statistics)")
    logger.info("\nTask 7 complete: Sweep validation on pilot data finished!")
    
    return 0


if __name__ == "__main__":
    exit(main())
