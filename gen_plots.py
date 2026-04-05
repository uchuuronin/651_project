"""
Generate reliability diagrams for model calibration analysis.
Visualizes expected vs actual accuracy at different confidence levels.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.thresholds import tau_U, tau_B, tau_CE

logger = logging.getLogger(__name__)


def compute_reliability_curve(
    results: List[Dict[str, Any]],
    num_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data points.

    Args:
        results: List of result dicts with 'prediction', 'ground_truth', 'confidence'
        num_bins: Number of bins for confidence intervals

    Returns:
        Tuple of (expected_conf, actual_acc, counts, bin_edges)
    """
    confidences = []
    correct = []

    for result in results:
        conf = result.get("confidence", 0.5)
        ground_truth = result.get("ground_truth", "").strip()
        pred = result.get("prediction", "").strip()

        # Check correctness (simple string matching on first token)
        if ground_truth and pred:
            is_correct = ground_truth.lower() in pred.lower() or pred.lower() in ground_truth.lower()
        else:
            is_correct = 0.0

        confidences.append(conf)
        correct.append(1.0 if is_correct else 0.0)

    confidences = np.array(confidences)
    correct = np.array(correct)

    # Create bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    expected_conf = []
    actual_acc = []
    counts = []

    for i in range(len(bin_edges) - 1):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # Include right edge in last bin
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            expected_conf.append(np.mean(confidences[mask]))
            actual_acc.append(np.mean(correct[mask]))
            counts.append(np.sum(mask))
        else:
            expected_conf.append(bin_centers[i])
            actual_acc.append(0.0)
            counts.append(0)

    return np.array(expected_conf), np.array(actual_acc), np.array(counts), bin_edges


def plot_reliability_diagram(
    results: List[Dict[str, Any]],
    output_path: Path,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Generate and save reliability diagram.

    Args:
        results: List of result dicts
        output_path: Path to save PNG
        num_bins: Number of bins

    Returns:
        Dict with calibration metrics (ECE, MCE, etc.)
    """
    expected_conf, actual_acc, counts, _ = compute_reliability_curve(results, num_bins)

    # Filter out empty bins
    mask = counts > 0
    expected_conf = expected_conf[mask]
    actual_acc = actual_acc[mask]
    counts = counts[mask]

    # Compute metrics
    overall_accuracy = np.mean([1.0 if r.get("ground_truth", "").strip() and (
        r.get("ground_truth", "").lower() in r.get("prediction", "").lower() or
        r.get("prediction", "").lower() in r.get("ground_truth", "").lower()
    ) else 0.0 for r in results])

    ece = np.mean(np.abs(expected_conf - actual_acc))  # Expected Calibration Error
    mce = np.max(np.abs(expected_conf - actual_acc))  # Maximum Calibration Error

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.3)

    # Reliability curve
    ax.scatter(expected_conf, actual_acc, s=counts * 3, alpha=0.6, label="Model")
    ax.plot(expected_conf, actual_acc, "o-", alpha=0.5)

    ax.set_xlabel("Expected Confidence (Predicted Probability)")
    ax.set_ylabel("Actual Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Title with metrics
    title = f"Reliability Diagram\nECE: {ece:.3f}, MCE: {mce:.3f}, Accuracy: {overall_accuracy:.3f}"
    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved reliability diagram to {output_path}")
    plt.close()

    return {
        "overall_accuracy": overall_accuracy,
        "ece": float(ece),
        "mce": float(mce),
        "num_examples": len(results),
    }


def plot_theoretical_curves(output_path: Path) -> None:
    """
    Generate comparison plot of all three threshold functions.

    Args:
        output_path: Path to save PNG
    """
    # Generate lambda values (loss weights)
    lambdas = np.linspace(0, 0.25, 1000)

    # Compute thresholds
    tau_u_vals = tau_U(lambdas)
    tau_b_vals = tau_B(lambdas)
    tau_ce_vals = tau_CE(lambdas)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(lambdas, tau_u_vals, label=r"$\tau_U(\lambda)$ (Utility)", linewidth=2)
    ax.plot(lambdas, tau_b_vals, label=r"$\tau_B(\lambda)$ (Brier)", linewidth=2)
    ax.plot(lambdas, tau_ce_vals, label=r"$\tau_{CE}(\lambda)$ (Cross-Entropy)", linewidth=2)

    ax.set_xlabel(r"Loss Weight $\lambda$", fontsize=12)
    ax.set_ylabel(r"Abstention Threshold $\tau$", fontsize=12)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")
    ax.set_title("Optimal Abstention Thresholds by Loss Function", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved theoretical curves to {output_path}")
    plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load pilot results
    results_path = Path("results/pilot_50_examples.json")
    with open(results_path) as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} results from {results_path}")

    # Generate reliability diagram
    output_path = Path("results/reliability_diagram_pilot.png")
    metrics = plot_reliability_diagram(results, output_path, num_bins=5)

    # Print metrics
    print("\nCalibration Metrics:")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"  ECE (Expected Calibration Error): {metrics['ece']:.3f}")
    print(f"  MCE (Maximum Calibration Error): {metrics['mce']:.3f}")
    print(f"  Examples: {metrics['num_examples']}")

    # Generate theoretical comparison figure
    print("\nGenerating theoretical threshold curves...")
    output_path_theo = Path("results/theoretical_thresholds.png")
    plot_theoretical_curves(output_path_theo)

    return 0


if __name__ == "__main__":
    exit(main())
