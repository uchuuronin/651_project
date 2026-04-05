"""
Run abstention sweep on pilot 50-example dataset.
Task 6: Test sweep functionality on pilot data.
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.sweep import AbstractionSweep, compute_metrics_from_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run sweep on 50-example pilot dataset."""
    # Paths
    pilot_file = Path("results/pilot_50_examples.json")
    output_file = Path("results/sweep_results_pilot_50.json")

    if not pilot_file.exists():
        logger.error(f"Pilot results file not found: {pilot_file}")
        return 1

    # Load pilot results
    logger.info(f"Loading pilot results from {pilot_file}")
    with open(pilot_file, "r") as f:
        pilot_results = json.load(f)

    if isinstance(pilot_results, dict) and "results" in pilot_results:
        results = pilot_results["results"]
    else:
        results = pilot_results

    logger.info(f"Loaded {len(results)} examples")

    # Extract confidences and correctness
    confidences, correct = compute_metrics_from_results(results)
    logger.info(f"Extracted confidences and correctness labels")
    logger.info(f"  - Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    logger.info(f"  - Accuracy: {np.mean(correct):.1%}")

    # Create and run sweep
    logger.info("Running sweep with all three loss functions...")
    sweep = AbstractionSweep(loss_functions=["utility", "brier", "cross-entropy"])
    sweep.evaluate(confidences, correct)

    # Save results
    logger.info(f"Saving sweep results to {output_file}")
    sweep.save_results(str(output_file))

    # Compute and display integrated metrics
    integrated_metrics = sweep.compute_integrated_metrics()
    logger.info("Integrated metrics:")
    for loss_fn, metrics in integrated_metrics.items():
        logger.info(f"  {loss_fn}:")
        for metric_name, value in metrics.items():
            logger.info(f"    {metric_name}: {value:.4f}")

    # Get optimal lambdas for each metric
    logger.info("Optimal lambdas:")
    for metric in ["accuracy", "coverage", "f1"]:
        optimal = sweep.get_optimal_lambda(metric=metric)
        logger.info(f"  {metric}:")
        for loss_fn, lam in optimal.items():
            logger.info(f"    {loss_fn}: {lam:.4f}")

    logger.info("Task 6 sweep pilot completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
