#!/usr/bin/env python3
"""
Run abstention sweep on full 500-example dataset.
Task 9: Execute full empirical evaluation sweep.

Note: This runs after Task 8 (full 500-example inference) is complete.
Expected runtime: ~5 minutes for 51 lambda points x 3 loss functions.
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.sweep import AbstractionSweep, compute_metrics_from_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_full_dataset_exists(file_path: Path) -> bool:
    """
    Verify that full 500-example dataset is available.
    
    Args:
        file_path: Path to full inference results
        
    Returns:
        True if file exists and contains valid results
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Full dataset not found: {file_path}")
    
    with open(file_path, "r") as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} results from full dataset")
    
    if len(results) < 400:
        logger.warning(f"Expected ~500 results but got {len(results)}")
    
    return True


def main():
    """Run sweep on full 500-example dataset."""
    # Input and output paths
    full_results_file = Path("results/full_500_examples.json")
    output_file = Path("results/sweep_results_full_500.json")
    
    # Verify dataset exists
    logger.info("=" * 70)
    logger.info("TASK 9: FULL SWEEP ON 500-EXAMPLE DATASET")
    logger.info("=" * 70)
    
    try:
        verify_full_dataset_exists(full_results_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Task 8 (full 500-example inference) must be completed first")
        return 1
    
    # Load full dataset
    logger.info(f"\nLoading full dataset from {full_results_file}")
    with open(full_results_file, "r") as f:
        full_results = json.load(f)
    
    logger.info(f"Loaded {len(full_results)} examples for sweep evaluation")
    
    # Extract confidences and correctness
    confidences, correct = compute_metrics_from_results(full_results)
    logger.info(f"\nExtracted metrics from {len(confidences)} examples")
    logger.info(f"  - Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    logger.info(f"  - Mean confidence: {confidences.mean():.3f}")
    logger.info(f"  - Dataset accuracy: {np.mean(correct):.1%} ({np.sum(correct):.0f}/{len(correct)})")
    
    # Create and run sweep
    logger.info(f"\nInitializing AbstractionSweep with all three loss functions...")
    sweep = AbstractionSweep(loss_functions=["utility", "brier", "cross-entropy"])
    
    logger.info("Running full sweep evaluation (51 lambda points × 3 loss functions)...")
    sweep.evaluate(confidences, correct)
    
    # Save results
    logger.info(f"\nSaving sweep results to {output_file}")
    sweep.save_results(str(output_file))
    
    # Compute integrated metrics
    logger.info("\nIntegrated metrics across sweep:")
    integrated_metrics = sweep.compute_integrated_metrics()
    for loss_fn, metrics in integrated_metrics.items():
        logger.info(f"  {loss_fn}:")
        for metric_name, value in metrics.items():
            if metric_name.startswith("auc"):
                logger.info(f"    {metric_name}: {value:.4f}")
            else:
                logger.info(f"    {metric_name}: {value:.4f}")
    
    # Get optimal lambdas
    logger.info("\nOptimal lambda values (per metric):")
    for metric in ["accuracy", "coverage", "f1"]:
        optimal = sweep.get_optimal_lambda(metric=metric)
        logger.info(f"  {metric}: {optimal}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TASK 9 COMPLETE: Full sweep on 500-example dataset finished!")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("Next: Task 10 (generate empirical vs theoretical comparison plots)")
    
    return 0


if __name__ == "__main__":
    exit(main())
