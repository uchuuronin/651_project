"""
Abstention sweep algorithm: Evaluate abstention strategy across lambda values.
For each lambda, compute optimal abstention thresholds and evaluate performance.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import LOSS_FUNCTIONS
from src.thresholds import tau_B, tau_CE, tau_U

logger = logging.getLogger(__name__)


class AbstractionSweep:
    """
    Evaluate abstention strategies at different cost-benefit tradeoffs.
    Computes metrics (accuracy, coverage, precision) for each loss weighting lambda.
    """

    def __init__(self, loss_functions: List[str] = None):
        """
        Initialize sweep with loss functions to evaluate.

        Args:
            loss_functions: List of loss function names ("utility", "brier", "cross-entropy")
                           If None, uses all available functions
        """
        self.loss_functions = loss_functions or list(LOSS_FUNCTIONS.keys())
        self.results = {}

    def evaluate(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        lambdas: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Run abstention sweep across loss weightings.

        Args:
            confidences: Array of model confidence scores [0, 1]
            correct: Array of correctness (0/1) - whether model was correct
            lambdas: Array of loss weights to evaluate. If None, uses default range.

        Returns:
            Dict with results keyed by loss function name
        """
        if lambdas is None:
            lambdas = np.linspace(0, 0.25, 51)  # 51 points: 0, 0.005, ..., 0.25

        results = {}

        for loss_fn in self.loss_functions:
            logger.info(f"Evaluating {loss_fn} loss function...")
            results[loss_fn] = self._sweep_single_loss(
                loss_fn, confidences, correct, lambdas
            )

        self.results = results
        return results

    def _sweep_single_loss(
        self,
        loss_fn: str,
        confidences: np.ndarray,
        correct: np.ndarray,
        lambdas: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run sweep for a single loss function.

        Args:
            loss_fn: Loss function name
            confidences: Array of confidence scores
            correct: Array of correctness labels
            lambdas: Array of lambda values to evaluate

        Returns:
            Dict with metrics arrays for each lambda
        """
        threshold_fn = LOSS_FUNCTIONS[loss_fn]

        accuracies = []
        precisions = []
        coverages = []
        num_abstained = []

        for lam in lambdas:
            # Compute optimal threshold
            threshold = threshold_fn(lam)

            # Apply abstention: answer if confidence >= threshold
            mask_answer = confidences >= threshold
            num_answer = np.sum(mask_answer)
            num_abstain = len(confidences) - num_answer

            if num_answer > 0:
                # Accuracy on answered questions
                accuracy = np.mean(correct[mask_answer])
            else:
                accuracy = 0.0  # No questions answered

            # Coverage: fraction of questions answered
            coverage = num_answer / len(confidences)

            # Precision: accuracy on answered questions
            precision = accuracy

            accuracies.append(accuracy)
            precisions.append(precision)
            coverages.append(coverage)
            num_abstained.append(num_abstain)

        return {
            "lambda": lambdas,
            "accuracy": np.array(accuracies),
            "precision": np.array(precisions),
            "coverage": np.array(coverages),
            "num_abstained": np.array(num_abstained),
        }

    def compute_integrated_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute integrated metrics across sweep (e.g., AUC-like measures).

        Returns:
            Dict with integrated metrics for each loss function
        """
        metrics = {}

        for loss_fn, results in self.results.items():
            lambdas = results["lambda"]
            accuracy = results["accuracy"]
            coverage = results["coverage"]

            # AUC-like metrics (trapezoidal integration)
            auc_accuracy = np.trapz(accuracy, x=lambdas)
            auc_coverage = np.trapz(coverage, x=lambdas)

            # F1-like score: harmonic mean of accuracy and coverage
            eps = 1e-9
            f1_scores = 2 * accuracy * coverage / (accuracy + coverage + eps)
            max_f1 = np.max(f1_scores)

            metrics[loss_fn] = {
                "auc_accuracy": auc_accuracy,
                "auc_coverage": auc_coverage,
                "max_f1": max_f1,
                "mean_accuracy": np.mean(accuracy),
                "mean_coverage": np.mean(coverage),
            }

        return metrics

    def get_optimal_lambda(self, metric: str = "f1") -> Dict[str, float]:
        """
        Get optimal lambda for each loss function by a given metric.

        Args:
            metric: Metric to optimize ("accuracy", "coverage", "f1")

        Returns:
            Dict mapping loss function names to optimal lambda
        """
        optimal = {}

        for loss_fn, results in self.results.items():
            lambdas = results["lambda"]

            if metric == "accuracy":
                scores = results["accuracy"]
            elif metric == "coverage":
                scores = results["coverage"]
            elif metric == "f1":
                accuracy = results["accuracy"]
                coverage = results["coverage"]
                eps = 1e-9
                scores = 2 * accuracy * coverage / (accuracy + coverage + eps)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            idx_optimal = np.argmax(scores)
            optimal[loss_fn] = float(lambdas[idx_optimal])

        return optimal


def compute_metrics_from_results(
    results: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract confidence and correctness arrays from inference results.

    Args:
        results: List of result dicts from inference

    Returns:
        Tuple of (confidences, correct) numpy arrays
    """
    confidences = []
    correct = []

    for result in results:
        confidence = result.get("confidence", 0.5)
        ground_truth = result.get("ground_truth", "").strip()
        pred = result.get("prediction", "").strip()

        # Check if correct (simple string matching)
        is_correct = 0.0
        if ground_truth and pred:
            if ground_truth.lower() in pred.lower() or pred.lower() in ground_truth.lower():
                is_correct = 1.0

        confidences.append(confidence)
        correct.append(is_correct)

    return np.array(confidences), np.array(correct)
