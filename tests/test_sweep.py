"""
Integration tests for the abstention sweep functionality.
Task 6 sweep integration testing.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.sweep import AbstractionSweep, compute_metrics_from_results


class TestSweepIntegration:
    """Integration tests for AbstractionSweep class."""

    @staticmethod
    def create_synthetic_data(n_samples=100, seed=42):
        """Create synthetic confidence and correctness data."""
        np.random.seed(seed)
        confidences = np.random.uniform(0.5, 1.0, n_samples)
        correct = np.random.binomial(1, 0.6, n_samples)
        return confidences, correct

    def test_sweep_runs_without_error(self):
        """Test that sweep executes without errors on valid data."""
        confidences, correct = self.create_synthetic_data()
        sweep = AbstractionSweep()
        results = sweep.evaluate(confidences, correct)

        # Check structure
        assert isinstance(results, dict)
        assert "utility" in results
        assert "brier" in results
        assert "cross-entropy" in results

    def test_sweep_output_structure(self):
        """Test that sweep returns correct output structure."""
        confidences, correct = self.create_synthetic_data()
        sweep = AbstractionSweep()
        results = sweep.evaluate(confidences, correct)

        for loss_fn, res_dict in results.items():
            assert "lambda" in res_dict
            assert "accuracy" in res_dict
            assert "precision" in res_dict
            assert "coverage" in res_dict
            assert "num_abstained" in res_dict

            # All arrays should have same length (51 lambdas)
            assert len(res_dict["lambda"]) == 51
            assert len(res_dict["accuracy"]) == 51
            assert len(res_dict["precision"]) == 51
            assert len(res_dict["coverage"]) == 51
            assert len(res_dict["num_abstained"]) == 51

    def test_metrics_are_valid_ranges(self):
        """Test that all metrics are in valid ranges."""
        confidences, correct = self.create_synthetic_data()
        sweep = AbstractionSweep()
        sweep.evaluate(confidences, correct)

        for loss_fn, res_dict in sweep.results.items():
            accuracy = res_dict["accuracy"]
            coverage = res_dict["coverage"]
            precision = res_dict["precision"]
            num_abstained = res_dict["num_abstained"]

            # All should be non-negative
            assert np.all(accuracy >= 0)
            assert np.all(coverage >= 0)
            assert np.all(precision >= 0)
            assert np.all(num_abstained >= 0)

            # Accuracy and precision should be <= 1
            assert np.all(accuracy <= 1)
            assert np.all(precision <= 1)

            # Coverage should be <= 1
            assert np.all(coverage <= 1)

            # num_abstained should be <= n_samples
            assert np.all(num_abstained <= len(confidences))

    def test_save_and_load_results(self):
        """Test that sweep results can be saved and loaded."""
        confidences, correct = self.create_synthetic_data()
        sweep1 = AbstractionSweep()
        sweep1.evaluate(confidences, correct)

        # Save results
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "sweep_test.json"
            sweep1.save_results(str(filepath))

            # Load results
            sweep2 = AbstractionSweep.load_results(str(filepath))

            # Check that loaded results match original
            assert sweep1.loss_functions == sweep2.loss_functions
            for loss_fn in sweep1.loss_functions:
                np.testing.assert_array_almost_equal(
                    sweep1.results[loss_fn]["lambda"],
                    sweep2.results[loss_fn]["lambda"],
                )
                np.testing.assert_array_almost_equal(
                    sweep1.results[loss_fn]["accuracy"],
                    sweep2.results[loss_fn]["accuracy"],
                )

    def test_integrated_metrics(self):
        """Test that integrated metrics are computed correctly."""
        confidences, correct = self.create_synthetic_data()
        sweep = AbstractionSweep()
        sweep.evaluate(confidences, correct)

        metrics = sweep.compute_integrated_metrics()

        # Check structure
        for loss_fn in ["utility", "brier", "cross-entropy"]:
            assert loss_fn in metrics
            metric_dict = metrics[loss_fn]
            assert "auc_accuracy" in metric_dict
            assert "auc_coverage" in metric_dict
            assert "max_f1" in metric_dict
            assert "mean_accuracy" in metric_dict
            assert "mean_coverage" in metric_dict

            # All should be valid floats
            for value in metric_dict.values():
                assert isinstance(value, (float, np.floating))
                assert not np.isnan(value)
                assert not np.isinf(value)

    def test_get_optimal_lambda(self):
        """Test that optimal lambda selection works for all metrics."""
        confidences, correct = self.create_synthetic_data()
        sweep = AbstractionSweep()
        sweep.evaluate(confidences, correct)

        for metric in ["accuracy", "coverage", "f1"]:
            optimal = sweep.get_optimal_lambda(metric=metric)

            # Should have entry for each loss function
            assert len(optimal) == 3
            for loss_fn in ["utility", "brier", "cross-entropy"]:
                assert loss_fn in optimal
                lam = optimal[loss_fn]
                # Should be in valid range
                assert 0 <= lam <= 0.25
                # Should be from the lambda grid
                assert lam in sweep.results[loss_fn]["lambda"]

    def test_compute_metrics_from_results(self):
        """Test extracting metrics from inference results."""
        results = [
            {
                "confidence": 0.9,
                "prediction": "Paris",
                "ground_truth": "Paris",
            },
            {
                "confidence": 0.7,
                "prediction": "Rome",
                "ground_truth": "London",
            },
            {
                "confidence": 0.8,
                "prediction": "Berlin",
                "ground_truth": "Berlin",
            },
        ]

        confidences, correct = compute_metrics_from_results(results)

        # Check counts
        assert len(confidences) == 3
        assert len(correct) == 3

        # Check values
        np.testing.assert_array_almost_equal(
            confidences, [0.9, 0.7, 0.8]
        )
        np.testing.assert_array_almost_equal(correct, [1.0, 0.0, 1.0])

    def test_different_loss_function_subsets(self):
        """Test sweep with different subsets of loss functions."""
        confidences, correct = self.create_synthetic_data()

        # Test with single loss function
        sweep = AbstractionSweep(loss_functions=["utility"])
        results = sweep.evaluate(confidences, correct)
        assert list(results.keys()) == ["utility"]

        # Test with two loss functions
        sweep = AbstractionSweep(loss_functions=["utility", "brier"])
        results = sweep.evaluate(confidences, correct)
        assert set(results.keys()) == {"utility", "brier"}

    def test_sweep_with_perfect_predictions(self):
        """Test sweep when all predictions are correct."""
        n_samples = 50
        confidences = np.random.uniform(0.5, 1.0, n_samples)
        correct = np.ones(n_samples)  # All correct

        sweep = AbstractionSweep()
        sweep.evaluate(confidences, correct)

        # Check that we get valid results
        for loss_fn, res_dict in sweep.results.items():
            accuracy = res_dict["accuracy"]
            # With all correct, high-coverage predictions should have high accuracy
            assert np.max(accuracy) > 0.5

    def test_sweep_with_random_predictions(self):
        """Test sweep when all predictions are random (no confidence signal)."""
        n_samples = 50
        confidences = np.random.uniform(0.5, 1.0, n_samples)
        correct = np.random.binomial(1, 0.5, n_samples)

        sweep = AbstractionSweep()
        sweep.evaluate(confidences, correct)

        # Should still complete without errors
        assert len(sweep.results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
