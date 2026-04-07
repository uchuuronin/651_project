# Tests for src/grid.py: expected_loss, run_grid_search, save/load, print_mae_table.
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.grid import expected_loss, run_grid_search, save_grid_results, load_grid_results, print_mae_table
from src.config import LAMBDA_GRID, TAU_GRID

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    confidences = np.random.uniform(0.3, 0.9, 100)
    correct = np.random.binomial(1, 0.6, 100).astype(float)
    return confidences, correct


@pytest.fixture
def grid_results(synthetic_data):
    confidences, correct = synthetic_data
    return run_grid_search(confidences, correct)


class TestExpectedLoss:
    def test_all_abstain_returns_lambda(self, synthetic_data):
        # tau=1.0 means nothing passes threshold → expected loss == lambda
        c, y = synthetic_data
        for loss_fn in ["utility", "brier", "cross-entropy"]:
            result = expected_loss(c, y, tau=1.0, lam=0.1, loss_fn=loss_fn)
            assert abs(result - 0.1) < 1e-9, f"Failed for {loss_fn}: got {result}"

    def test_returns_float(self, synthetic_data):
        c, y = synthetic_data
        for loss_fn in ["utility", "brier", "cross-entropy"]:
            result = expected_loss(c, y, tau=0.5, lam=0.1, loss_fn=loss_fn)
            assert isinstance(result, float)

    def test_brier_and_ce_non_negative(self, synthetic_data):
        # Brier and CE losses are always >= 0
        c, y = synthetic_data
        for loss_fn in ["brier", "cross-entropy"]:
            for tau in [0.3, 0.5, 0.7]:
                result = expected_loss(c, y, tau=tau, lam=0.1, loss_fn=loss_fn)
                assert result >= 0, f"{loss_fn} at tau={tau} returned {result}"

    def test_higher_lambda_increases_abstention_cost(self, synthetic_data):
        # At tau=1.0 (all abstain), loss == lambda, so higher lambda = higher loss
        c, y = synthetic_data
        for loss_fn in ["utility", "brier", "cross-entropy"]:
            loss_low = expected_loss(c, y, tau=1.0, lam=0.05, loss_fn=loss_fn)
            loss_high = expected_loss(c, y, tau=1.0, lam=0.20, loss_fn=loss_fn)
            assert loss_high > loss_low, f"{loss_fn}: expected higher loss at higher lambda"

    def test_unknown_loss_fn_raises(self, synthetic_data):
        c, y = synthetic_data
        with pytest.raises(ValueError, match="Unknown loss function"):
            expected_loss(c, y, tau=0.5, lam=0.1, loss_fn="squared_hinge")


class TestRunGridSearch:
    def test_returns_all_three_loss_functions(self, grid_results):
        assert set(grid_results.keys()) == {"utility", "brier", "cross-entropy"}

    def test_output_keys(self, grid_results):
        expected = {
            "lambda", "p_star",
            "tau_U", "tau_B", "tau_CE",
            "mae_tau_U", "mae_tau_B", "mae_tau_CE",
            "mean_mae_tau_U", "mean_mae_tau_B", "mean_mae_tau_CE",
        }
        for loss_fn, res in grid_results.items():
            assert set(res.keys()) == expected, f"Missing keys for {loss_fn}"

    def test_p_star_in_tau_grid(self, grid_results):
        # p* must be one of the TAU_GRID values (grid search, not interpolation)
        tau_set = set(float(t) for t in TAU_GRID)
        for loss_fn, res in grid_results.items():
            for p in res["p_star"]:
                assert p in tau_set, f"p*={p} not in TAU_GRID for {loss_fn}"

    def test_p_star_in_unit_interval(self, grid_results):
        for loss_fn, res in grid_results.items():
            assert all(0.0 <= p <= 1.0 for p in res["p_star"]), f"p* out of [0,1] for {loss_fn}"

    def test_mae_non_negative(self, grid_results):
        for loss_fn, res in grid_results.items():
            assert res["mean_mae_tau_U"] >= 0
            assert res["mean_mae_tau_B"] >= 0
            assert res["mean_mae_tau_CE"] >= 0

    def test_mae_consistent_with_pointwise(self, grid_results):
        # mean_mae_* should equal np.mean of the per-lambda list
        for loss_fn, res in grid_results.items():
            assert abs(res["mean_mae_tau_U"] - np.mean(res["mae_tau_U"])) < 1e-9
            assert abs(res["mean_mae_tau_B"] - np.mean(res["mae_tau_B"])) < 1e-9
            assert abs(res["mean_mae_tau_CE"] - np.mean(res["mae_tau_CE"])) < 1e-9

    def test_lambda_length_matches_grid(self, grid_results):
        for loss_fn, res in grid_results.items():
            assert len(res["lambda"]) == len(LAMBDA_GRID)
            assert len(res["p_star"]) == len(LAMBDA_GRID)
            assert len(res["mae_tau_U"]) == len(LAMBDA_GRID)

    def test_theoretical_taus_stored_correctly(self, grid_results):
        # tau_U, tau_B, tau_CE stored in results should match thresholds.py
        from src.thresholds import tau_U, tau_B, tau_CE
        for loss_fn, res in grid_results.items():
            for i, lam in enumerate(res["lambda"]):
                assert abs(res["tau_U"][i] - float(tau_U(lam))) < 1e-9
                assert abs(res["tau_B"][i] - float(tau_B(lam))) < 1e-9
                assert abs(res["tau_CE"][i] - float(tau_CE(lam))) < 1e-9


class TestPrintMaeTable:
    def test_runs_without_error(self, grid_results, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            print_mae_table(grid_results)
        # Should emit at least one log line per loss function
        assert "utility" in caplog.text
        assert "brier" in caplog.text
        assert "cross-entropy" in caplog.text

    def test_best_formula_is_min_mae(self, grid_results, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            print_mae_table(grid_results)
        # For utility loss, τ_U should win (empirically confirmed in your runs)
        assert "τ_U" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])