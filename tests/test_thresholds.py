import pytest
import numpy as np
import sys
import os
from src.thresholds import tau_U, tau_B, tau_CE

class TestTauU:
    def test_known_value_01(self):
        # tau_U(0.1) = 0.1/1.1 = 0.0909...
        assert abs(tau_U(0.1) - 0.09090909) < 1e-6

    def test_known_value_025(self):
        # tau_U(0.25) = 0.25/1.25 = 0.2
        assert abs(tau_U(0.25) - 0.2) < 1e-6

    def test_boundary_zero(self):
        # lambda 0 == threshold 0 (always answer)
        assert tau_U(0.0) == 0.0

    def test_approaches_one_at_large_lambda(self):
        # lambda inf == threshold 1 (always abstain)
        assert tau_U(1000.0) > 0.999

    def test_scalar_returns_float(self):
        # return Python float, not array
        result = tau_U(0.1)
        assert isinstance(result, float)

    def test_array_input_shape_preserved(self):
        # array input must return same-shape array
        lam = np.array([0.05, 0.1, 0.2])
        result = tau_U(lam)
        assert result.shape == (3,)

    def test_monotone_increasing(self):
        # higher abstention cost results in higher threshold required
        lam = np.linspace(0.01, 0.9, 100)
        vals = tau_U(lam)
        assert np.all(np.diff(vals) > 0)

    def test_output_in_zero_one(self):
        lam = np.linspace(0.0, 10.0, 200)
        vals = tau_U(lam)
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            tau_U(-0.1)


class TestTauB:
    def test_known_value_005(self):
        # tau_B(0.05) = (1 + sqrt(0.8))/2 ≈ 0.9472
        expected = (1 + np.sqrt(1 - 4 * 0.05)) / 2
        assert abs(tau_B(0.05) - expected) < 1e-9

    def test_known_value_01(self):
        # tau_B(0.1) = (1 + sqrt(0.6))/2 ≈ 0.8873
        expected = (1 + np.sqrt(1 - 4 * 0.1)) / 2
        assert abs(tau_B(0.1) - expected) < 1e-9

    def test_at_boundary_025(self):
        # tau_B(0.25) = (1 + sqrt(0))/2 = 0.5 exactly
        assert abs(tau_B(0.25) - 0.5) < 1e-9

    def test_above_025_always_answer(self):
        # lambda > 0.25: c(1-c) <= 1/4 < lambda for all c should result threshold = 0.0
        assert tau_B(0.3) == 0.0
        assert tau_B(0.5) == 0.0
        assert tau_B(1.0) == 0.0

    def test_scalar_returns_float(self):
        result = tau_B(0.1)
        assert isinstance(result, float)

    def test_array_input_shape_preserved(self):
        lam = np.array([0.05, 0.1, 0.25])
        result = tau_B(lam)
        assert result.shape == (3,)

    def test_monotone_decreasing_in_valid_domain(self):
        # tau_B decreases as lambda increases toward 0.25
        lam = np.linspace(0.001, 0.249, 100)
        vals = tau_B(lam)
        assert np.all(np.diff(vals) < 0)

    def test_output_in_zero_one_valid_domain(self):
        lam = np.linspace(0.0, 0.25, 100)
        vals = tau_B(lam)
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_no_nan_at_boundary(self):
        # discriminant near zero should not produce nan
        assert not np.isnan(tau_B(0.2499999))
        assert not np.isnan(tau_B(0.25))

    def test_mixed_array_valid_and_invalid(self):
        # array with both valid (<= 0.25) and invalid (> 0.25) lambda values
        lam = np.array([0.1, 0.2, 0.3])
        result = tau_B(lam)
        assert result[0] > 0.5 #valid domain, threshold above 0.5
        assert result[1] > 0.5 #valid domain
        assert result[2] == 0.0 #above domain, always answer
        
    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            tau_B(-0.1)

class TestTauCE:
    def test_scalar_returns_float(self):
        result = tau_CE(0.1)
        assert isinstance(result, float)

    def test_array_input_shape_preserved(self):
        lam = np.array([0.05, 0.1, 0.2])
        result = tau_CE(lam)
        assert result.shape == (3,)

    def test_known_values_in_valid_domain(self):
        # Both should be in (0.5, 1) for valid lambda values
        assert 0.5 < tau_CE(0.1) < 1.0
        assert 0.5 < tau_CE(0.2) < 1.0

    def test_monotone_decreasing_in_valid_domain(self):
        lam = np.linspace(0.01, 0.69, 50)
        vals = tau_CE(lam)
        assert np.all(np.diff(vals) < 0)

    def test_above_log2_returns_half(self):
        # lambda >= log(2): always abstain, threshold = 0.5
        log2 = np.log(2)
        assert tau_CE(log2) == 0.5
        assert tau_CE(log2 + 0.1) == 0.5

    def test_output_in_valid_range(self):
        lam = np.linspace(0.0, 0.69, 50)
        vals = tau_CE(lam)
        assert np.all(vals >= 0.5) and np.all(vals <= 1.0)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            tau_CE(-0.1)

class TestThresholdOrdering:
    def test_tau_B_greater_than_tau_U_across_valid_domain(self):
        #tau_B > tau_U for all lambda in (0, 0.25)
        #tau_CE > tau_B > tau_U
        lam = np.linspace(0.01, 0.24, 200)
        assert np.all(tau_B(lam) > tau_U(lam)), ("Ordering violation: tau_B must exceed tau_U across valid domain. ")
        
    def test_tau_CE_greater_than_tau_B_across_valid_domain(self):
        # tau_CE > tau_B > tau_U across [0, 0.25]
        lam = np.linspace(0.01, 0.24, 200)
        assert np.all(tau_CE(lam) >= tau_B(lam) - 1e-4), \
            "Ordering violation: tau_CE must exceed tau_B across valid domain."
 
    def test_full_ordering(self):
        # Spot check at a few known lambda values
        for lam in [0.05, 0.1, 0.15, 0.2]:
            u, b, ce = tau_U(lam), tau_B(lam), tau_CE(lam)
            assert ce >= b >= u, f"Ordering tau_CE >= tau_B >= tau_U violated at lambda={lam}: CE={ce:.3f}, B={b:.3f}, U={u:.3f}"
        
# python -m pytest tests/test_thresholds.py -v