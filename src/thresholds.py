# Closed-form optimal abstention thresholds under three loss functions.
# Implements tau_U and tau_B (TODO: tau_CE).
#
# All functions accept scalar or numpy array input and return the same type.

import numpy as np
import scipy.optimize as opt


def tau_U(lam):
    """
    Optimal abstention threshold under asymmetric utility loss.
    Derived via expected utility maximization under calibration P(correct|c)=c
    Scoring rule: +1 correct, -lambda wrong, 0 abstain
    """
    scalar = np.isscalar(lam)
    lam = np.atleast_1d(np.asarray(lam, dtype=float))

    if np.any(lam < 0):
        raise ValueError(f"lambda must be >= 0, got {lam[lam < 0]}")

    out = lam / (1.0 + lam)

    return float(out[0]) if scalar else out


def tau_B(lam):
    """
    Optimal abstention threshold under Brier score loss.
    Derived from: answer iff E[Brier | answer, c] = c*(1-c) <= lambda
    Quadratic c^2 - c + lambda = 0, upper root taken (high-confidence region).
    Valid domain: lambda in [0, 0.25]. Above 0.25: always answer (threshold = 0.0).
    Numerical: discriminant as 4*(0.25 - lambda) avoids cancellation near boundary.
    """
    scalar = np.isscalar(lam)
    lam = np.atleast_1d(np.asarray(lam, dtype=float))

    if np.any(lam < 0):
        raise ValueError(f"lambda must be >= 0, got {lam[lam < 0]}")

    out = np.zeros_like(lam) #Default 0.0 (always answer) used when lambda > 0.25

    mask = lam <= 0.25

    if np.any(mask):
        discriminant = 4.0 * (0.25 - lam[mask]) # avoids cancellation when lambda is close to 0.25
        discriminant = np.maximum(discriminant, 0.0)  # guard exact boundary
        out[mask] = (1.0 + np.sqrt(discriminant)) / 2.0

    return float(out[0]) if scalar else out


# TODO: tau_CE