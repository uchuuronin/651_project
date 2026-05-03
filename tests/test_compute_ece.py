import numpy as np

from compute_ece import compute_ece


def test_compute_ece_perfectly_calibrated():
    confidences = np.array([0.1] * 10 + [0.9] * 10)
    correct = np.array([0.0] * 9 + [1.0] + [1.0] * 9 + [0.0])
    ece, details = compute_ece(confidences, correct, n_bins=2)
    assert abs(ece - 0.0) < 1e-9
    assert len(details) == 2


def test_compute_ece_penalizes_miscalibration():
    confidences = np.array([0.1, 0.1, 0.9, 0.9])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    ece, details = compute_ece(confidences, correct, n_bins=2)
    assert abs(ece - 0.9) < 1e-9
    assert any(d["gap"] > 0 for d in details)
