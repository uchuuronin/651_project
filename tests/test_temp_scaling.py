import numpy as np

from temp_scaling import fit_temperature, temperature_scale


def test_temperature_scale_identity_at_one():
    confidences = np.array([0.1, 0.25, 0.7, 0.9])
    scaled = temperature_scale(confidences, temperature=1.0)
    assert np.allclose(scaled, confidences, atol=1e-9)


def test_fit_temperature_returns_positive_value_for_calibrated_data():
    confidences = np.array([0.1, 0.9, 0.2, 0.8])
    correct = np.array([0.0, 1.0, 0.0, 1.0])
    temperature, nll = fit_temperature(confidences, correct)
    assert temperature > 0.0
    assert nll >= 0.0
    nll_at_one = -np.mean(correct * np.log(confidences + 1e-9) + (1.0 - correct) * np.log(1.0 - confidences + 1e-9))
    assert nll <= nll_at_one + 1e-9
