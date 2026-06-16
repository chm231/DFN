import numpy as np
import pytest

from trace_analysis.fixed_bound_tpl import diameter_bin_probability
from trace_analysis.jcv_likelihood import fit_alpha_jcv_poisson


def test_fit_alpha_jcv_poisson_recovers_synthetic_parameters():
    rng = np.random.default_rng(123)
    alpha_true = 2.6
    rho_true = 3.5
    diameter_bins = np.array(
        [[2.0, 4.0], [4.0, 8.0], [8.0, 16.0], [16.0, 32.0], [32.0, 64.0], [64.0, 128.0], [128.0, 256.0], [256.0, 500.0]]
    )
    q = np.array([diameter_bin_probability(left, right, alpha_true) for left, right in diameter_bins])
    # Use a higher-exposure synthetic tensor so the 1D profile likelihood is identifiable.
    jcv_tensor = rng.uniform(0.5, 2.0, size=(2, 2, 2, len(diameter_bins))) * 100.0
    mu = rho_true * np.tensordot(jcv_tensor, q, axes=([3], [0]))
    counts = rng.poisson(mu)
    result = fit_alpha_jcv_poisson(counts, jcv_tensor, diameter_bins)
    assert result.alpha_pdf_exponent == pytest.approx(alpha_true, abs=0.5)
    assert result.rho == pytest.approx(rho_true, abs=0.5)


def test_fit_alpha_jcv_poisson_validation_errors():
    bins = np.array([[2.0, 4.0], [4.0, 8.0]])
    counts = np.ones((1, 1, 1))
    tensor = np.ones((1, 1, 1, 3))
    with pytest.raises(ValueError):
        fit_alpha_jcv_poisson(counts, tensor, bins)
    with pytest.raises(ValueError):
        fit_alpha_jcv_poisson(np.array([[[-1.0]]]), np.ones((1, 1, 1, 2)), bins)
    with pytest.raises(ValueError):
        fit_alpha_jcv_poisson(np.ones((1, 1, 1)), -np.ones((1, 1, 1, 2)), bins)
