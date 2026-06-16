import numpy as np
import pytest

from trace_analysis.fixed_bound_tpl import (
    DEFAULT_D_MAX_M,
    DEFAULT_D_MIN_M,
    DEFAULT_R_MAX_M,
    DEFAULT_R_MIN_M,
    diameter_bin_probability,
    expected_radius,
    radius_cdf,
    radius_pdf,
    radius_ppf,
    radius_survival,
)


def test_radius_cdf_boundaries_and_ppf_roundtrip():
    alpha = 2.3
    assert radius_cdf(DEFAULT_R_MIN_M, alpha) == pytest.approx(0.0)
    assert radius_cdf(DEFAULT_R_MAX_M, alpha) == pytest.approx(1.0)
    assert radius_survival(DEFAULT_R_MIN_M, alpha) == pytest.approx(1.0)
    assert radius_survival(DEFAULT_R_MAX_M, alpha) == pytest.approx(0.0)
    for r_val in [1.5, 5.0, 25.0, 100.0]:
        p = float(radius_cdf(r_val, alpha))
        recovered = float(radius_ppf(p, alpha))
        assert recovered == pytest.approx(r_val, rel=1e-6, abs=1e-6)


def test_diameter_bin_probabilities_sum_to_one_with_log_bins():
    alpha = 2.2
    bins = np.geomspace(DEFAULT_D_MIN_M, DEFAULT_D_MAX_M, 33)
    probs = [
        diameter_bin_probability(left, right, alpha)
        for left, right in zip(bins[:-1], bins[1:])
    ]
    assert np.sum(probs) == pytest.approx(1.0, abs=2e-3)
    assert diameter_bin_probability(-10.0, 1.0, alpha) == pytest.approx(0.0)
    assert diameter_bin_probability(400.0, 700.0, alpha) > 0.0


def test_alpha_near_one_and_two_are_stable():
    alpha_one = 1.0 + 1e-10
    alpha_two = 2.0 + 1e-10
    r = np.array([1.0, 2.0, 10.0, 250.0])
    pdf_vals = radius_pdf(r, alpha_one)
    cdf_vals = radius_cdf(r, alpha_one)
    mean_r = expected_radius(alpha_two)
    assert np.all(np.isfinite(pdf_vals))
    assert np.all(np.isfinite(cdf_vals))
    assert np.isfinite(mean_r)
