import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from trace_analysis.fixed_bound_tpl import DEFAULT_D_MAX_M, diameter_pdf
from trace_analysis.trace_likelihood import fit_alpha_ideal, trace_pdf_ideal


def _sample_size_biased_diameters(alpha: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    d_grid = np.linspace(2.0, 500.0, 4000)
    weights = d_grid * diameter_pdf(d_grid, alpha)
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)
    return rng.choice(d_grid, size=n_samples, p=weights)


def _sample_trace_lengths(alpha: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    diameters = _sample_size_biased_diameters(alpha, n_samples, rng)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    return diameters * np.sqrt(1.0 - u**2)


def test_trace_pdf_ideal_normalizes():
    alpha = 2.4
    integral, _ = quad(lambda l: trace_pdf_ideal(l, alpha), 1e-6, DEFAULT_D_MAX_M - 1e-6, limit=200)
    assert integral == pytest.approx(1.0, abs=2e-2)


def test_fit_alpha_ideal_recovers_synthetic_alpha():
    rng = np.random.default_rng(42)
    alpha_true = 2.35
    lengths = _sample_trace_lengths(alpha_true, 250, rng)
    traces = pd.DataFrame(
        {
            "joint_set": np.ones(len(lengths), dtype=int),
            "trace_length": lengths,
            "censor_class": ["complete"] * len(lengths),
        }
    )
    result = fit_alpha_ideal(
        traces,
        joint_set_col="joint_set",
        length_col="trace_length",
        censor_col="censor_class",
        detection_limit=0.1,
    )[1]
    assert result.alpha_pdf_exponent == pytest.approx(alpha_true, abs=0.45)


def test_trace_likelihood_validation_errors():
    traces = pd.DataFrame({"joint_set": [1], "trace_length": [-1.0]})
    with pytest.raises(ValueError):
        fit_alpha_ideal(traces, detection_limit=0.1)
    traces = pd.DataFrame({"joint_set": [1], "trace_length": [501.0]})
    with pytest.raises(ValueError):
        fit_alpha_ideal(traces, detection_limit=0.1)
    traces = pd.DataFrame({"joint_set": [1], "trace_length": [1.0], "censor_class": ["bad"]})
    with pytest.raises(ValueError):
        fit_alpha_ideal(traces, censor_col="censor_class", detection_limit=0.1)
    traces = pd.DataFrame({"joint_set": [1], "trace_length": [1.0]})
    with pytest.raises(ValueError):
        fit_alpha_ideal(traces, detection_limit=None)
