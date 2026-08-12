import numpy as np
from scipy.integrate import quad

from dfn_analysis.estimate_radius_powerlaw_from_traces import (
    _radius_grid,
    chord_pdf_given_radius,
    chord_survival_given_radius,
    fit_kr_radius,
    radius_powerlaw_pdf,
    search_interval_hit,
)


def _sample_intersected_radii(kr: float, rmin: float, rmax: float, size: int, rng: np.random.Generator) -> np.ndarray:
    exponent = 2.0 - kr
    u = rng.uniform(0.0, 1.0, size=size)
    if abs(exponent) < 1e-12:
        return rmin * np.exp(u * np.log(rmax / rmin))
    return (u * (rmax**exponent - rmin**exponent) + rmin**exponent) ** (1.0 / exponent)


def _sample_chords_from_intersected_radii(radii: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    offsets = rng.uniform(0.0, radii)
    return 2.0 * np.sqrt(radii * radii - offsets * offsets)


def test_finite_radius_powerlaw_pdf_normalizes() -> None:
    rmin = 1.0
    rmax = 250.0
    kr = 3.0
    radii = np.geomspace(rmin, rmax, 20000)
    integral = np.trapezoid(radius_powerlaw_pdf(radii, kr, rmin, rmax), radii)
    assert abs(integral - 1.0) < 1e-3


def test_chord_pdf_given_radius_normalizes() -> None:
    radius = 8.0
    integral, _ = quad(lambda length: chord_pdf_given_radius(length, radius), 0.0, 2.0 * radius)
    assert abs(integral - 1.0) < 1e-7


def test_chord_survival_given_radius_is_monotone_decreasing() -> None:
    radius = 5.0
    lengths = np.linspace(0.0, 2.0 * radius, 300)
    survival = chord_survival_given_radius(lengths, radius)
    assert np.all(np.diff(survival) <= 1e-12)
    assert survival[0] == 1.0
    assert survival[-1] == 0.0


def test_synthetic_kr_true_recovery() -> None:
    rng = np.random.default_rng(123)
    kr_true = 3.0
    rmin = 1.0
    rmax = 80.0
    radii = _sample_intersected_radii(kr_true, rmin, rmax, 900, rng)
    lengths = _sample_chords_from_intersected_radii(radii, rng)
    mask = lengths >= 0.5
    radius_grid = _radius_grid(rmin, rmax)
    fit = fit_kr_radius(lengths[mask], np.zeros(np.sum(mask), dtype=np.int32), rmin, rmax, 1.5, 5.5, radius_grid)
    assert fit["success"]
    assert abs(fit["kr_radius_hat"] - kr_true) < 0.5


def test_moderate_censoring_converges_without_boundary_solution() -> None:
    rng = np.random.default_rng(456)
    kr_true = 3.2
    rmin = 1.0
    rmax = 80.0
    radii = _sample_intersected_radii(kr_true, rmin, rmax, 700, rng)
    lengths = _sample_chords_from_intersected_radii(radii, rng)
    mask = lengths >= 0.5
    lengths = lengths[mask]
    censoring = np.zeros(len(lengths), dtype=np.int32)
    cutoff = np.percentile(lengths, 70)
    censored = lengths >= cutoff
    censoring[censored] = 1
    lengths[censored] *= 0.85

    radius_grid = _radius_grid(rmin, rmax)
    fit = fit_kr_radius(lengths, censoring, rmin, rmax, 1.5, 5.5, radius_grid)
    assert fit["success"]
    assert search_interval_hit(fit["kr_radius_hat"], 1.5, 5.5) == "none"
