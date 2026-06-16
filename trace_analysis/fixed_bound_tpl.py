"""Fixed-bound truncated power-law utilities for fracture radius and diameter in meters.

This module assumes fracture radius follows a fixed-bound truncated power-law:

    R ~ TPL(alpha, r_min, r_max)

where ``alpha`` is the PDF exponent, not the CCDF exponent, and

    p_R(r | alpha) ∝ r^(-alpha)

All distances in this module are in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import numpy as np

DEFAULT_R_MIN_M = 1.0
DEFAULT_R_MAX_M = 250.0
DEFAULT_D_MIN_M = 2.0
DEFAULT_D_MAX_M = 500.0
DEFAULT_ALPHA_BOUNDS = (1.01, 6.0)

_ALPHA_ONE_TOL = 1e-8
_ALPHA_TWO_TOL = 1e-8


def _validate_bounds(lower_m: float, upper_m: float, name: str) -> None:
    if not np.isfinite(lower_m) or not np.isfinite(upper_m):
        raise ValueError(f"{name} bounds must be finite.")
    if lower_m <= 0.0:
        raise ValueError(f"{name} lower bound must be positive.")
    if upper_m <= lower_m:
        raise ValueError(f"{name} upper bound must be greater than lower bound.")


def _to_array(x: float | Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _radius_norm_const(alpha: float, r_min: float, r_max: float) -> float:
    if abs(alpha - 1.0) <= _ALPHA_ONE_TOL:
        return 1.0 / np.log(r_max / r_min)
    exponent = 1.0 - alpha
    denominator = r_max**exponent - r_min**exponent
    return exponent / denominator


def radius_pdf(
    r: float | Iterable[float] | np.ndarray,
    alpha: float,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> np.ndarray | float:
    """Return the fixed-bound truncated power-law radius PDF in meters.

    ``alpha`` is the PDF exponent, not the CCDF exponent.
    """
    _validate_bounds(r_min, r_max, "Radius")
    r_arr = _to_array(r)
    pdf = np.zeros_like(r_arr, dtype=float)
    mask = (r_arr >= r_min) & (r_arr <= r_max)
    if np.any(mask):
        norm_const = _radius_norm_const(alpha, r_min, r_max)
        pdf[mask] = norm_const * np.power(r_arr[mask], -alpha)
    return float(pdf) if np.isscalar(r) else pdf


def radius_cdf(
    r: float | Iterable[float] | np.ndarray,
    alpha: float,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> np.ndarray | float:
    """Return the fixed-bound truncated power-law radius CDF in meters."""
    _validate_bounds(r_min, r_max, "Radius")
    r_arr = _to_array(r)
    cdf = np.zeros_like(r_arr, dtype=float)
    cdf[r_arr >= r_max] = 1.0
    interior = (r_arr > r_min) & (r_arr < r_max)
    if np.any(interior):
        if abs(alpha - 1.0) <= _ALPHA_ONE_TOL:
            cdf[interior] = np.log(r_arr[interior] / r_min) / np.log(r_max / r_min)
        else:
            exponent = 1.0 - alpha
            denominator = r_max**exponent - r_min**exponent
            cdf[interior] = (r_arr[interior] ** exponent - r_min**exponent) / denominator
    return float(cdf) if np.isscalar(r) else cdf


def radius_survival(
    r: float | Iterable[float] | np.ndarray,
    alpha: float,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> np.ndarray | float:
    """Return the fixed-bound truncated power-law radius survival function in meters."""
    cdf = radius_cdf(r, alpha, r_min=r_min, r_max=r_max)
    survival = 1.0 - cdf
    if np.isscalar(r):
        if float(r) <= r_min:
            return 1.0
        if float(r) >= r_max:
            return 0.0
        return float(survival)
    r_arr = _to_array(r)
    survival = np.asarray(survival, dtype=float)
    survival[r_arr <= r_min] = 1.0
    survival[r_arr >= r_max] = 0.0
    return survival


def radius_ppf(
    p: float | Iterable[float] | np.ndarray,
    alpha: float,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> np.ndarray | float:
    """Return the fixed-bound truncated power-law radius quantile function in meters."""
    _validate_bounds(r_min, r_max, "Radius")
    p_arr = _to_array(p)
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError("Probability input to radius_ppf must lie in [0, 1].")
    q = np.empty_like(p_arr, dtype=float)
    q[p_arr <= 0.0] = r_min
    q[p_arr >= 1.0] = r_max
    interior = (p_arr > 0.0) & (p_arr < 1.0)
    if np.any(interior):
        if abs(alpha - 1.0) <= _ALPHA_ONE_TOL:
            q[interior] = r_min * np.exp(p_arr[interior] * np.log(r_max / r_min))
        else:
            exponent = 1.0 - alpha
            q[interior] = (
                p_arr[interior] * (r_max**exponent - r_min**exponent) + r_min**exponent
            ) ** (1.0 / exponent)
    return float(q) if np.isscalar(p) else q


def expected_radius(
    alpha: float,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> float:
    """Return E[R] in meters under the fixed-bound truncated power-law."""
    _validate_bounds(r_min, r_max, "Radius")
    if abs(alpha - 2.0) <= _ALPHA_TWO_TOL:
        norm_const = _radius_norm_const(alpha, r_min, r_max)
        return float(norm_const * np.log(r_max / r_min))
    norm_const = _radius_norm_const(alpha, r_min, r_max)
    exponent = 2.0 - alpha
    return float(norm_const * (r_max**exponent - r_min**exponent) / exponent)


def diameter_pdf(
    diameter_m: float | Iterable[float] | np.ndarray,
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> np.ndarray | float:
    """Return the diameter PDF in meters where D = 2R and alpha is the PDF exponent."""
    _validate_bounds(d_min, d_max, "Diameter")
    d_arr = _to_array(diameter_m)
    pdf = 0.5 * radius_pdf(0.5 * d_arr, alpha, r_min=0.5 * d_min, r_max=0.5 * d_max)
    if np.isscalar(diameter_m):
        return float(pdf)
    pdf = np.asarray(pdf, dtype=float)
    pdf[(d_arr < d_min) | (d_arr > d_max)] = 0.0
    return pdf


def diameter_cdf(
    diameter_m: float | Iterable[float] | np.ndarray,
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> np.ndarray | float:
    """Return the diameter CDF in meters where D = 2R."""
    _validate_bounds(d_min, d_max, "Diameter")
    d_arr = _to_array(diameter_m)
    cdf = radius_cdf(0.5 * d_arr, alpha, r_min=0.5 * d_min, r_max=0.5 * d_max)
    if np.isscalar(diameter_m):
        return float(cdf)
    cdf = np.asarray(cdf, dtype=float)
    cdf[d_arr <= d_min] = 0.0
    cdf[d_arr >= d_max] = 1.0
    return cdf


def diameter_survival(
    diameter_m: float | Iterable[float] | np.ndarray,
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> np.ndarray | float:
    """Return the diameter survival function in meters where D = 2R."""
    cdf = diameter_cdf(diameter_m, alpha, d_min=d_min, d_max=d_max)
    survival = 1.0 - cdf
    if np.isscalar(diameter_m):
        if float(diameter_m) <= d_min:
            return 1.0
        if float(diameter_m) >= d_max:
            return 0.0
        return float(survival)
    d_arr = _to_array(diameter_m)
    survival = np.asarray(survival, dtype=float)
    survival[d_arr <= d_min] = 1.0
    survival[d_arr >= d_max] = 0.0
    return survival


def diameter_bin_probability(
    d_left: float,
    d_right: float,
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> float:
    """Return P(D in [d_left, d_right]) clipped to the support [d_min, d_max], in meters."""
    _validate_bounds(d_min, d_max, "Diameter")
    left = max(float(d_left), d_min)
    right = min(float(d_right), d_max)
    if right <= left:
        return 0.0
    cdf_right = float(diameter_cdf(right, alpha, d_min=d_min, d_max=d_max))
    cdf_left = float(diameter_cdf(left, alpha, d_min=d_min, d_max=d_max))
    return max(0.0, min(1.0, cdf_right - cdf_left))


def expected_diameter(
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> float:
    """Return E[D] in meters where D = 2R."""
    return 2.0 * expected_radius(alpha, r_min=0.5 * d_min, r_max=0.5 * d_max)


@dataclass
class JointSetTPLResult:
    """Result object for per-joint-set fixed-bound TPL estimation."""

    joint_set: Any
    alpha_pdf_exponent: float
    alpha_ci: tuple[float, float] | None
    rho: float | None
    r_min_m: float
    r_max_m: float
    d_min_m: float
    d_max_m: float
    loglik: float
    n_traces: int
    detection_limit_m: float | None
    R50_m: float
    R80_m: float
    R90_m: float
    R95_m: float
    expected_R_m: float
    profile_loglik: Dict[str, list[float]]
    warnings: list[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """Convert the result to a flat record for DataFrame/JSON export."""
        return {
            "joint_set": self.joint_set,
            "r_min_m": self.r_min_m,
            "r_max_m": self.r_max_m,
            "d_min_m": self.d_min_m,
            "d_max_m": self.d_max_m,
            "alpha_pdf_exponent": self.alpha_pdf_exponent,
            "rho": self.rho,
            "n_traces": self.n_traces,
            "detection_limit_m": self.detection_limit_m,
            "R50_m": self.R50_m,
            "R80_m": self.R80_m,
            "R90_m": self.R90_m,
            "R95_m": self.R95_m,
            "expected_R_m": self.expected_R_m,
            "loglik": self.loglik,
            "warnings": "; ".join(self.warnings),
        }
