"""Ideal infinite-plane trace likelihood for fixed-bound truncated power-law radius models.

This module assumes fracture radius follows:

    R ~ TPL(alpha, 1 m, 250 m)

where ``alpha`` is the PDF exponent, not the CCDF exponent. Observed trace length is
the chord length produced by the intersection between a 3D circular fracture disc and
the observation plane. It is not used directly as a radius sample.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from trace_analysis.fixed_bound_tpl import (
    DEFAULT_ALPHA_BOUNDS,
    DEFAULT_D_MAX_M,
    DEFAULT_D_MIN_M,
    DEFAULT_R_MAX_M,
    DEFAULT_R_MIN_M,
    JointSetTPLResult,
    diameter_pdf,
    expected_diameter,
    expected_radius,
    radius_ppf,
)

_LIK_EPS = 1e-300
_QUAD_EPSABS = 1e-8
_QUAD_EPSREL = 1e-6


def _as_float_array(values: float | Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _theta_limits(length_m: float, d_min: float, d_max: float) -> tuple[float, float] | None:
    if length_m <= 0.0 or length_m >= d_max:
        return None
    lower_d = max(length_m, d_min)
    theta_lo = float(np.arccos(np.clip(length_m / lower_d, 0.0, 1.0)))
    theta_hi = float(np.arccos(np.clip(length_m / d_max, 0.0, 1.0)))
    if theta_hi <= theta_lo:
        return None
    return theta_lo, theta_hi


def _trace_pdf_ideal_scalar(length_m: float, alpha: float, d_min: float, d_max: float) -> float:
    if length_m <= 0.0 or length_m >= d_max:
        return 0.0
    limits = _theta_limits(length_m, d_min=d_min, d_max=d_max)
    if limits is None:
        return 0.0
    theta_lo, theta_hi = limits
    e_d = expected_diameter(alpha, d_min=d_min, d_max=d_max)

    def integrand(theta: float) -> float:
        diameter_m = length_m / np.cos(theta)
        return diameter_pdf(diameter_m, alpha, d_min=d_min, d_max=d_max) / np.cos(theta)

    value, _ = quad(integrand, theta_lo, theta_hi, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL, limit=200)
    return float(max(0.0, length_m * value / e_d))


def _trace_survival_ideal_scalar(length_m: float, alpha: float, d_min: float, d_max: float) -> float:
    if length_m <= 0.0:
        return 1.0
    if length_m >= d_max:
        return 0.0
    lower_d = max(length_m, d_min)
    e_d = expected_diameter(alpha, d_min=d_min, d_max=d_max)

    def integrand(diameter_m: float) -> float:
        return diameter_pdf(diameter_m, alpha, d_min=d_min, d_max=d_max) * np.sqrt(
            max(diameter_m**2 - length_m**2, 0.0)
        )

    value, _ = quad(integrand, lower_d, d_max, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL, limit=200)
    return float(np.clip(value / e_d, 0.0, 1.0))


def trace_pdf_ideal(
    length_m: float | Iterable[float] | np.ndarray,
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> np.ndarray | float:
    """Return the ideal infinite-plane trace PDF for chord length in meters."""
    lengths = _as_float_array(length_m)
    values = np.array([
        _trace_pdf_ideal_scalar(float(val), alpha=alpha, d_min=d_min, d_max=d_max) for val in lengths.flat
    ]).reshape(lengths.shape)
    return float(values) if np.isscalar(length_m) else values


def trace_survival_ideal(
    length_m: float | Iterable[float] | np.ndarray,
    alpha: float,
    d_min: float = DEFAULT_D_MIN_M,
    d_max: float = DEFAULT_D_MAX_M,
) -> np.ndarray | float:
    """Return the ideal infinite-plane trace survival function for chord length in meters."""
    lengths = _as_float_array(length_m)
    values = np.array([
        _trace_survival_ideal_scalar(float(val), alpha=alpha, d_min=d_min, d_max=d_max) for val in lengths.flat
    ]).reshape(lengths.shape)
    return float(values) if np.isscalar(length_m) else values


def _normalize_censor_value(value: Any) -> str:
    val = str(value).strip().lower()
    valid = {"complete", "one_end", "two_end", "censored"}
    if val not in valid:
        raise ValueError(
            f"Unsupported censor class '{value}'. Expected one of: complete, one_end, two_end, censored."
        )
    return val


def _prepare_trace_table(
    traces: pd.DataFrame,
    joint_set_col: str,
    length_col: str,
    censor_col: str | None,
    d_max_m: float,
) -> pd.DataFrame:
    if not isinstance(traces, pd.DataFrame):
        traces = pd.DataFrame(traces)
    required = [joint_set_col, length_col]
    missing = [col for col in required if col not in traces.columns]
    if missing:
        raise ValueError(f"Trace table is missing required columns: {', '.join(missing)}")
    work = traces.copy()
    work[length_col] = pd.to_numeric(work[length_col], errors="coerce")
    if work[length_col].isna().any():
        raise ValueError(f"Column '{length_col}' contains non-numeric values.")
    if (work[length_col] <= 0.0).any():
        raise ValueError("Trace length must be strictly positive.")
    if (work[length_col] > d_max_m).any():
        bad_values = work.loc[work[length_col] > d_max_m, length_col].tolist()
        raise ValueError(
            f"Trace length exceeds D_max={d_max_m} m. Example values: {bad_values[:5]}"
        )
    if censor_col is None:
        work["_censor_state"] = "complete"
    else:
        if censor_col not in work.columns:
            raise ValueError(f"Trace table is missing censor column '{censor_col}'.")
        work["_censor_state"] = work[censor_col].map(_normalize_censor_value)
    return work


def _compute_profile_loglik_for_group(
    lengths_m: np.ndarray,
    censor_states: np.ndarray,
    detection_limit_m: float,
    alpha: float,
    d_min_m: float,
    d_max_m: float,
) -> tuple[float, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "pdf_floor_count": 0,
        "survival_floor_count": 0,
        "detection_survival_floor_count": 0,
    }
    survival_det = _trace_survival_ideal_scalar(detection_limit_m, alpha, d_min_m, d_max_m)
    if survival_det <= 0.0:
        diagnostics["detection_survival_floor_count"] += 1
    log_survival_det = np.log(max(survival_det, _LIK_EPS))
    total = 0.0
    for length_m, censor_state in zip(lengths_m, censor_states):
        if censor_state == "complete":
            pdf_value = _trace_pdf_ideal_scalar(length_m, alpha, d_min_m, d_max_m)
            if pdf_value <= 0.0:
                diagnostics["pdf_floor_count"] += 1
            total += np.log(max(pdf_value, _LIK_EPS)) - log_survival_det
        else:
            survival_value = _trace_survival_ideal_scalar(length_m, alpha, d_min_m, d_max_m)
            if survival_value <= 0.0:
                diagnostics["survival_floor_count"] += 1
            total += np.log(max(survival_value, _LIK_EPS)) - log_survival_det
    return float(total), diagnostics


def _estimate_alpha_ci(alpha_grid: np.ndarray, loglik_grid: np.ndarray) -> tuple[float, float] | None:
    if len(alpha_grid) == 0:
        return None
    max_loglik = float(np.max(loglik_grid))
    threshold = max_loglik - 0.5 * chi2.ppf(0.95, df=1)
    support = alpha_grid[loglik_grid >= threshold]
    if len(support) == 0:
        return None
    return float(np.min(support)), float(np.max(support))


def fit_alpha_ideal(
    traces: pd.DataFrame,
    joint_set_col: str = "joint_set",
    length_col: str = "trace_length",
    censor_col: str | None = None,
    detection_limit: float | None = None,
    alpha_bounds: tuple[float, float] = DEFAULT_ALPHA_BOUNDS,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> Dict[Any, JointSetTPLResult]:
    """Fit per-joint-set alpha using the ideal infinite-plane trace likelihood.

    ``alpha`` is the PDF exponent. Detection limit is the trace detection threshold in meters
    and is distinct from ``r_min``.
    """
    if detection_limit is None:
        raise ValueError("detection_limit must be provided explicitly for fit_alpha_ideal.")
    if detection_limit <= 0.0:
        raise ValueError("detection_limit must be strictly positive.")
    d_min_m = 2.0 * r_min
    d_max_m = 2.0 * r_max
    if detection_limit >= d_max_m:
        raise ValueError(f"detection_limit must be smaller than D_max={d_max_m} m.")
    alpha_lo, alpha_hi = alpha_bounds
    if alpha_lo <= 1.0 or alpha_hi <= alpha_lo:
        raise ValueError("alpha_bounds must satisfy 1.0 < lower < upper.")

    work = _prepare_trace_table(
        traces=traces,
        joint_set_col=joint_set_col,
        length_col=length_col,
        censor_col=censor_col,
        d_max_m=d_max_m,
    )

    results: Dict[Any, JointSetTPLResult] = {}
    for joint_set, group in work.groupby(joint_set_col, sort=True):
        lengths_m = group[length_col].to_numpy(dtype=float)
        censor_states = group["_censor_state"].to_numpy(dtype=str)
        warnings: list[str] = []
        if len(lengths_m) < 5:
            warnings.append("Trace sample size is very small; alpha estimate may be unstable.")

        alpha_grid = np.linspace(alpha_lo, alpha_hi, 64)
        loglik_grid = []
        profile_diagnostics = []
        for alpha in alpha_grid:
            loglik, diag = _compute_profile_loglik_for_group(
                lengths_m=lengths_m,
                censor_states=censor_states,
                detection_limit_m=detection_limit,
                alpha=alpha,
                d_min_m=d_min_m,
                d_max_m=d_max_m,
            )
            loglik_grid.append(loglik)
            profile_diagnostics.append(diag)
        loglik_grid_arr = np.asarray(loglik_grid, dtype=float)
        best_grid_idx = int(np.argmax(loglik_grid_arr))
        best_grid_alpha = float(alpha_grid[best_grid_idx])

        def objective(alpha: float) -> float:
            loglik, _ = _compute_profile_loglik_for_group(
                lengths_m=lengths_m,
                censor_states=censor_states,
                detection_limit_m=detection_limit,
                alpha=alpha,
                d_min_m=d_min_m,
                d_max_m=d_max_m,
            )
            return -loglik

        opt = minimize_scalar(objective, bounds=alpha_bounds, method="bounded", options={"xatol": 1e-3})
        if not opt.success:
            warnings.append(f"Alpha optimization did not fully converge: {opt.message}")
            alpha_hat = best_grid_alpha
            loglik_hat, diagnostics = _compute_profile_loglik_for_group(
                lengths_m=lengths_m,
                censor_states=censor_states,
                detection_limit_m=detection_limit,
                alpha=alpha_hat,
                d_min_m=d_min_m,
                d_max_m=d_max_m,
            )
        else:
            alpha_hat = float(opt.x)
            loglik_hat, diagnostics = _compute_profile_loglik_for_group(
                lengths_m=lengths_m,
                censor_states=censor_states,
                detection_limit_m=detection_limit,
                alpha=alpha_hat,
                d_min_m=d_min_m,
                d_max_m=d_max_m,
            )

        ci = _estimate_alpha_ci(alpha_grid, loglik_grid_arr)
        diagnostics.update(
            {
                "coarse_grid_alpha_best": best_grid_alpha,
                "coarse_grid_loglik_best": float(loglik_grid_arr[best_grid_idx]),
                "profile_grid_warnings": profile_diagnostics[best_grid_idx],
                "trace_pdf_model": "ideal_infinite_plane",
                "alpha_parameterization": "PDF exponent",
            }
        )
        result = JointSetTPLResult(
            joint_set=joint_set,
            alpha_pdf_exponent=float(alpha_hat),
            alpha_ci=ci,
            rho=None,
            r_min_m=float(r_min),
            r_max_m=float(r_max),
            d_min_m=float(d_min_m),
            d_max_m=float(d_max_m),
            loglik=float(loglik_hat),
            n_traces=int(len(lengths_m)),
            detection_limit_m=float(detection_limit),
            R50_m=float(radius_ppf(0.50, alpha_hat, r_min=r_min, r_max=r_max)),
            R80_m=float(radius_ppf(0.80, alpha_hat, r_min=r_min, r_max=r_max)),
            R90_m=float(radius_ppf(0.90, alpha_hat, r_min=r_min, r_max=r_max)),
            R95_m=float(radius_ppf(0.95, alpha_hat, r_min=r_min, r_max=r_max)),
            expected_R_m=float(expected_radius(alpha_hat, r_min=r_min, r_max=r_max)),
            profile_loglik={
                "alpha_pdf_exponent": alpha_grid.tolist(),
                "loglik": loglik_grid_arr.tolist(),
            },
            warnings=warnings,
            diagnostics=diagnostics,
        )
        results[joint_set] = result
    return results
