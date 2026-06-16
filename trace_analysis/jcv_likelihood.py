"""JCV-Poisson profile likelihood for fixed-bound truncated power-law diameter probabilities."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from trace_analysis.fixed_bound_tpl import (
    DEFAULT_ALPHA_BOUNDS,
    DEFAULT_R_MAX_M,
    DEFAULT_R_MIN_M,
    JointSetTPLResult,
    diameter_bin_probability,
    expected_radius,
    radius_ppf,
)

_MU_EPS = 1e-300


def _coerce_counts_array(observed_counts: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(observed_counts, pd.DataFrame):
        required = {"panel_index", "block_index", "context_index", "count"}
        if not required.issubset(observed_counts.columns):
            raise ValueError(
                "Long-form observed_counts table must contain panel_index, block_index, context_index, count."
            )
        shape = tuple(int(observed_counts[col].max()) + 1 for col in ["panel_index", "block_index", "context_index"])
        arr = np.zeros(shape, dtype=float)
        for row in observed_counts.itertuples(index=False):
            arr[int(row.panel_index), int(row.block_index), int(row.context_index)] = float(row.count)
        return arr
    return np.asarray(observed_counts, dtype=float)


def _coerce_jcv_tensor(jcv_tensor: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(jcv_tensor, pd.DataFrame):
        required = {"panel_index", "block_index", "context_index", "diameter_bin_index", "exposure"}
        if not required.issubset(jcv_tensor.columns):
            raise ValueError(
                "Long-form jcv_tensor table must contain panel_index, block_index, context_index, diameter_bin_index, exposure."
            )
        shape = tuple(int(jcv_tensor[col].max()) + 1 for col in ["panel_index", "block_index", "context_index", "diameter_bin_index"])
        arr = np.zeros(shape, dtype=float)
        for row in jcv_tensor.itertuples(index=False):
            arr[int(row.panel_index), int(row.block_index), int(row.context_index), int(row.diameter_bin_index)] = float(row.exposure)
        return arr
    return np.asarray(jcv_tensor, dtype=float)


def fit_alpha_jcv_poisson(
    observed_counts: np.ndarray | pd.DataFrame,
    jcv_tensor: np.ndarray | pd.DataFrame,
    diameter_bins: np.ndarray,
    alpha_bounds: tuple[float, float] = DEFAULT_ALPHA_BOUNDS,
    r_min: float = DEFAULT_R_MIN_M,
    r_max: float = DEFAULT_R_MAX_M,
) -> JointSetTPLResult:
    """Fit alpha and rho from JCV-Poisson profile likelihood.

    ``alpha`` is the PDF exponent. Diameter bins are given in meters as [left, right].
    """
    counts = _coerce_counts_array(observed_counts)
    tensor = _coerce_jcv_tensor(jcv_tensor)
    bins = np.asarray(diameter_bins, dtype=float)
    if tensor.ndim != 4:
        raise ValueError("jcv_tensor must have shape [p, b, c, j].")
    if counts.shape != tensor.shape[:3]:
        raise ValueError("observed_counts shape must match the first three dimensions of jcv_tensor.")
    if bins.ndim != 2 or bins.shape[1] != 2:
        raise ValueError("diameter_bins must have shape [J, 2].")
    if bins.shape[0] != tensor.shape[3]:
        raise ValueError("The number of diameter bins must match the last dimension of jcv_tensor.")
    if np.any(counts < 0.0):
        raise ValueError("observed_counts cannot contain negative values.")
    if np.any(tensor < 0.0):
        raise ValueError("jcv_tensor cannot contain negative values.")
    exposure = np.sum(tensor, axis=3)
    if np.any(exposure <= 0.0):
        raise ValueError("JCV tensor exposure denominator contains zero-valued entries.")

    d_min_m = 2.0 * r_min
    d_max_m = 2.0 * r_max
    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], 64)
    warnings: list[str] = []
    diagnostics: Dict[str, Any] = {
        "alpha_parameterization": "PDF exponent",
        "trace_pdf_model": "JCV-Poisson profile likelihood",
    }

    total_count = float(np.sum(counts))
    if total_count <= 0.0:
        raise ValueError("observed_counts sum must be positive.")

    def q_bins(alpha: float) -> np.ndarray:
        probs = np.array(
            [diameter_bin_probability(left, right, alpha, d_min=d_min_m, d_max=d_max_m) for left, right in bins],
            dtype=float,
        )
        return probs

    def profiled_loglik(alpha: float) -> tuple[float, float, np.ndarray, Dict[str, Any]]:
        q = q_bins(alpha)
        q_sum = float(np.sum(q))
        diag = {"q_bin_sum": q_sum}
        if not np.isclose(q_sum, 1.0, atol=1e-4):
            diag["q_bin_warning"] = "Diameter bin probabilities do not sum to 1 within tolerance."
        weighted_exposure = np.tensordot(tensor, q, axes=([3], [0]))
        denom = float(np.sum(weighted_exposure))
        if denom <= 0.0:
            raise ValueError("JCV weighted exposure denominator is zero.")
        rho_hat = total_count / denom
        mu = rho_hat * weighted_exposure
        zero_mu_with_counts = bool(np.any((mu <= 0.0) & (counts > 0.0)))
        if zero_mu_with_counts:
            diag["zero_mu_positive_count"] = True
            return -np.inf, rho_hat, q, diag
        mu_safe = np.maximum(mu, _MU_EPS)
        loglik = float(np.sum(counts * np.log(mu_safe) - mu_safe - gammaln(counts + 1.0)))
        return loglik, rho_hat, q, diag

    grid_loglik = []
    grid_rho = []
    grid_q = []
    grid_diag = []
    for alpha in alpha_grid:
        loglik, rho_hat, q, diag = profiled_loglik(float(alpha))
        grid_loglik.append(loglik)
        grid_rho.append(rho_hat)
        grid_q.append(q)
        grid_diag.append(diag)
    grid_loglik_arr = np.asarray(grid_loglik, dtype=float)
    best_idx = int(np.argmax(grid_loglik_arr))
    best_grid_alpha = float(alpha_grid[best_idx])

    def objective(alpha: float) -> float:
        loglik, _, _, _ = profiled_loglik(alpha)
        return -loglik if np.isfinite(loglik) else np.inf

    opt = minimize_scalar(objective, bounds=alpha_bounds, method="bounded", options={"xatol": 1e-3})
    if not opt.success:
        warnings.append(f"Alpha optimization did not fully converge: {opt.message}")
        alpha_hat = best_grid_alpha
    else:
        alpha_hat = float(opt.x)

    loglik_hat, rho_hat, q_hat, diag_hat = profiled_loglik(alpha_hat)
    diagnostics.update(diag_hat)
    result = JointSetTPLResult(
        joint_set="all",
        alpha_pdf_exponent=alpha_hat,
        alpha_ci=None,
        rho=float(rho_hat),
        r_min_m=float(r_min),
        r_max_m=float(r_max),
        d_min_m=float(d_min_m),
        d_max_m=float(d_max_m),
        loglik=float(loglik_hat),
        n_traces=int(total_count),
        detection_limit_m=None,
        R50_m=float(radius_ppf(0.50, alpha_hat, r_min=r_min, r_max=r_max)),
        R80_m=float(radius_ppf(0.80, alpha_hat, r_min=r_min, r_max=r_max)),
        R90_m=float(radius_ppf(0.90, alpha_hat, r_min=r_min, r_max=r_max)),
        R95_m=float(radius_ppf(0.95, alpha_hat, r_min=r_min, r_max=r_max)),
        expected_R_m=float(expected_radius(alpha_hat, r_min=r_min, r_max=r_max)),
        profile_loglik={
            "alpha_pdf_exponent": alpha_grid.tolist(),
            "loglik": grid_loglik_arr.tolist(),
            "rho": [float(val) for val in grid_rho],
            "q_diameter_bins": [vals.tolist() for vals in grid_q],
        },
        warnings=warnings,
        diagnostics=diagnostics,
    )
    result.diagnostics["q_diameter_bins"] = q_hat.tolist()
    return result
