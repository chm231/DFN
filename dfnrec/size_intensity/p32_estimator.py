"""P32 and size distribution inversion from trace observations.

Estimation pipeline
-------------------
1. Collect traces for a set_id from multiple faces.
2. Estimate (alpha, r_min) via MLE on chord-length distribution.
3. Estimate P32 from the observed trace density, corrected for:
   - Orientation factor C_s = E[|n · m_face|]
   - Area fraction factor F_A (fraction of face area with active traces)
4. Compute derived quantities: P30, n0, P21, N_expected.

P32 definition
--------------
P32 [m²/m³] = total fracture area per unit volume.
P32_eff = P32 restricted to r ≥ r_min.
P30 [m⁻³] = number of disc centres per unit volume.
P21 [m/m²] = total trace length per unit observation area.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from dfnrec.models import (
    Trace,
    Face,
    ReconstructedDisc,
    FractureSetOrientation,
    FractureSetSizeIntensity,
    SizeModel,
)
from dfnrec.geometry.vector import normalize
from dfnrec.size_intensity.chord_likelihood import censored_chord_log_likelihood


def _mean_r2_power_law(alpha: float, r_min: float, r_max: float) -> float:
    """E[r^2] under truncated power-law f(r) ∝ r^{-alpha}, r in [r_min, r_max]."""
    if alpha == 3.0:
        Z = math.log(r_max / r_min)
        return math.log(r_max / r_min) / (1.0 / r_min - 1.0 / r_max) if Z > 1e-9 else r_min**2
    exp = 3.0 - alpha
    numerator = (r_max**exp - r_min**exp) / exp
    if abs(1.0 - alpha) < 1e-9:
        denominator = math.log(r_max / r_min)
    else:
        denominator = (r_max**(1 - alpha) - r_min**(1 - alpha)) / (1 - alpha)
    if abs(denominator) < 1e-15:
        return r_min**2
    return numerator / denominator


def _mean_r2_exponential(lmb: float, r_min: float) -> float:
    """E[r^2] under exponential f(r) = lambda * exp(-lambda * (r - r_min)) for r >= r_min."""
    return r_min**2 + 2.0 * r_min / lmb + 2.0 / (lmb**2)


def _orientation_factor(
    normal_xyz: np.ndarray,
    faces: List[Face],
) -> float:
    """C_s = mean ||n × m_face|| averaged over all faces.

    This is the sampling area correction for a disc of normal n:
    the expected trace length per unit area on a face with normal m is
    proportional to ||n × m||.
    """
    if not faces:
        return 1.0
    vals = []
    n = normalize(np.asarray(normal_xyz, dtype=float))
    for face in faces:
        m = normalize(np.asarray(face.normal_xyz, dtype=float))
        # ||n x m|| = sin(theta)
        cross = np.cross(n, m)
        vals.append(float(np.linalg.norm(cross)))
    return float(np.mean(vals))


class SizeEstimateResult(tuple):
    """Subclass of tuple to hold size MLE parameters with backwards compatibility."""
    def __new__(cls, alpha_or_lambda: float, r_min: float, size_model: str = "POWER_LAW"):
        return super().__new__(cls, (alpha_or_lambda, r_min))
    def __init__(self, alpha_or_lambda: float, r_min: float, size_model: str = "POWER_LAW"):
        self.alpha_or_lambda = alpha_or_lambda
        self.r_min_val = r_min
        self.size_model = size_model


def estimate_size_model(
    traces: List[Trace],
    set_id: str,
    r_min: float = 0.5,
    r_max: float = 30.0,
    L_min: float = 0.1,
) -> SizeEstimateResult:
    """Estimate best size model (POWER_LAW or EXPONENTIAL) and its parameters via joint MLE on chord lengths.

    Parameters
    ----------
    traces : list of Trace
    set_id : str
    r_min, r_max : float
    L_min : float

    Returns
    -------
    SizeEstimateResult (unpacks as (alpha_or_lambda, r_min_used))
    """
    set_traces = [t for t in traces if t.set_id == set_id]
    if not set_traces:
        return SizeEstimateResult(3.5, r_min, "POWER_LAW")

    chord_lengths = np.array([t.observed_length for t in set_traces])
    is_contained = np.array([t.is_contained for t in set_traces])

    # 1. Fit POWER_LAW
    best_pl_ll = -1e10
    best_pl_alpha = 3.5
    best_pl_rmin = r_min

    # Search r_min in range [0.1, 1.5] and alpha in [1.5, 6.0]
    r_min_grid = np.linspace(0.1, 1.5, 29)
    alpha_grid = np.linspace(1.5, 6.0, 46)

    for r_cand in r_min_grid:
        for alpha_cand in alpha_grid:
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, alpha_cand, r_cand, r_max, L_min, size_model="POWER_LAW"
            )
            if ll > best_pl_ll:
                best_pl_ll = ll
                best_pl_alpha = alpha_cand
                best_pl_rmin = r_cand

    # 2. Fit EXPONENTIAL
    best_exp_ll = -1e10
    best_exp_lambda = 0.25
    best_exp_rmin = r_min

    # Search lambda in range [0.05, 1.0]
    lambda_grid = np.linspace(0.05, 1.0, 39)

    for r_cand in r_min_grid:
        for lambda_cand in lambda_grid:
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, lambda_cand, r_cand, r_max, L_min, size_model="EXPONENTIAL"
            )
            if ll > best_exp_ll:
                best_exp_ll = ll
                best_exp_lambda = lambda_cand
                best_exp_rmin = r_cand

    if best_pl_ll >= best_exp_ll:
        return SizeEstimateResult(best_pl_alpha, best_pl_rmin, "POWER_LAW")
    else:
        return SizeEstimateResult(best_exp_lambda, best_exp_rmin, "EXPONENTIAL")


def _area_integral_power_law(r: float, alpha: float) -> float:
    if abs(3.0 - alpha) < 1e-9:
        return math.log(r)
    return (r ** (3.0 - alpha)) / (3.0 - alpha)


def _area_integral_exponential(r: float, lmb: float) -> float:
    return -math.exp(-lmb * r) * (r**2 + 2.0 * r / lmb + 2.0 / (lmb**2))


def estimate_p32(
    traces: List[Trace],
    faces: List[Face],
    orientation_result: Optional[FractureSetOrientation],
    alpha: float,
    r_min: float,
    r_max: float = 30.0,
    L_min: float = 0.1,
    discs: Optional[List[ReconstructedDisc]] = None,
    size_model: str = "POWER_LAW",
) -> FractureSetSizeIntensity:
    """Estimate P32 and intensity parameters for a fracture set.

    Parameters
    ----------
    traces : list of Trace for this set
    faces : list of Face
    orientation_result : FractureSetOrientation or None
    alpha : float
        Power-law PDF exponent (k_r = alpha - 1) or Exponential rate (lambda).
    r_min : float
        Estimated r_min_mle.
    r_max : float
    L_min : float
    discs : list of ReconstructedDisc or None
    size_model : str
        POWER_LAW or EXPONENTIAL.
    """
    sid = orientation_result.set_id if orientation_result else "unknown"

    # Observed P21 = total trace length / total face area
    total_trace_length = sum(t.observed_length for t in traces)
    total_face_area = sum(f.window_area() for f in faces)
    P21_obs = total_trace_length / max(total_face_area, 1e-9)
    P20_obs = len(traces) / max(total_face_area, 1e-9)

    # Orientation factor C_s = ||n x m_face|| (sin theta)
    if orientation_result is not None:
        from dfnrec.geometry.vector import normal_from_trend_plunge
        mean_normal = normal_from_trend_plunge(
            orientation_result.mean_trend_deg,
            orientation_result.mean_plunge_deg,
        )
        C_s = _orientation_factor(mean_normal, faces)
    else:
        C_s = 0.5  # isotropic default

    C_s_safe = max(C_s, 0.01)

    # Compute mean area E[pi r^2]
    if size_model == "POWER_LAW":
        mean_r2 = _mean_r2_power_law(alpha, r_min, r_max)
    else:
        mean_r2 = _mean_r2_exponential(alpha, r_min)
    mean_area = math.pi * mean_r2

    # P32_eff is estimated via exact stereology relation: P32 = P21 / Cs
    P32_eff = P21_obs / C_s_safe

    # Correct/Scale up P32_total using the area integral ratio from r_min_mle to the target r_min
    # Note: S4 exponential target range starts from 0.0 in the DFN generator config
    r_target_min = 0.0 if size_model == "EXPONENTIAL" else 0.5
    
    if size_model == "POWER_LAW":
        num_area = _area_integral_power_law(r_max, alpha) - _area_integral_power_law(r_min, alpha)
        den_area = _area_integral_power_law(r_max, alpha) - _area_integral_power_law(r_target_min, alpha)
    else:
        num_area = _area_integral_exponential(r_max, alpha) - _area_integral_exponential(r_min, alpha)
        den_area = _area_integral_exponential(r_max, alpha) - _area_integral_exponential(r_target_min, alpha)
        
    F_area = num_area / max(den_area, 1e-9)
    # P32_total is the scaled total area density in the target range
    P32_total = P32_eff / max(F_area, 0.01)

    # Number density n0 = P32_total / mean_area
    n0 = P32_total / max(mean_area, 1e-9)

    # Simulated trace length density P21_sim = P32_eff * Cs
    P21_sim = P32_eff * C_s_safe

    return FractureSetSizeIntensity(
        set_id=sid,
        size_model=SizeModel.POWER_LAW if size_model == "POWER_LAW" else SizeModel.EXPONENTIAL,
        k_r=alpha - 1.0 if size_model == "POWER_LAW" else alpha,
        r_min=r_min,
        r_max=r_max,
        lambda_exp=alpha if size_model == "EXPONENTIAL" else None,
        P32_total=P32_total,
        P32_eff=P32_eff,
        P30=n0,
        n0=n0,
        P21_observed=P21_obs,
        P21_simulated=P21_sim,
        P20_observed=P20_obs,
        N_traces_observed=len(traces),
        C_s=C_s,
        n_discs_used=len(discs) if discs else 0,
        metadata={
            "alpha": alpha,
            "mean_area_m2": mean_area,
            "F_area": F_area,
        },
    )
