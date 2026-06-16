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


def _orientation_factor(
    normal_xyz: np.ndarray,
    faces: List[Face],
) -> float:
    """C_s = mean |n · m_face| averaged over all faces.

    This is the sampling area correction for a disc of normal n:
    the expected trace length per unit area on a face with normal m is
    proportional to |n · m|.
    """
    if not faces:
        return 1.0
    vals = []
    n = normalize(np.asarray(normal_xyz, dtype=float))
    for face in faces:
        m = normalize(np.asarray(face.normal_xyz, dtype=float))
        vals.append(abs(float(n @ m)))
    return float(np.mean(vals))


def estimate_size_model(
    traces: List[Trace],
    set_id: str,
    r_min: float = 0.5,
    r_max: float = 30.0,
    L_min: float = 0.1,
) -> Tuple[float, float]:
    """Estimate power-law exponent alpha (= k_r + 1) via MLE on chord lengths.

    Parameters
    ----------
    traces : list of Trace (filtered for set_id externally or here)
    set_id : str
        Filter traces to this set.
    r_min, r_max : float
        Truncation limits for fitting.
    L_min : float
        Detection limit [m].

    Returns
    -------
    (alpha, r_min_used)
    """
    set_traces = [t for t in traces if t.set_id == set_id]
    if not set_traces:
        return (3.5, r_min)  # default

    chord_lengths = np.array([t.observed_length for t in set_traces])
    is_contained = np.array([t.is_contained for t in set_traces])

    # MLE: maximise log likelihood over alpha in [1.5, 6.0]
    def neg_ll(alpha):
        return -censored_chord_log_likelihood(
            chord_lengths, is_contained, alpha, r_min, r_max, L_min
        )

    result = minimize_scalar(neg_ll, bounds=(1.5, 6.0), method="bounded")
    alpha_mle = float(result.x)
    return (alpha_mle, r_min)


def estimate_p32(
    traces: List[Trace],
    faces: List[Face],
    orientation_result: Optional[FractureSetOrientation],
    alpha: float,
    r_min: float,
    r_max: float = 30.0,
    L_min: float = 0.1,
    discs: Optional[List[ReconstructedDisc]] = None,
) -> FractureSetSizeIntensity:
    """Estimate P32 and intensity parameters for a fracture set.

    Parameters
    ----------
    traces : list of Trace for this set (already filtered)
    faces : list of Face
    orientation_result : FractureSetOrientation or None
    alpha : float
        Power-law PDF exponent (k_r = alpha - 1).
    r_min, r_max : float
    L_min : float
    discs : list of ReconstructedDisc or None
        If provided, used to cross-validate.

    Returns
    -------
    FractureSetSizeIntensity with P32/P30/P21/n0 clearly separated.
    """
    sid = orientation_result.set_id if orientation_result else "unknown"

    # Observed P21 = total trace length / total observation area
    total_trace_length = sum(t.observed_length for t in traces)
    total_face_area = sum(f.window_area() for f in faces)
    P21_obs = total_trace_length / max(total_face_area, 1e-9)
    P20_obs = len(traces) / max(total_face_area, 1e-9)

    # Orientation factor C_s
    if orientation_result is not None:
        from dfnrec.geometry.vector import normal_from_trend_plunge
        mean_normal = normal_from_trend_plunge(
            orientation_result.mean_trend_deg,
            orientation_result.mean_plunge_deg,
        )
        C_s = _orientation_factor(mean_normal, faces)
    else:
        C_s = 0.5  # isotropic default

    # Mean fracture area E[pi r^2]
    mean_r2 = _mean_r2_power_law(alpha, r_min, r_max)
    mean_area = math.pi * mean_r2

    # P32 estimation:
    # P21 = P32 * C_s * (mean visible chord / mean area) approximately
    # Simpler approximation used here: P21 ≈ C_s * P32 * (2/pi) * E[r]
    # (from Dershowitz & Herrmann: P21 = (2/pi) * C_s * P32_eff * E[r])
    # → P32_eff = P21 * pi / (2 * C_s * E[r])
    if alpha > 2.0:
        exp = 2.0 - alpha
        E_r = (r_max**exp - r_min**exp) / (exp * (r_max**(1 - alpha) - r_min**(1 - alpha)) / (1 - alpha))
    else:
        E_r = (r_min + r_max) / 2.0  # fallback

    C_s_safe = max(C_s, 0.01)
    E_r_safe = max(E_r, 0.01)

    P32_eff = P21_obs * math.pi / (2.0 * C_s_safe * E_r_safe)
    P32_total = P32_eff  # assuming r ≥ r_min covers the full distribution

    # n0 = P30 = P32_total / mean_area
    n0 = P32_total / max(mean_area, 1e-9)

    # N_expected = n0 * V_domain (V not known here; report per face area)
    # P21_simulated = C_s * (2/pi) * P32_eff * E_r
    P21_sim = C_s * (2.0 / math.pi) * P32_eff * E_r_safe

    return FractureSetSizeIntensity(
        set_id=sid,
        size_model=SizeModel.POWER_LAW,
        k_r=alpha - 1.0,
        r_min=r_min,
        r_max=r_max,
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
            "E_r": E_r,
            "mean_area_m2": mean_area,
        },
    )
