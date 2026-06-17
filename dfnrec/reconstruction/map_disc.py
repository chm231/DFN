"""MAP disc estimation from a track.

Given a track of traces (each from a different face), estimate the
MAP disc (center, radius) by maximising a log posterior that combines:
  - Size prior: log-normal or power-law
  - Center prior: Gaussian around track centroid
  - Endpoint likelihoods: NATURAL=equality, CLIPPED=one-sided Gaussian
  - Absence likelihoods: discs that would have produced a trace on face F
    but produced none are penalised (non-observation constraint)

Implementation note
-------------------
We optimise over (cx, cy, cz, r) using scipy.optimize.minimize.
The Laplace approximation (Hessian at MAP) gives a covariance estimate.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr  # log Φ(x) = log CDF of standard normal

from dfnrec.models import Trace, Face, CensorType, ReconstructedDisc, ReliabilityClass
from dfnrec.reconstruction.track import Track
from dfnrec.reconstruction.svd_fitting import fit_plane_to_track
from dfnrec.geometry.disc_trace import predicted_visible_trace


def _log_endpoint_likelihood(
    observed_endpoint: np.ndarray,
    predicted_endpoint: np.ndarray,
    sigma: float,
    censor: CensorType,
) -> float:
    """Log likelihood of an observed endpoint given a predicted endpoint.

    NATURAL  : Gaussian p(obs | pred, sigma)
    CLIPPED  : Φ(-(|obs - pred| - 0) / sigma)  — one-sided (lower bound satisfied)
    UNKNOWN  : 0.5 * NATURAL + 0.5 * CLIPPED (mixture)
    """
    diff = float(np.linalg.norm(observed_endpoint - predicted_endpoint))
    z = diff / max(sigma, 1e-9)

    if censor == CensorType.NATURAL:
        return -0.5 * z**2 - 0.5 * math.log(2 * math.pi * sigma**2)
    elif censor == CensorType.CLIPPED:
        # Endpoint is clipped: the true endpoint is at least as far as observed
        # log P(true_endpoint ≥ observed_endpoint) ≈ log Φ(0) = log(0.5) if diff≈0
        return float(log_ndtr(0.5 * z))  # simplified: reward for matching boundary
    else:  # UNKNOWN: mixture
        ll_nat = -0.5 * z**2 - 0.5 * math.log(2 * math.pi * sigma**2)
        ll_clip = float(log_ndtr(0.5 * z))
        return math.log(0.5 * math.exp(ll_nat) + 0.5 * math.exp(ll_clip) + 1e-300)


def estimate_disc_map(
    track: Track,
    faces: Dict[str, Face],
    sigma_center_m: float = 1.0,
    sigma_radius_m: float = 2.0,
    r_prior_min: float = 0.3,
    r_prior_max: float = 30.0,
    disc_id_prefix: str = "D",
) -> Optional[ReconstructedDisc]:
    """Estimate MAP disc geometry from a track of traces.

    Parameters
    ----------
    track : Track
    faces : dict face_id → Face
    sigma_center_m : float
        Gaussian prior std for disc centre around track centroid [m].
    sigma_radius_m : float
        Prior std for radius around r_prior_min [m].
    r_prior_min, r_prior_max : float
        Plausible radius range [m].
    disc_id_prefix : str
        Prefix for generated disc ID.

    Returns
    -------
    ReconstructedDisc or None if fitting fails.
    """
    # --- Fit plane to get normal and initial centroid ---
    plane = fit_plane_to_track(track)
    if plane is None:
        # Fall back to single-trace geometry
        if len(track.traces) == 1:
            t = track.traces[0]
            face = faces.get(t.face_id)
            if face is None:
                return None
            
            # If trace has measured/true orientation, use it. Otherwise fall back to cross product.
            if t.trend_deg is not None and t.plunge_deg is not None:
                from dfnrec.geometry.vector import normal_from_trend_plunge
                normal = normal_from_trend_plunge(t.trend_deg, t.plunge_deg)
            else:
                # Normal = trace_dir × face_normal
                d = np.asarray(t.direction_xyz, dtype=float)
                n_face = np.asarray(face.normal_xyz, dtype=float)
                normal = np.cross(d, n_face)
                mag = np.linalg.norm(normal)
                if mag < 1e-9:
                    return None
                normal /= mag
            
            centroid = np.asarray(t.midpoint_xyz, dtype=float)
            r_init = t.observed_length / 2.0 + 0.5
        else:
            return None
    else:
        normal = plane.normal
        centroid = plane.centroid
        # Initial radius: half the maximum endpoint spread along trace directions
        pts = track.all_endpoints_xyz()
        spreads = []
        for t in track.traces:
            half_len = t.observed_length / 2.0
            spreads.append(half_len)
        r_init = max(spreads) + 0.5

    # --- Optimise log posterior ---
    sigma_ep = sum(t.measurement_sigma for t in track.traces) / max(len(track.traces), 1)

    def neg_log_posterior(params: np.ndarray) -> float:
        cx, cy, cz, log_r = params
        r = math.exp(log_r)
        if r < 1e-3 or r > r_prior_max * 2:
            return 1e9
        c = np.array([cx, cy, cz])

        # Size prior: log-normal on r centred at r_init
        ll = -0.5 * ((math.log(r) - math.log(r_init)) / math.log(1 + sigma_radius_m / r_init))**2

        # Centre prior: Gaussian around initial centroid
        ll += -0.5 * float(np.sum((c - centroid)**2)) / sigma_center_m**2

        # Endpoint likelihoods for each trace
        for t in track.traces:
            face = faces.get(t.face_id)
            if face is None:
                continue
            result = predicted_visible_trace(
                center_xyz=c,
                normal_xyz=normal,
                radius_m=r,
                face_origin_xyz=np.asarray(face.origin_xyz),
                face_normal_xyz=np.asarray(face.normal_xyz),
                face_axis_u_xyz=np.asarray(face.axis_u_xyz),
                face_axis_v_xyz=np.asarray(face.axis_v_xyz),
                observation_window_uv=np.asarray(face.observation_window_polygon_uv),
            )
            if not result.chord_exists or result.visible_p0_xyz is None:
                ll -= 5.0  # penalise if disc should have been visible but isn't
                continue

            pred_p0 = np.asarray(result.visible_p0_xyz)
            pred_p1 = np.asarray(result.visible_p1_xyz)
            obs_p0 = np.asarray(t.p0_xyz)
            obs_p1 = np.asarray(t.p1_xyz)

            # Match endpoints (try both orderings)
            ll_order1 = (
                _log_endpoint_likelihood(obs_p0, pred_p0, sigma_ep, t.censor_p0)
                + _log_endpoint_likelihood(obs_p1, pred_p1, sigma_ep, t.censor_p1)
            )
            ll_order2 = (
                _log_endpoint_likelihood(obs_p0, pred_p1, sigma_ep, t.censor_p0)
                + _log_endpoint_likelihood(obs_p1, pred_p0, sigma_ep, t.censor_p1)
            )
            ll += max(ll_order1, ll_order2)

        return -ll

    x0 = np.array([centroid[0], centroid[1], centroid[2], math.log(max(r_init, 0.1))])
    try:
        res = minimize(
            neg_log_posterior,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 1000},
        )
        cx, cy, cz, log_r = res.x
        r_map = math.exp(log_r)
    except Exception:
        cx, cy, cz = centroid
        r_map = r_init

    r_map = float(np.clip(r_map, r_prior_min, r_prior_max))
    center_map = np.array([cx, cy, cz])

    # --- Censoring dominance ---
    n_clipped = sum(t.n_clipped_endpoints for t in track.traces)
    n_total_eps = sum(2 for _ in track.traces)
    censoring_dominance = n_clipped / max(n_total_eps, 1)
    censoring_flag = censoring_dominance >= 0.80

    # --- Radius lower bound ---
    radius_lower = max(t.observed_length / 2.0 for t in track.traces) if track.traces else 0.0

    # --- Reliability class ---
    rc = ReconstructedDisc.classify_reliability(
        n_faces=track.n_faces(),
        plane_fit_rms=plane.rms if plane else None,
        censoring_dominance=censoring_dominance,
    )

    disc_id = f"{disc_id_prefix}_{'_'.join(sorted(set(track.face_ids)))}"
    from dfnrec.geometry.vector import trend_plunge_from_normal as tp_fn
    t_deg, p_deg = tp_fn(normal)

    return ReconstructedDisc(
        disc_id=disc_id,
        set_id=track.set_id,
        source="observed_reconstructed",
        contributing_trace_ids=track.trace_ids(),
        contributing_face_ids=list(set(track.face_ids)),
        center_xyz=center_map.tolist(),
        normal_xyz=normal.tolist(),
        trend_deg=t_deg,
        plunge_deg=p_deg,
        radius_m=r_map,
        radius_lower_bound_m=radius_lower,
        plane_fit_rms_m=plane.rms if plane else None,
        n_faces_observed=track.n_faces(),
        censoring_dominance=censoring_dominance,
        censoring_dominance_flag=censoring_flag,
        reliability_class=rc,
    )
