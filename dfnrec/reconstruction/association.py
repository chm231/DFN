"""Trace-to-disc association scoring.

For each pair of traces (from different faces), compute a log Bayes factor
that quantifies how likely they belong to the same fracture disc.

Factors considered
------------------
1. Orientation consistency (are trace directions compatible with a shared plane?)
2. Spatial consistency (do trace midpoints align with a shared plane and disc?)
3. Set membership prior (do both traces belong to the same set_id?)
4. Persistence prior (does a fracture large enough to span the faces make sense?)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dfnrec.models import Trace, Face
from dfnrec.geometry.vector import normalize, axial_angle
from dfnrec.geometry.plane import plane_plane_intersection


@dataclass
class AssociationScore:
    """Decomposed log Bayes factor for a trace pair."""
    log_p_orient: float
    """Log-likelihood contribution from orientation compatibility."""
    log_p_spatial: float
    """Log-likelihood contribution from spatial / plane-fit compatibility."""
    log_p_prior: float
    """Log prior: set-membership match and face-distance plausibility."""
    log_p_persist: float
    """Log prior: fracture persistence (large enough to span faces)."""
    log_bf: float
    """Total log Bayes factor = sum of all terms."""

    trace_i_id: str = ""
    trace_j_id: str = ""


def _trace_direction(trace: Trace) -> np.ndarray:
    """Unit direction vector of a trace (p0 → p1)."""
    p0 = np.asarray(trace.p0_xyz, dtype=float)
    p1 = np.asarray(trace.p1_xyz, dtype=float)
    d = p1 - p0
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        raise ValueError(f"Trace {trace.trace_id} has zero length")
    return d / norm


def _midpoint(trace: Trace) -> np.ndarray:
    p0 = np.asarray(trace.p0_xyz, dtype=float)
    p1 = np.asarray(trace.p1_xyz, dtype=float)
    return (p0 + p1) / 2.0


def compute_log_bayes_factor(
    trace_i: Trace,
    trace_j: Trace,
    face_i: Face,
    face_j: Face,
    sigma_angle_rad: float = math.radians(15.0),
    sigma_spatial_m: float = 0.5,
    r_min: float = 0.3,
    r_max: float = 30.0,
) -> AssociationScore:
    """Compute log Bayes factor for traces i and j belonging to the same disc.

    Parameters
    ----------
    trace_i, trace_j : Trace
        Two traces from different faces.
    face_i, face_j : Face
        The faces on which traces i and j were observed.
    sigma_angle_rad : float
        Prior angular tolerance [rad]. Larger = more permissive.
    sigma_spatial_m : float
        Prior spatial tolerance [m]. Larger = more permissive.
    r_min, r_max : float
        Plausible radius range for persistence prior [m].
    """
    # ── 1. Orientation compatibility ───────────────────────────────────
    # Each trace direction lies in the fracture plane.
    # If they share a plane, the cross product (chord_dir × face_normal) should
    # point in the same direction for both traces.
    di = _trace_direction(trace_i)
    dj = _trace_direction(trace_j)
    n_fi = normalize(np.asarray(face_i.normal_xyz, dtype=float))
    n_fj = normalize(np.asarray(face_j.normal_xyz, dtype=float))

    # Candidate disc normals from each trace:
    # n = trace_dir × face_normal (in-face normal of fracture)
    ni_candidate = np.cross(di, n_fi)
    nj_candidate = np.cross(dj, n_fj)
    mag_i = np.linalg.norm(ni_candidate)
    mag_j = np.linalg.norm(nj_candidate)

    if mag_i < 1e-9 or mag_j < 1e-9:
        # Trace is parallel to face normal → degenerate
        return AssociationScore(
            log_p_orient=-1e6, log_p_spatial=-1e6,
            log_p_prior=-1e6, log_p_persist=-1e6, log_bf=-1e6,
            trace_i_id=trace_i.trace_id, trace_j_id=trace_j.trace_id,
        )

    ni_candidate /= mag_i
    nj_candidate /= mag_j

    ang = axial_angle(ni_candidate, nj_candidate)
    log_p_orient = -0.5 * (ang / sigma_angle_rad) ** 2

    # ── 2. Spatial consistency ─────────────────────────────────────────
    # Fit plane through midpoints and check residuals.
    mid_i = _midpoint(trace_i)
    mid_j = _midpoint(trace_j)

    # Candidate plane normal (average of two candidates, axially)
    n_plane = (ni_candidate + nj_candidate) if np.dot(ni_candidate, nj_candidate) >= 0 else (ni_candidate - nj_candidate)
    mag_plane = np.linalg.norm(n_plane)
    if mag_plane < 1e-9:
        n_plane = ni_candidate
    else:
        n_plane /= mag_plane

    # Residuals of midpoints to candidate plane (using face origin as anchor)
    origin_i = np.asarray(face_i.origin_xyz, dtype=float)
    d_plane = float(n_plane @ origin_i)  # plane passes through face_i origin approximately
    res_i = abs(float(n_plane @ mid_i) - d_plane)
    res_j = abs(float(n_plane @ mid_j) - d_plane)
    spatial_residual = math.sqrt(res_i**2 + res_j**2)
    log_p_spatial = -0.5 * (spatial_residual / sigma_spatial_m) ** 2

    # ── 3. Set membership prior ────────────────────────────────────────
    if trace_i.set_id is not None and trace_j.set_id is not None:
        log_p_prior = 0.0 if trace_i.set_id == trace_j.set_id else -4.0
    else:
        log_p_prior = -0.5  # unknown: slight penalty

    # ── 4. Persistence prior ───────────────────────────────────────────
    # Minimum radius needed to span the two faces: ≈ distance between midpoints
    d_centres = float(np.linalg.norm(mid_j - mid_i))
    r_needed = d_centres / 2.0

    if r_needed < r_min:
        log_p_persist = 0.0  # easily satisfied
    elif r_needed > r_max:
        log_p_persist = -1e3  # implausibly large
    else:
        # Power-law prior: P(r ≥ r_needed) ∝ (r_min/r_needed)^k_r with k_r~2.5
        k_r = 2.5
        log_p_persist = -k_r * math.log(r_needed / r_min)

    log_bf = log_p_orient + log_p_spatial + log_p_prior + log_p_persist
    return AssociationScore(
        log_p_orient=log_p_orient,
        log_p_spatial=log_p_spatial,
        log_p_prior=log_p_prior,
        log_p_persist=log_p_persist,
        log_bf=log_bf,
        trace_i_id=trace_i.trace_id,
        trace_j_id=trace_j.trace_id,
    )
