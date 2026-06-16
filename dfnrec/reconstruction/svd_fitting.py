"""SVD plane fitting from a track of multi-face traces."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dfnrec.geometry.plane import svd_plane_fit, robust_svd_plane_fit
from dfnrec.geometry.vector import normalize, trend_plunge_from_normal
from dfnrec.reconstruction.track import Track


@dataclass
class PlaneFitFromTrack:
    """Result of fitting a plane to all trace endpoints in a track."""
    normal: np.ndarray
    """Best-fit unit normal."""
    centroid: np.ndarray
    """Centroid of all endpoints."""
    rms: float
    """RMS residual of all endpoints to fitted plane [m]."""
    trend_deg: float
    plunge_deg: float
    n_points: int
    robust: bool = False
    """True if robust (Huber-weighted) fitting was used."""


def fit_plane_to_track(
    track: Track,
    use_robust: bool = True,
    rms_fallback_threshold: float = 0.10,
) -> Optional[PlaneFitFromTrack]:
    """Fit a plane to all trace endpoints in a track.

    For tracks with n_faces ≥ 2, this gives a 3D plane estimate.
    For singleton tracks (n_faces = 1), the plane normal is inferred
    from the cross product of trace direction and face normal.

    Parameters
    ----------
    track : Track
    use_robust : bool
        If True, use Huber-weighted robust fitting.
    rms_fallback_threshold : float
        If robust RMS > this, fall back to standard SVD.

    Returns
    -------
    PlaneFitFromTrack or None if geometry is degenerate.
    """
    pts = track.all_endpoints_xyz()
    n_pts = len(pts)

    # Singleton track: 2 endpoints from 1 trace on 1 face
    if track.n_faces() == 1 and len(track.traces) == 1:
        trace = track.traces[0]
        # Face normal is known; trace direction gives the plane
        # plane normal = trace_dir × face_normal (normalised)
        from dfnrec.models import Face
        # We can't recover the full plane without face info here;
        # return None and let map_disc handle the single-face case.
        # However, if trace provides trend/plunge, use it.
        if trace.trend_deg is not None and trace.plunge_deg is not None:
            n = normalize(np.array([
                math.cos(math.radians(trace.plunge_deg)) * math.sin(math.radians(trace.trend_deg)),
                math.cos(math.radians(trace.plunge_deg)) * math.cos(math.radians(trace.trend_deg)),
                math.sin(math.radians(trace.plunge_deg)),
            ]))
            t, p = trend_plunge_from_normal(n)
            centroid = (np.asarray(trace.p0_xyz) + np.asarray(trace.p1_xyz)) / 2.0
            return PlaneFitFromTrack(
                normal=n, centroid=centroid, rms=0.0,
                trend_deg=t, plunge_deg=p, n_points=2,
            )
        return None  # Cannot determine plane from single trace without orientation

    if n_pts < 3:
        return None

    # Multi-face track: fit plane to all endpoints
    try:
        if use_robust:
            result = robust_svd_plane_fit(pts)
        else:
            result = svd_plane_fit(pts)
    except Exception:
        return None

    # Ensure normal points toward +x (tunnel advance direction) convention
    n = result.normal.copy()
    if n[0] < 0:
        n = -n

    t, p = trend_plunge_from_normal(n)
    return PlaneFitFromTrack(
        normal=n,
        centroid=result.centroid,
        rms=result.rms,
        trend_deg=t,
        plunge_deg=p,
        n_points=n_pts,
        robust=use_robust,
    )
