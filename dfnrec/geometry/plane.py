"""Plane fitting and intersection utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dfnrec.geometry.vector import normalize


@dataclass
class PlaneFitResult:
    """Result of SVD plane fitting."""
    normal: np.ndarray
    """Unit normal vector (pointing toward the smallest singular value direction)."""
    centroid: np.ndarray
    """Mean of input points."""
    rms: float
    """RMS of point-to-plane residuals [same units as input points]."""
    singular_values: np.ndarray
    """All three singular values (largest first)."""


def svd_plane_fit(
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> PlaneFitResult:
    """Fit a plane to 3D points using (optionally weighted) SVD.

    The plane equation is: normal · (x - centroid) = 0.

    Parameters
    ----------
    points : (N, 3) array
        N ≥ 3 points in 3D.
    weights : (N,) array or None
        Non-negative weights. If None, uniform weights are used.

    Returns
    -------
    PlaneFitResult
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must be (N,)")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    w_sum = w.sum()
    if w_sum < 1e-12:
        raise ValueError("Sum of weights is zero")

    centroid = (pts * w[:, None]).sum(axis=0) / w_sum
    centred = pts - centroid
    # Apply sqrt(w) to each row before SVD
    A = centred * np.sqrt(w[:, None])
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    normal = Vt[-1]  # last row = smallest singular value direction

    # RMS residuals
    residuals = centred @ normal
    rms = float(np.sqrt((w * residuals**2).sum() / w_sum))

    return PlaneFitResult(
        normal=normal,
        centroid=centroid,
        rms=rms,
        singular_values=sv,
    )


def robust_svd_plane_fit(
    points: np.ndarray,
    max_iter: int = 10,
    c: float = 1.345,
) -> PlaneFitResult:
    """Iteratively re-weighted SVD plane fit using Huber weighting.

    Parameters
    ----------
    points : (N, 3) array
    max_iter : int
        Maximum Huber iterations.
    c : float
        Huber threshold (in units of MAD-scaled residuals).
    """
    pts = np.asarray(points, dtype=float)
    w = np.ones(len(pts), dtype=float)

    for _ in range(max_iter):
        result = svd_plane_fit(pts, w)
        residuals = (pts - result.centroid) @ result.normal
        mad = np.median(np.abs(residuals))
        sigma = mad / 0.6745 + 1e-12  # robust σ estimate
        r_scaled = np.abs(residuals) / (c * sigma)
        # Huber weights: 1 if |r| <= c*sigma, else c*sigma/|r|
        w_new = np.where(r_scaled <= 1.0, 1.0, 1.0 / r_scaled)
        if np.allclose(w, w_new, rtol=1e-4):
            break
        w = w_new

    return svd_plane_fit(pts, w)


@dataclass
class PlaneIntersectionResult:
    """Result of plane-plane intersection."""
    direction: np.ndarray
    """Unit direction vector of the intersection line."""
    point: np.ndarray
    """A point on the intersection line (minimum-norm solution)."""
    is_parallel: bool
    """True if planes are (nearly) parallel — no intersection line."""


def plane_plane_intersection(
    n1: np.ndarray,
    d1: float,
    n2: np.ndarray,
    d2: float,
    tol: float = 1e-9,
) -> PlaneIntersectionResult:
    """Compute the intersection line of two planes.

    Plane equations:
      n1 · x = d1
      n2 · x = d2

    Parameters
    ----------
    n1, n2 : (3,) array
        Unit normal vectors of the two planes.
    d1, d2 : float
        Plane offsets: n · origin.
    tol : float
        Parallelism tolerance on |n1 × n2|.
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    direction = np.cross(n1, n2)
    mag = np.linalg.norm(direction)
    if mag < tol:
        return PlaneIntersectionResult(
            direction=direction, point=np.zeros(3), is_parallel=True
        )
    direction = direction / mag

    # Find a point on the line: solve via least-squares (3 equations, 3 unknowns
    # but rank 2; use pseudo-inverse)
    A = np.array([n1, n2, direction])
    b = np.array([d1, d2, 0.0])
    point = np.linalg.lstsq(A, b, rcond=None)[0]

    return PlaneIntersectionResult(direction=direction, point=point, is_parallel=False)
