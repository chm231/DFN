"""Vector utilities for DFN reconstruction geometry."""
from __future__ import annotations

import math
import numpy as np
from typing import Optional


def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector.  Raises if v is a zero vector."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Cannot normalize zero vector {v}")
    return v / n


def axial_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two **axial** (undirected) unit vectors [radians].

    Axial vectors represent undirected orientations: a and -a are the same.
    The result is always in [0, π/2].

    Parameters
    ----------
    a, b : (3,) array-like
        Unit vectors (normalised inside this function).
    """
    a = normalize(np.asarray(a, dtype=float))
    b = normalize(np.asarray(b, dtype=float))
    dot = np.clip(abs(np.dot(a, b)), 0.0, 1.0)  # axial: take |cos θ|
    return math.acos(dot)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two **polar** unit vectors [radians] in [0, π]."""
    a = normalize(np.asarray(a, dtype=float))
    b = normalize(np.asarray(b, dtype=float))
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.acos(dot)


def pca_line_direction(points: np.ndarray) -> np.ndarray:
    """Estimate the best-fit line direction through a set of 3D points.

    Returns the first principal component (unit vector).

    Parameters
    ----------
    points : (N, 3) array
        N ≥ 2 points in 3D.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(points) < 2:
        raise ValueError("Need at least 2 points")
    centred = points - points.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt[0]  # first row = largest singular value direction


def trend_plunge_from_normal(normal: np.ndarray) -> tuple[float, float]:
    """Convert a pole normal vector to (trend [deg], plunge [deg]).

    Convention: normal points into the upper hemisphere (z ≥ 0) before
    conversion.  Trend is azimuth from North (x=East, y=North frame) or
    from tunnel-axis (x=advance) depending on input.

    Assumes x=East, y=North, z=Up (right-hand).
    trend  = azimuth of normal projected on horizontal [0, 360) deg
    plunge = angle below horizontal [0, 90] deg
    """
    n = normalize(np.asarray(normal, dtype=float))
    if n[2] < 0:
        n = -n  # flip to upper hemisphere
    # Horizontal component magnitude
    horiz = math.sqrt(n[0] ** 2 + n[1] ** 2)
    plunge = math.degrees(math.atan2(n[2], horiz))
    trend = math.degrees(math.atan2(n[0], n[1])) % 360.0
    return trend, plunge


def normal_from_trend_plunge(trend_deg: float, plunge_deg: float) -> np.ndarray:
    """Convert (trend, plunge) to a unit pole normal vector.

    x=East, y=North, z=Up.
    """
    t = math.radians(trend_deg)
    p = math.radians(plunge_deg)
    cos_p = math.cos(p)
    return np.array([cos_p * math.sin(t), cos_p * math.cos(t), math.sin(p)])
