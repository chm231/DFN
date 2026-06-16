"""2D clipping and face-local UV coordinate utilities."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from dfnrec.geometry.vector import normalize


def local_uv_transform(
    origin_xyz: np.ndarray,
    axis_u_xyz: np.ndarray,
    axis_v_xyz: np.ndarray,
) -> tuple:
    """Return forward/inverse transform functions for face-local UV coordinates.

    Parameters
    ----------
    origin_xyz : (3,) array
        Face origin in global 3D.
    axis_u_xyz : (3,) array
        Unit u-axis in global 3D.
    axis_v_xyz : (3,) array
        Unit v-axis in global 3D.

    Returns
    -------
    xyz_to_uv : callable (N,3) -> (N,2)
    uv_to_xyz : callable (N,2) -> (N,3)
    """
    o = np.asarray(origin_xyz, dtype=float)
    u = normalize(np.asarray(axis_u_xyz, dtype=float))
    v = normalize(np.asarray(axis_v_xyz, dtype=float))

    def xyz_to_uv(pts_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xyz, dtype=float)
        rel = pts - o
        return np.column_stack([rel @ u, rel @ v])

    def _uv_to_xyz(pts_uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_uv, dtype=float)
        if pts.ndim == 1:
            return o + pts[0] * u + pts[1] * v
        return o + pts[:, 0:1] * u + pts[:, 1:2] * v

    return xyz_to_uv, _uv_to_xyz


def uv_to_xyz(
    pts_uv: np.ndarray,
    origin_xyz: np.ndarray,
    axis_u_xyz: np.ndarray,
    axis_v_xyz: np.ndarray,
) -> np.ndarray:
    """Convert face-local UV coordinates to global 3D XYZ."""
    o = np.asarray(origin_xyz, dtype=float)
    u = normalize(np.asarray(axis_u_xyz, dtype=float))
    v = normalize(np.asarray(axis_v_xyz, dtype=float))
    pts = np.asarray(pts_uv, dtype=float)
    if pts.ndim == 1:
        return o + pts[0] * u + pts[1] * v
    return o + pts[:, 0:1] * u + pts[:, 1:2] * v


def line_circle_intersection_2d(
    p: np.ndarray,
    d: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find intersections of a line with a circle in 2D.

    Parameters
    ----------
    p : (2,) array
        A point on the line.
    d : (2,) array
        Direction vector of the line (need not be unit).
    center : (2,) array
        Circle centre.
    radius : float
        Circle radius > 0.

    Returns
    -------
    (pt1, pt2) or None
        Two intersection points (may coincide for a tangent line).
        None if no real intersection.
    """
    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)
    c = np.asarray(center, dtype=float)

    # Parameterize: x = p + t*d
    # |p + t*d - c|^2 = radius^2
    oc = p - c
    a = d @ d
    b = 2.0 * (oc @ d)
    cc = (oc @ oc) - radius**2

    disc = b**2 - 4.0 * a * cc
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(max(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    return p + t1 * d, p + t2 * d


def _inside_half_plane(pt: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    """Sutherland-Hodgman: is pt on the inside of the directed edge?"""
    edge = edge_end - edge_start
    normal = np.array([-edge[1], edge[0]])  # left-pointing normal
    return float(normal @ (pt - edge_start)) >= 0.0


def _intersect_segment_edge(
    p1: np.ndarray,
    p2: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> np.ndarray:
    """Intersection of segment p1-p2 with infinite line edge_start→edge_end."""
    d1 = p2 - p1
    d2 = edge_end - edge_start
    # p1 + t*d1 = edge_start + s*d2
    # solve: d1*t - d2*s = edge_start - p1
    A = np.column_stack([d1, -d2])
    b = edge_start - p1
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return p1  # parallel / coincident — return first point
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    return p1 + t * d1


def segment_polygon_clip(
    p_start: np.ndarray,
    p_end: np.ndarray,
    polygon: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a 2D line segment to the interior of a convex polygon.

    Uses the Cohen-Sutherland / Sutherland-Hodgman approach adapted for a
    single segment rather than a polygon.

    Parameters
    ----------
    p_start, p_end : (2,) array
        Segment endpoints in UV space.
    polygon : (M, 2) array
        Convex polygon vertices (counter-clockwise).

    Returns
    -------
    (clipped_start, clipped_end) or None
        Clipped segment endpoints, or None if segment is fully outside.
    """
    poly = np.asarray(polygon, dtype=float)
    M = len(poly)
    # Represent segment as a degenerate polygon with 2 vertices
    seg = [np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float)]

    output = seg
    for i in range(M):
        if not output:
            return None
        input_list = output
        output = []
        edge_start = poly[i]
        edge_end = poly[(i + 1) % M]
        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]
            if _inside_half_plane(current, edge_start, edge_end):
                if not _inside_half_plane(previous, edge_start, edge_end):
                    output.append(_intersect_segment_edge(previous, current, edge_start, edge_end))
                output.append(current)
            elif _inside_half_plane(previous, edge_start, edge_end):
                output.append(_intersect_segment_edge(previous, current, edge_start, edge_end))

    if len(output) < 2:
        return None
    return output[0], output[-1]
