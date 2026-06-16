"""Disc-face intersection: predict visible trace on a tunnel face.

This module answers the key geometric question of DFN reconstruction:
given a fracture disc (center, normal, radius) and a tunnel face (plane +
observation window), what trace would be visible?

Coordinate convention
---------------------
  x = tunnel advance direction
  y = horizontal lateral
  z = vertical
All coordinates are in global 3D.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from dfnrec.geometry.vector import normalize
from dfnrec.geometry.clipping import (
    local_uv_transform,
    line_circle_intersection_2d,
    segment_polygon_clip,
)


@dataclass
class VisibleTraceResult:
    """Result of the disc-face trace prediction."""
    # --- Flags ---
    intersects_face: bool = False
    """True if the disc plane intersects the face plane at all."""
    chord_exists: bool = False
    """True if the intersection line passes through the disc (chord exists)."""
    clipped_by_window: bool = False
    """True if the chord was clipped by the observation window boundary."""
    fully_inside_window: bool = False
    """True if the entire chord lies within the observation window."""

    # --- Lengths ---
    full_chord_length: float = 0.0
    """Full chord of the disc cut by the face plane [m] (=2*sqrt(r²-h²), h=signed distance)."""
    visible_length: float = 0.0
    """Length of chord that lies within the observation window [m]."""

    # --- Endpoints in 3D global XYZ ---
    chord_p0_xyz: Optional[List[float]] = None
    """Chord endpoint 0 (disc rim intersection) in global XYZ."""
    chord_p1_xyz: Optional[List[float]] = None
    """Chord endpoint 1 (disc rim intersection) in global XYZ."""
    visible_p0_xyz: Optional[List[float]] = None
    """Visible trace endpoint 0 in global XYZ (may be window-clipped)."""
    visible_p1_xyz: Optional[List[float]] = None
    """Visible trace endpoint 1 in global XYZ (may be window-clipped)."""

    # --- Endpoints in face-local UV [m] ---
    chord_p0_uv: Optional[List[float]] = None
    chord_p1_uv: Optional[List[float]] = None
    visible_p0_uv: Optional[List[float]] = None
    visible_p1_uv: Optional[List[float]] = None

    # --- Midpoint 3D ---
    chord_midpoint_xyz: Optional[List[float]] = None
    visible_midpoint_xyz: Optional[List[float]] = None

    diagnostics: dict = field(default_factory=dict)


def predicted_visible_trace(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    radius_m: float,
    face_origin_xyz: np.ndarray,
    face_normal_xyz: np.ndarray,
    face_axis_u_xyz: np.ndarray,
    face_axis_v_xyz: np.ndarray,
    observation_window_uv: np.ndarray,
    tol: float = 1e-9,
) -> VisibleTraceResult:
    """Predict the visible trace of a fracture disc on a tunnel face.

    Workflow
    --------
    1. Compute signed distance from disc centre to face plane.
    2. If |dist| > radius, disc misses the face → no intersection.
    3. Find chord direction = disc_normal × face_normal (normalised).
    4. Find chord midpoint = disc_centre + proj of (-dist * face_normal) onto disc.
    5. Half-chord = sqrt(r² - dist²).
    6. Project chord into face UV; clip against observation window.
    7. Return VisibleTraceResult.

    Parameters
    ----------
    center_xyz : (3,) array
        Disc centre in global XYZ [m].
    normal_xyz : (3,) array
        Disc unit normal.
    radius_m : float
        Disc radius [m].
    face_origin_xyz : (3,) array
        A point on the face plane.
    face_normal_xyz : (3,) array
        Unit outward normal of the face.
    face_axis_u_xyz, face_axis_v_xyz : (3,) array
        Orthonormal basis vectors for face-local UV.
    observation_window_uv : (M, 2) array
        Convex polygon defining the observable region in face UV [m].
    tol : float
        Numerical tolerance.
    """
    result = VisibleTraceResult()
    c = np.asarray(center_xyz, dtype=float)
    n_disc = normalize(np.asarray(normal_xyz, dtype=float))
    n_face = normalize(np.asarray(face_normal_xyz, dtype=float))
    o_face = np.asarray(face_origin_xyz, dtype=float)
    window = np.asarray(observation_window_uv, dtype=float)

    # ── Step 1: check planes intersect ─────────────────────────────────
    cross = np.cross(n_disc, n_face)
    cross_mag = np.linalg.norm(cross)
    if cross_mag < tol:
        # Planes are parallel (disc plane parallel to face)
        result.intersects_face = False
        result.diagnostics["reason"] = "disc_plane_parallel_to_face"
        return result
    result.intersects_face = True
    chord_dir = cross / cross_mag  # direction of intersection line

    # ── Step 2: signed distance from disc centre to face plane ─────────
    dist = float(n_face @ (c - o_face))
    # dist > 0 → centre on the +normal side of face

    if abs(dist) > radius_m - tol:
        result.chord_exists = False
        result.diagnostics["reason"] = "disc_too_far_from_face"
        result.diagnostics["dist"] = dist
        return result
    result.chord_exists = True

    half_chord = math.sqrt(max(radius_m**2 - dist**2, 0.0))

    # ── Step 3: chord midpoint on face plane ───────────────────────────
    # Closest point from disc centre to face plane:  c - dist * n_face
    # Then project that point onto disc plane to get chord midpoint.
    # Since chord_dir ⊥ n_disc and ⊥ n_face, the midpoint is c - dist*n_face
    # projected along n_disc direction back onto disc plane:
    # midpoint_on_disc = (c - dist*n_face) - ((c - dist*n_face - c)·n_disc)*n_disc
    #                  = c - dist*n_face + dist*(n_face·n_disc)*n_disc
    # But chord midpoint is also on face plane.
    # Simpler: midpoint = c + (-dist)*n_face_proj_onto_disc
    # where n_face_proj_onto_disc is n_face minus its n_disc component.
    n_face_proj = n_face - (n_face @ n_disc) * n_disc  # in-disc component of n_face
    n_face_proj_norm = np.linalg.norm(n_face_proj)
    if n_face_proj_norm < tol:
        # n_face ∥ n_disc → already handled above by cross_mag check
        result.diagnostics["reason"] = "degenerate_projection"
        return result
    n_face_proj /= n_face_proj_norm
    # Distance from disc centre to face plane along n_face is `dist`.
    # The foot of the perpendicular from c to the face is c - dist*n_face.
    # Chord midpoint m is on both planes; project foot onto disc:
    foot = c - dist * n_face
    # Residual from disc centre to foot, in disc plane:
    foot_rel = foot - c
    d_in_disc = float(foot_rel @ n_face_proj)
    chord_mid = c + d_in_disc * n_face_proj

    # Verify chord_mid is on face plane (sanity)
    # chord_mid_dist_to_face = n_face @ (chord_mid - o_face)  # should ≈ 0

    chord_p0 = chord_mid - half_chord * chord_dir
    chord_p1 = chord_mid + half_chord * chord_dir
    result.full_chord_length = 2.0 * half_chord
    result.chord_p0_xyz = chord_p0.tolist()
    result.chord_p1_xyz = chord_p1.tolist()
    result.chord_midpoint_xyz = chord_mid.tolist()

    # ── Step 4: project chord onto face UV ─────────────────────────────
    xyz_to_uv, _uv_to_xyz = local_uv_transform(o_face, face_axis_u_xyz, face_axis_v_xyz)
    chord_p0_uv = xyz_to_uv(chord_p0.reshape(1, 3))[0]
    chord_p1_uv = xyz_to_uv(chord_p1.reshape(1, 3))[0]
    result.chord_p0_uv = chord_p0_uv.tolist()
    result.chord_p1_uv = chord_p1_uv.tolist()

    # ── Step 5: clip against observation window ─────────────────────────
    clipped = segment_polygon_clip(chord_p0_uv, chord_p1_uv, window)
    if clipped is None:
        # Chord falls entirely outside the window
        result.visible_length = 0.0
        result.clipped_by_window = True
        result.diagnostics["reason"] = "chord_outside_window"
        return result

    vis_p0_uv, vis_p1_uv = clipped
    vis_length = float(np.linalg.norm(vis_p1_uv - vis_p0_uv))
    result.visible_length = vis_length
    result.visible_p0_uv = vis_p0_uv.tolist()
    result.visible_p1_uv = vis_p1_uv.tolist()

    # Convert back to 3D
    axis_u = normalize(np.asarray(face_axis_u_xyz, dtype=float))
    axis_v = normalize(np.asarray(face_axis_v_xyz, dtype=float))
    vis_p0_xyz = o_face + vis_p0_uv[0] * axis_u + vis_p0_uv[1] * axis_v
    vis_p1_xyz = o_face + vis_p1_uv[0] * axis_u + vis_p1_uv[1] * axis_v
    result.visible_p0_xyz = vis_p0_xyz.tolist()
    result.visible_p1_xyz = vis_p1_xyz.tolist()
    result.visible_midpoint_xyz = ((vis_p0_xyz + vis_p1_xyz) / 2).tolist()

    # Determine if clipping occurred
    chord_len = result.full_chord_length
    eps = 1e-6
    if chord_len > eps and abs(vis_length - chord_len) / chord_len > 0.01:
        result.clipped_by_window = True
        result.fully_inside_window = False
    else:
        result.clipped_by_window = False
        result.fully_inside_window = True

    return result
