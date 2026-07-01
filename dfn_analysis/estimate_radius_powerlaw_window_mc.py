import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np


EPS = 1e-12
WINDOW_WARNING_POLYGON = "v4.1 polygon window MC uses tunnel polygon clipping; rough face mesh geometry is not directly modeled yet"
WINDOW_WARNING_BBOX = "bbox window MC is a fallback/diagnostic approximation, not final window-aware likelihood"
SET4_WARNING = "Set 4 is high-censoring; final status capped at provisional_ok"

# ---------------------------------------------------------------------------
# Fisher orientation parameters derived from DFN normals
# (trend deg, plunge deg, kappa) — tunnel X-axis, lower-hemisphere convention
# ---------------------------------------------------------------------------
FORSMARK_FISHER_PARAMS: Dict[int, Tuple[float, float, float]] = {
    1: (182.8, -1.7, 22.1),
    2: (134.8, -2.7, 21.8),
    3: (229.4, -2.2, 24.0),
    4: (252.1,  0.7,  3.1),
    5: (176.9, 18.0,  0.9),
}
LAXEMAR_FISHER_PARAMS: Dict[int, Tuple[float, float, float]] = {
    1: (118.3,  4.3,  4.7),
    2: (169.6, -0.2, 20.1),
    3: (233.9,  0.9,  7.0),
    4: (186.4,-11.8,  0.8),
    5: (207.0, 24.4, 24.0),
}
SITE_FISHER_PARAMS: Dict[str, Dict[int, Tuple[float, float, float]]] = {
    "forsmark": FORSMARK_FISHER_PARAMS,
    "laxemar":  LAXEMAR_FISHER_PARAMS,
}
SITE_SET_SUPPORT_INFO: Dict[str, Dict[int, Dict[str, float | str]]] = {
    "forsmark": {
        1: {"type": "powerlaw", "table_r0": 0.28},
        2: {"type": "powerlaw", "table_r0": 0.25},
        3: {"type": "powerlaw", "table_r0": 0.14},
        4: {"type": "powerlaw", "table_r0": 0.15},
        5: {"type": "powerlaw", "table_r0": 0.25},
    },
    "laxemar": {
        1: {"type": "powerlaw", "table_r0": 0.328},
        2: {"type": "powerlaw", "table_r0": 0.977},
        3: {"type": "powerlaw", "table_r0": 0.858},
        4: {"type": "exponential", "table_r0": 4.0},
        5: {"type": "powerlaw", "table_r0": 0.400},
    },
}

# ---------------------------------------------------------------------------
# Recovery / adoption status thresholds
# ---------------------------------------------------------------------------
_RECOVERY_GOOD_THRESHOLD      = 0.3
_RECOVERY_MODERATE_THRESHOLD  = 0.6


def mean_pole_from_trend_plunge(trend_deg: float, plunge_deg: float) -> np.ndarray:
    """Convert (trend, plunge) in degrees to a unit pole vector [x, y, z]."""
    t = np.radians(trend_deg)
    p = np.radians(plunge_deg)
    # NED convention → x=East, y=North, z=Up  (here we use xyz with z upward)
    # pole unit vector: (cos p cos t, cos p sin t, -sin p)
    return np.array([np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), -np.sin(p)])


def sample_fisher_normals(mean_pole: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n unit normals from a Von Mises-Fisher distribution on S^2
    with mean direction mean_pole and concentration kappa.
    Uses the Wood (1994) rejection-sampler when kappa > 0,
    falls back to uniform sphere when kappa == 0.
    """
    mu = mean_pole / np.linalg.norm(mean_pole)
    if kappa < 1e-4:
        # Uniform sphere
        v = rng.standard_normal((n, 3))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-12)

    # Rejection sampler (Ulrich 1984 / Wood 1994)
    samples = np.empty((n, 3), dtype=np.float64)
    generated = 0
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + 4.0)) / 2.0
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + 2.0 * np.log(1.0 - x0 ** 2)

    while generated < n:
        batch = max(n - generated, 256)
        Z = rng.beta(1.0, 1.0, size=batch)
        W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
        U = rng.uniform(0.0, 1.0, size=batch)
        accept = kappa * W + 2.0 * np.log(1.0 - x0 * W) - c >= np.log(U)
        w = W[accept]
        if len(w) == 0:
            continue
        # Random unit vectors in the plane perp to mu
        v_raw = rng.standard_normal((len(w), 3))
        v_raw -= (v_raw @ mu)[:, None] * mu
        nrm = np.linalg.norm(v_raw, axis=1, keepdims=True)
        v_hat = v_raw / np.maximum(nrm, 1e-12)
        s = np.sqrt(np.maximum(1.0 - w ** 2, 0.0))
        pts = s[:, None] * v_hat + w[:, None] * mu
        take = min(len(pts), n - generated)
        samples[generated : generated + take] = pts[:take]
        generated += take

    return samples


def normals_to_trace_directions_yz(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the YZ-plane trace direction for each fracture normal when
    intersected by the tunnel face (X = const plane, normal = [1, 0, 0]).

    trace direction = n × x_axis   (cross product gives strike line on face)

    Returns:
        directions_yz : (M, 2) array of unit YZ directions
        valid         : (N,) bool mask — False if n is parallel to X-axis
    """
    x_axis = np.array([1.0, 0.0, 0.0])
    crosses = np.cross(normals, x_axis)          # (N, 3)
    norms = np.linalg.norm(crosses, axis=1)      # (N,)
    valid = norms > 1e-6
    directions_yz = np.zeros((len(normals), 2), dtype=np.float64)
    directions_yz[valid] = crosses[valid, 1:] / norms[valid, np.newaxis]
    # Canonical orientation: flip so first non-zero component is positive
    for i in np.where(valid)[0]:
        d = directions_yz[i]
        if d[0] < 0.0 or (abs(d[0]) < 1e-12 and d[1] < 0.0):
            directions_yz[i] *= -1.0
    return directions_yz, valid


def orientation_conditioned_trace_directions_yz(
    set_id: int,
    site: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n trace directions on the YZ face from the orientation-conditioned model:
      1. Sample fracture poles from Fisher(mean_pole, kappa)
      2. Convert to YZ trace directions via cross product with X-axis
      3. Reject normals parallel to X-axis (prob ≈ 0 for typical kappa)
    Returns (M, 2) array of unit directions, M <= n.
    """
    params = SITE_FISHER_PARAMS.get(site, {}).get(set_id)
    if params is None:
        raise ValueError(f"No Fisher params for site={site}, set_id={set_id}")
    trend, plunge, kappa = params
    mu = mean_pole_from_trend_plunge(trend, plunge)

    # Oversample to account for rejection
    oversample = max(n * 2, 1000)
    normals = sample_fisher_normals(mu, kappa, oversample, rng)
    dirs_yz, valid = normals_to_trace_directions_yz(normals)
    dirs_yz = dirs_yz[valid]
    if len(dirs_yz) == 0:
        # Fallback: uniform directions
        angles = rng.uniform(0, np.pi, n)
        dirs_yz = np.column_stack([np.cos(angles), np.sin(angles)])
    return dirs_yz


def determine_recovery_status(kr_hat: float, kr_true: Optional[float]) -> str:
    if kr_true is None or not np.isfinite(kr_true):
        return "unknown"
    err = abs(kr_hat - kr_true)
    if err <= _RECOVERY_GOOD_THRESHOLD:
        return "good_recovery"
    if err <= _RECOVERY_MODERATE_THRESHOLD:
        return "moderate_recovery"
    return "failed_recovery"


def determine_adoption_status(
    fit_status: str,
    recovery_status: str,
    rmin_support_status: str = "matched",
) -> str:
    if rmin_support_status != "matched":
        return "diagnostic_only_rmin_support_mismatch"
    good_fit = fit_status in ("ok", "provisional_ok")
    if good_fit and recovery_status == "good_recovery":
        return "accepted"
    if good_fit and recovery_status in ("moderate_recovery", "unknown"):
        return "provisional_accepted"
    return "rejected"


def load_trace_data_from_h5(h5_path: str) -> tuple[List[dict], Optional[np.ndarray]]:
    rows: List[dict] = []
    polygon_yz = None
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        if "meta" in f and "tunnel_poly_yz" in f["meta"]:
            polygon_yz = f["meta/tunnel_poly_yz"][:].astype(np.float64)

        grp = f["traces"]
        p0 = grp["p0_xyz"][:].astype(np.float64)
        p1 = grp["p1_xyz"][:].astype(np.float64)
        radius_m = grp["radius_m"][:].astype(np.float64) if "radius_m" in grp else None
        for idx in range(len(grp["set_id"])):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "face_id": int(grp["face_id"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "radius_m": float(radius_m[idx]) if radius_m is not None else float("nan"),
                    "p0_y": float(p0[idx, 1]),
                    "p0_z": float(p0[idx, 2]),
                    "p1_y": float(p1[idx, 1]),
                    "p1_z": float(p1[idx, 2]),
                }
            )
    return rows, polygon_yz


def load_trace_rmin_metadata_from_h5(h5_path: str) -> dict:
    metadata = {
        "generation_rmin": None,
        "generation_rmax": None,
        "set_ids": None,
        "set_table_r0": None,
        "set_generation_rmin": None,
        "set_effective_rmin": None,
    }
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            return metadata
        meta = f["meta"]
        if "generation_rmin" in meta:
            metadata["generation_rmin"] = float(np.asarray(meta["generation_rmin"][()]).ravel()[0])
        if "generation_rmax" in meta:
            metadata["generation_rmax"] = float(np.asarray(meta["generation_rmax"][()]).ravel()[0])
        for key in ("set_ids", "set_table_r0", "set_generation_rmin", "set_effective_rmin"):
            if key in meta:
                metadata[key] = meta[key][:].ravel()
    return metadata


def build_set_rmin_lookup(
    site: str,
    global_generation_rmin: float,
    trace_rmin_metadata: dict,
) -> Dict[int, dict]:
    lookup: Dict[int, dict] = {}
    meta_ids = trace_rmin_metadata.get("set_ids")
    meta_table_r0 = trace_rmin_metadata.get("set_table_r0")
    meta_generation = trace_rmin_metadata.get("set_generation_rmin")
    meta_effective = trace_rmin_metadata.get("set_effective_rmin")

    if meta_ids is not None:
        for idx, set_id in enumerate(np.asarray(meta_ids, dtype=np.int32)):
            lookup[int(set_id)] = {
                "table_r0": float(meta_table_r0[idx]) if meta_table_r0 is not None else float("nan"),
                "generation_rmin": float(meta_generation[idx]) if meta_generation is not None else float(global_generation_rmin),
                "effective_generation_rmin": float(meta_effective[idx]) if meta_effective is not None else float(global_generation_rmin),
            }

    for set_id, info in SITE_SET_SUPPORT_INFO.get(site, {}).items():
        table_r0 = float(info["table_r0"])
        dist_type = str(info["type"])
        if dist_type == "powerlaw":
            effective_rmin = max(float(global_generation_rmin), table_r0)
        else:
            effective_rmin = float(global_generation_rmin)
        lookup.setdefault(
            int(set_id),
            {
                "table_r0": table_r0,
                "generation_rmin": effective_rmin,
                "effective_generation_rmin": effective_rmin,
            },
        )
    return lookup


def resolve_set_likelihood_rmin(
    set_id: int,
    set_rmin_mode: str,
    global_rmin: float,
    set_rmin_lookup: Dict[int, dict],
) -> tuple[float, float, float, str]:
    set_meta = set_rmin_lookup.get(int(set_id), {})
    table_r0 = float(set_meta.get("table_r0", np.nan))
    effective_generation_rmin = float(set_meta.get("effective_generation_rmin", global_rmin))

    if set_rmin_mode == "effective_generation":
        likelihood_rmin = effective_generation_rmin
    elif set_rmin_mode == "table_r0":
        likelihood_rmin = table_r0 if np.isfinite(table_r0) else float(global_rmin)
    else:
        likelihood_rmin = float(global_rmin)

    tolerance = 1e-6
    if likelihood_rmin < effective_generation_rmin - tolerance:
        support_status = "estimator_lower_than_generated_support"
    elif abs(likelihood_rmin - effective_generation_rmin) <= tolerance:
        support_status = "matched"
    else:
        support_status = "estimator_higher_than_generated_support"
    return likelihood_rmin, effective_generation_rmin, table_r0, support_status


def load_trace_data_from_csv(csv_path: str) -> tuple[List[dict], Optional[np.ndarray]]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"set_id", "face_id", "observed_length_m", "censoring_class", "p0_y", "p0_z", "p1_y", "p1_z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "face_id": int(row["face_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                    "radius_m": float(row["radius_m"]) if "radius_m" in row and row["radius_m"] else float("nan"),
                    "p0_y": float(row["p0_y"]),
                    "p0_z": float(row["p0_z"]),
                    "p1_y": float(row["p1_y"]),
                    "p1_z": float(row["p1_z"]),
                }
            )
    return rows, None


def group_rows_by_set(rows: Sequence[dict], target_sets: Optional[Set[int]]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        set_id = int(row["set_id"])
        if target_sets is not None and set_id not in target_sets:
            continue
        grouped.setdefault(set_id, []).append(row)
    return {set_id: grouped[set_id] for set_id in sorted(grouped)}


def empirical_trace_directions_yz(rows: Sequence[dict]) -> np.ndarray:
    directions = []
    for row in rows:
        dy = float(row["p1_y"]) - float(row["p0_y"])
        dz = float(row["p1_z"]) - float(row["p0_z"])
        norm = float(np.hypot(dy, dz))
        if norm <= 0.0:
            continue
        direction = np.array([dy / norm, dz / norm], dtype=np.float64)
        if direction[0] < 0.0 or (abs(direction[0]) < 1e-12 and direction[1] < 0.0):
            direction *= -1.0
        directions.append(direction)
    if not directions:
        return np.array([[1.0, 0.0]], dtype=np.float64)
    return np.vstack(directions)


def point_in_polygon(point_yz: np.ndarray, polygon_yz: np.ndarray) -> bool:
    y, z = float(point_yz[0]), float(point_yz[1])
    inside = False
    n = len(polygon_yz)
    for i in range(n):
        y0, z0 = polygon_yz[i]
        y1, z1 = polygon_yz[(i + 1) % n]
        crosses = (z0 > z) != (z1 > z)
        if crosses:
            y_intersect = (y1 - y0) * (z - z0) / (z1 - z0 + EPS) + y0
            if y < y_intersect:
                inside = not inside
    return inside


def _segment_intersection_t(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> Optional[float]:
    r = p1 - p0
    s = q1 - q0
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    qp = q0 - p0
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
        return float(np.clip(t, 0.0, 1.0))
    return None


def clip_segment_to_polygon(p0_yz: np.ndarray, p1_yz: np.ndarray, polygon_yz: np.ndarray) -> tuple[float, int]:
    t_values = [0.0, 1.0]
    for idx in range(len(polygon_yz)):
        t = _segment_intersection_t(p0_yz, p1_yz, polygon_yz[idx], polygon_yz[(idx + 1) % len(polygon_yz)])
        if t is not None:
            t_values.append(t)
    t_values = sorted(set(round(t, 12) for t in t_values))

    visible_length = 0.0
    segment_length = float(np.linalg.norm(p1_yz - p0_yz))
    for a, b in zip(t_values[:-1], t_values[1:]):
        if b <= a:
            continue
        mid = p0_yz + 0.5 * (a + b) * (p1_yz - p0_yz)
        if point_in_polygon(mid, polygon_yz):
            visible_length += (b - a) * segment_length

    p0_inside = point_in_polygon(p0_yz, polygon_yz)
    p1_inside = point_in_polygon(p1_yz, polygon_yz)
    if visible_length <= 1e-10:
        return 0.0, -1
    if p0_inside and p1_inside:
        return visible_length, 0
    if p0_inside or p1_inside:
        return visible_length, 1
    return visible_length, 2


def clip_segments_to_bbox_vectorized(
    centers_yz: np.ndarray,
    directions_yz: np.ndarray,
    true_lengths: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0 = centers_yz - 0.5 * true_lengths[:, None] * directions_yz
    p1 = centers_yz + 0.5 * true_lengths[:, None] * directions_yz
    d = p1 - p0
    t0 = np.zeros(len(true_lengths), dtype=np.float64)
    t1 = np.ones(len(true_lengths), dtype=np.float64)
    valid = np.ones(len(true_lengths), dtype=bool)

    for axis in range(2):
        parallel = np.abs(d[:, axis]) < 1e-12
        valid &= ~(parallel & ((p0[:, axis] < bbox_min[axis]) | (p0[:, axis] > bbox_max[axis])))
        non_parallel = ~parallel
        inv_d = np.zeros(len(true_lengths), dtype=np.float64)
        inv_d[non_parallel] = 1.0 / d[non_parallel, axis]
        ta = (bbox_min[axis] - p0[:, axis]) * inv_d
        tb = (bbox_max[axis] - p0[:, axis]) * inv_d
        t_enter = np.minimum(ta, tb)
        t_exit = np.maximum(ta, tb)
        t0 = np.where(non_parallel, np.maximum(t0, t_enter), t0)
        t1 = np.where(non_parallel, np.minimum(t1, t_exit), t1)

    valid &= t1 > t0
    visible_lengths = np.where(valid, (t1 - t0) * true_lengths, 0.0)
    p0_inside = np.all((p0 >= bbox_min) & (p0 <= bbox_max), axis=1)
    p1_inside = np.all((p1 >= bbox_min) & (p1 <= bbox_max), axis=1)
    classes = np.full(len(true_lengths), -1, dtype=np.int32)
    classes[valid & p0_inside & p1_inside] = 0
    classes[valid & (p0_inside ^ p1_inside)] = 1
    classes[valid & ~(p0_inside | p1_inside)] = 2
    return visible_lengths, classes


def clip_segments_to_polygon_loop(
    centers_yz: np.ndarray,
    directions_yz: np.ndarray,
    true_lengths: np.ndarray,
    polygon_yz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    visible_lengths = np.zeros(len(true_lengths), dtype=np.float64)
    classes = np.full(len(true_lengths), -1, dtype=np.int32)
    p0 = centers_yz - 0.5 * true_lengths[:, None] * directions_yz
    p1 = centers_yz + 0.5 * true_lengths[:, None] * directions_yz
    for idx in range(len(true_lengths)):
        visible_length, cls = clip_segment_to_polygon(p0[idx], p1[idx], polygon_yz)
        visible_lengths[idx] = visible_length
        classes[idx] = cls
    return visible_lengths, classes


def signed_polygon_area(polygon_yz: np.ndarray) -> float:
    y = polygon_yz[:, 0]
    z = polygon_yz[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


def clip_segments_to_convex_polygon_vectorized(
    centers_yz: np.ndarray,
    directions_yz: np.ndarray,
    true_lengths: np.ndarray,
    polygon_yz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0 = centers_yz - 0.5 * true_lengths[:, None] * directions_yz
    p1 = centers_yz + 0.5 * true_lengths[:, None] * directions_yz
    d = p1 - p0
    t0 = np.zeros(len(true_lengths), dtype=np.float64)
    t1 = np.ones(len(true_lengths), dtype=np.float64)
    valid = np.ones(len(true_lengths), dtype=bool)
    orientation = 1.0 if signed_polygon_area(polygon_yz) >= 0.0 else -1.0

    for idx in range(len(polygon_yz)):
        v0 = polygon_yz[idx]
        v1 = polygon_yz[(idx + 1) % len(polygon_yz)]
        edge = v1 - v0
        a = orientation * (edge[0] * (p0[:, 1] - v0[1]) - edge[1] * (p0[:, 0] - v0[0]))
        b = orientation * (edge[0] * d[:, 1] - edge[1] * d[:, 0])
        parallel = np.abs(b) < 1e-12
        valid &= ~(parallel & (a < 0.0))

        non_parallel = ~parallel
        t_cross = np.zeros(len(true_lengths), dtype=np.float64)
        t_cross[non_parallel] = -a[non_parallel] / b[non_parallel]
        entering = non_parallel & (b > 0.0)
        exiting = non_parallel & (b < 0.0)
        t0 = np.where(entering, np.maximum(t0, t_cross), t0)
        t1 = np.where(exiting, np.minimum(t1, t_cross), t1)

    valid &= t1 > t0
    visible_lengths = np.where(valid, (t1 - t0) * true_lengths, 0.0)
    p0_inside = np.all(
        [
            orientation
            * (
                (polygon_yz[(idx + 1) % len(polygon_yz), 0] - polygon_yz[idx, 0]) * (p0[:, 1] - polygon_yz[idx, 1])
                - (polygon_yz[(idx + 1) % len(polygon_yz), 1] - polygon_yz[idx, 1]) * (p0[:, 0] - polygon_yz[idx, 0])
            )
            >= -1e-10
            for idx in range(len(polygon_yz))
        ],
        axis=0,
    )
    p1_inside = np.all(
        [
            orientation
            * (
                (polygon_yz[(idx + 1) % len(polygon_yz), 0] - polygon_yz[idx, 0]) * (p1[:, 1] - polygon_yz[idx, 1])
                - (polygon_yz[(idx + 1) % len(polygon_yz), 1] - polygon_yz[idx, 1]) * (p1[:, 0] - polygon_yz[idx, 0])
            )
            >= -1e-10
            for idx in range(len(polygon_yz))
        ],
        axis=0,
    )
    classes = np.full(len(true_lengths), -1, dtype=np.int32)
    classes[valid & p0_inside & p1_inside] = 0
    classes[valid & (p0_inside ^ p1_inside)] = 1
    classes[valid & ~(p0_inside | p1_inside)] = 2
    return visible_lengths, classes


def sample_size_biased_radius(kr: float, rmin: float, rmax: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # Original f_R(r) is proportional to r^-(kr+1); intersected g_R(r) is proportional to r^-kr.
    exponent = 1.0 - kr
    u = rng.uniform(0.0, 1.0, size=size)
    if abs(exponent) < 1e-12:
        return rmin * np.exp(u * np.log(rmax / rmin))
    return (u * (rmax**exponent - rmin**exponent) + rmin**exponent) ** (1.0 / exponent)


def sample_true_chords(radii: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    offsets = rng.uniform(0.0, radii)
    return 2.0 * np.sqrt(np.maximum(radii * radii - offsets * offsets, 0.0))


def simulate_window_samples(
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    window_mode: str = "polygon",
    direction_mode: str = "empirical_trace",
    set_id: int = 0,
    site: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate accepted visible traces for a provided radius sample."""
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    w_bbox = bbox_max[0] - bbox_min[0]
    h_bbox = bbox_max[1] - bbox_min[1]
    true_lengths = sample_true_chords(radii, rng)
    n_samples = len(radii)

    if direction_mode == "orientation_conditioned":
        dir_pool = orientation_conditioned_trace_directions_yz(set_id, site, n_samples * 3, rng)
        direction_idx = rng.integers(0, len(dir_pool), size=n_samples)
        directions = dir_pool[direction_idx]
    else:
        direction_idx = rng.integers(0, len(directions_yz), size=n_samples)
        directions = directions_yz[direction_idx]

    proposal_areas = (w_bbox + true_lengths * np.abs(directions[:, 0])) * (h_bbox + true_lengths * np.abs(directions[:, 1]))
    expand = 0.5 * true_lengths[:, None]
    centers = rng.uniform(bbox_min - expand, bbox_max + expand)
    if window_mode == "bbox":
        visible_lengths, classes = clip_segments_to_bbox_vectorized(centers, directions, true_lengths, bbox_min, bbox_max)
    elif window_mode == "polygon":
        visible_lengths, classes = clip_segments_to_convex_polygon_vectorized(centers, directions, true_lengths, polygon_yz)
    else:
        raise ValueError(f"Unsupported window_mode: {window_mode}")
    accepted = (classes >= 0) & (visible_lengths > 0.0)
    return visible_lengths[accepted], classes[accepted], proposal_areas[accepted], radii[accepted]


def simulate_window_observations(
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    directions_yz: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    window_mode: str = "polygon",
    direction_mode: str = "empirical_trace",
    set_id: int = 0,
    site: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate window-observed trace lengths, censoring classes, and their center proposal areas.

    direction_mode:
        "empirical_trace"       — resample from observed directions_yz (default)
        "orientation_conditioned" — generate directions from Fisher(mean_pole, kappa)
                                    using SITE_FISHER_PARAMS[site][set_id]
    """
    radii = sample_size_biased_radius(kr, rmin, rmax, n_samples, rng)
    visible_lengths, classes, proposal_areas, _ = simulate_window_samples(
        polygon_yz,
        directions_yz,
        radii,
        rng,
        window_mode=window_mode,
        direction_mode=direction_mode,
        set_id=set_id,
        site=site,
    )
    return visible_lengths, classes, proposal_areas


def make_length_edges(lengths: np.ndarray, lmin_fit: float, bin_count: int, mode: str) -> np.ndarray:
    upper = max(float(np.max(lengths)) if len(lengths) else lmin_fit * 1.01, lmin_fit * 1.01)
    if mode == "log":
        return np.geomspace(lmin_fit, upper * 1.000001, bin_count + 1)
    return np.linspace(lmin_fit, upper * 1.000001, bin_count + 1)


def binned_counts(
    lengths: np.ndarray,
    classes: np.ndarray,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    table = np.zeros((len(edges) - 1, 3), dtype=np.float64)
    mask = (lengths >= edges[0]) & (classes >= 0) & (classes <= 2)
    bin_idx = np.searchsorted(edges, lengths[mask], side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    
    if weights is None:
        for b, c in zip(bin_idx, classes[mask]):
            table[int(b), int(c)] += 1.0
    else:
        w_mask = weights[mask]
        for b, c, w in zip(bin_idx, classes[mask], w_mask):
            table[int(b), int(c)] += float(w)
    return table


def probability_table(
    lengths: np.ndarray,
    classes: np.ndarray,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    counts = binned_counts(lengths, classes, edges, weights=weights)
    total = float(np.sum(counts))
    if total == 0.0:
        return np.full_like(counts, 1.0 / counts.size), 0.0
    return counts / total, total


def marginal_length_probability(joint_prob: np.ndarray) -> np.ndarray:
    return np.sum(joint_prob, axis=1)


def marginal_class_probability(joint_prob: np.ndarray) -> np.ndarray:
    return np.sum(joint_prob, axis=0)


def loglik_from_tables(observed_counts: np.ndarray, model_prob: np.ndarray) -> float:
    return float(np.sum(observed_counts * np.log(np.clip(model_prob, 1e-12, 1.0))))


def loglik_from_marginal_counts(observed_counts: np.ndarray, model_prob: np.ndarray) -> float:
    return float(np.sum(observed_counts * np.log(np.clip(model_prob, 1e-12, 1.0))))


def score_candidate(
    loglik_joint: float,
    loglik_length: float,
    loglik_class: float,
    likelihood_component: str,
    class_likelihood_weight: float,
) -> float:
    if likelihood_component == "length_only":
        return loglik_length
    if likelihood_component == "class_only":
        return loglik_class
    if abs(class_likelihood_weight - 1.0) > 1e-12:
        return loglik_length + class_likelihood_weight * loglik_class
    return loglik_joint


def summarize_profile(profile_rows: Sequence[dict], kr_min: float, kr_max: float) -> dict:
    loglik = np.asarray([float(row["loglik"]) for row in profile_rows], dtype=np.float64)
    kr = np.asarray([float(row["kr_window_mc"]) for row in profile_rows], dtype=np.float64)
    max_loglik = float(np.max(loglik))
    inside = kr[(max_loglik - loglik) <= 2.0]
    width = float(np.max(inside) - np.min(inside)) if len(inside) else float("nan")
    return {
        "max_loglik": max_loglik,
        "profile_width_delta2": width,
        "weak_identifiability_flag": bool(np.isfinite(width) and width > 0.5 * (kr_max - kr_min)),
    }


def fraction_by_class(classes: np.ndarray) -> tuple[float, float, float]:
    if len(classes) == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(np.mean(classes == cls)) for cls in (0, 1, 2))


def determine_status(
    set_id: int,
    n_model_accepted: int,
    q90_ratio: float,
    q95_ratio: float,
    class_l1: float,
    weak_identifiability: bool,
    window_mode: str,
) -> tuple[str, bool, str, List[str]]:
    warnings = [WINDOW_WARNING_POLYGON if window_mode == "polygon" else WINDOW_WARNING_BBOX]
    if n_model_accepted < 5000:
        return "low_mc_acceptance", True, "MC accepted samples < 5000", warnings
    if weak_identifiability:
        return "weak_identifiability", True, "profile likelihood delta<=2 interval too wide", warnings
    if np.isfinite(q90_ratio) and (q90_ratio < 1.0 / 3.0 or q90_ratio > 3.0):
        return "posterior_predictive_failed", True, "q90_model/q90_observed outside [1/3, 3]", warnings
    if np.isfinite(q95_ratio) and (q95_ratio < 1.0 / 3.0 or q95_ratio > 3.0):
        return "posterior_predictive_failed", True, "q95_model/q95_observed outside [1/3, 3]", warnings
    if class_l1 > 0.30:
        return "class_fraction_mismatch", True, "class_fraction_l1_error > 0.30", warnings
    if set_id == 4:
        warnings.append(SET4_WARNING)
        return "provisional_ok", False, "Set 4 capped at provisional_ok", warnings
    return "ok", False, "", warnings


def center_weighting_status(center_weighting: str) -> str:
    if center_weighting == "proposal_area":
        return "preferred_for_window_mc"
    return "legacy_diagnostic"


def determine_bias_source(
    kr_true: Optional[float],
    kr_hat_length_only: float,
    kr_hat_class_only: float,
    kr_hat_joint: float,
) -> str:
    if kr_true is None or not np.isfinite(kr_true):
        return "unknown"
    err_length = abs(kr_hat_length_only - kr_true)
    err_class = abs(kr_hat_class_only - kr_true)
    err_joint = abs(kr_hat_joint - kr_true)
    if err_length <= _RECOVERY_MODERATE_THRESHOLD and err_joint > _RECOVERY_MODERATE_THRESHOLD:
        return "class_likelihood_contaminates_kr"
    if kr_hat_class_only < kr_true - _RECOVERY_MODERATE_THRESHOLD and kr_hat_joint < kr_true - _RECOVERY_MODERATE_THRESHOLD:
        return "censoring_class_process_bias"
    if kr_hat_length_only < kr_true - _RECOVERY_MODERATE_THRESHOLD:
        return "length_process_bias"
    if err_length > _RECOVERY_MODERATE_THRESHOLD and err_class > _RECOVERY_MODERATE_THRESHOLD:
        return "visibility_forward_model_bias"
    return "mixed_or_reduced_bias"


def build_component_row(candidate: dict, kr_true: Optional[float]) -> dict:
    recovery_status = determine_recovery_status(float(candidate["kr"]), kr_true)
    adoption_status = determine_adoption_status(str(candidate["fit_status"]), recovery_status)
    return {
        "kr_hat": float(candidate["kr"]),
        "kr_abs_error": abs(float(candidate["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "q90_ratio": float(candidate["q90_ratio"]),
        "q95_ratio": float(candidate["q95_ratio"]),
        "class_l1": float(candidate["class_l1"]),
        "fit_status": str(candidate["fit_status"]),
        "recovery_status": recovery_status,
        "adoption_status": adoption_status,
    }


def fit_set_lmin(
    set_id: int,
    set_rows: Sequence[dict],
    polygon_yz: np.ndarray,
    kr_grid: np.ndarray,
    rmin: float,
    rmax: float,
    lmin_fit: float,
    mc_samples_per_grid: int,
    bin_count: int,
    bin_mode: str,
    window_mode: str,
    direction_mode: str = "empirical_trace",
    site: str = "",
    kr_true: Optional[float] = None,
    center_weighting: str = "unweighted",
    likelihood_component: str = "joint",
    class_likelihood_weight: float = 1.0,
    oracle_radius_mode: str = "none",
    run_bootstrap: bool = False,
    n_bootstrap: int = 100,
) -> tuple[dict, List[dict], List[dict], List[dict]]:
    lengths_all = np.asarray([float(row["observed_length_m"]) for row in set_rows], dtype=np.float64)
    classes_all = np.asarray([int(row["censoring_class"]) for row in set_rows], dtype=np.int32)
    used = lengths_all >= lmin_fit
    obs_lengths = lengths_all[used]
    obs_classes = classes_all[used]
    edges = make_length_edges(obs_lengths, lmin_fit, bin_count, bin_mode)
    obs_counts = binned_counts(obs_lengths, obs_classes, edges)
    obs_length_counts = np.sum(obs_counts, axis=1)
    obs_class_counts = np.sum(obs_counts, axis=0)
    directions = empirical_trace_directions_yz(set_rows)
    oracle_radii_all = np.asarray(
        [float(row.get("radius_m", float("nan"))) for row in set_rows if np.isfinite(float(row.get("radius_m", float("nan"))))],
        dtype=np.float64,
    )

    profile_rows: List[dict] = []
    pp_rows: List[dict] = []
    rng_base = 91000 + set_id * 1000 + int(round(lmin_fit * 1000.0))
    best = None
    candidate_rows: List[dict] = []
    
    # Store precomputed likelihood summaries for bootstrap and decomposition
    grid_prob_summaries = []
    
    for idx, kr in enumerate(kr_grid):
        rng = np.random.default_rng(rng_base + idx)
        if oracle_radius_mode == "observed_trace_radii":
            if len(oracle_radii_all) == 0:
                raise ValueError("oracle_radius_mode=observed_trace_radii requires radius_m in observed trace rows.")
            oracle_radii = rng.choice(oracle_radii_all, size=mc_samples_per_grid, replace=True)
            sim_lengths, sim_classes, sim_areas, sim_radii = simulate_window_samples(
                polygon_yz,
                directions,
                oracle_radii,
                rng,
                window_mode=window_mode,
                direction_mode=direction_mode,
                set_id=set_id,
                site=site,
            )
        else:
            sim_radii = None
            sim_lengths, sim_classes, sim_areas = simulate_window_observations(
                float(kr), rmin, rmax, polygon_yz, directions, mc_samples_per_grid, rng,
                window_mode=window_mode, direction_mode=direction_mode,
                set_id=set_id, site=site,
            )
        sim_used = sim_lengths >= lmin_fit
        sim_lengths_used = sim_lengths[sim_used]
        sim_classes_used = sim_classes[sim_used]
        sim_areas_used = sim_areas[sim_used]
        sim_radii_used = sim_radii[sim_used] if sim_radii is not None else None
        
        # Determine weighting
        weights = sim_areas_used if center_weighting == "proposal_area" else None
        
        prob_joint, n_model_used = probability_table(sim_lengths_used, sim_classes_used, edges, weights=weights)
        prob_length = marginal_length_probability(prob_joint)
        prob_class = marginal_class_probability(prob_joint)

        loglik_joint = loglik_from_tables(obs_counts, prob_joint)
        loglik_length = loglik_from_marginal_counts(obs_length_counts, prob_length)
        loglik_class = loglik_from_marginal_counts(obs_class_counts, prob_class)
        loglik = score_candidate(
            loglik_joint,
            loglik_length,
            loglik_class,
            likelihood_component,
            class_likelihood_weight,
        )

        obs_unc, obs_one, obs_two = fraction_by_class(obs_classes)
        mod_unc, mod_one, mod_two = fraction_by_class(sim_classes_used)
        class_l1 = float(abs(obs_unc - mod_unc) + abs(obs_one - mod_one) + abs(obs_two - mod_two))
        q90_obs = float(np.percentile(obs_lengths, 90)) if len(obs_lengths) else float("nan")
        q95_obs = float(np.percentile(obs_lengths, 95)) if len(obs_lengths) else float("nan")
        q90_mod = float(np.percentile(sim_lengths_used, 90)) if len(sim_lengths_used) else float("nan")
        q95_mod = float(np.percentile(sim_lengths_used, 95)) if len(sim_lengths_used) else float("nan")
        q90_ratio = q90_mod / q90_obs if q90_obs > 0.0 else float("nan")
        q95_ratio = q95_mod / q95_obs if q95_obs > 0.0 else float("nan")

        candidate_rows.append(
            {
                "kr": float(kr),
                "loglik_joint": loglik_joint,
                "loglik_length_only": loglik_length,
                "loglik_class_only": loglik_class,
                "loglik": loglik,
                "n_model_used": n_model_used,
                "sim_lengths": sim_lengths_used,
                "sim_classes": sim_classes_used,
                "sim_areas": sim_areas_used,
                "sim_radii": sim_radii_used,
                "q90_ratio": q90_ratio,
                "q95_ratio": q95_ratio,
                "class_l1": class_l1,
                "obs_unc": obs_unc,
                "obs_one": obs_one,
                "obs_two": obs_two,
                "mod_unc": mod_unc,
                "mod_one": mod_one,
                "mod_two": mod_two,
                "prob_joint": prob_joint,
                "prob_length": prob_length,
                "prob_class": prob_class,
            }
        )
        grid_prob_summaries.append(
            {
                "joint": prob_joint,
                "length_only": prob_length,
                "class_only": prob_class,
            }
        )

        profile_rows.append(
            {
                "set_id": set_id,
                "lmin_fit": float(lmin_fit),
                "kr_window_mc": float(kr),
                "loglik": loglik,
                "loglik_joint": loglik_joint,
                "loglik_length_only": loglik_length,
                "loglik_class_only": loglik_class,
                "mc_accepted_used": n_model_used,
            }
        )
        if best is None or loglik > best["loglik"]:
            best = dict(candidate_rows[-1])

    assert best is not None
    profile_summary = summarize_profile(profile_rows, float(kr_grid[0]), float(kr_grid[-1]))
    for row in profile_rows:
        row["max_loglik"] = profile_summary["max_loglik"]
        row["delta_loglik"] = profile_summary["max_loglik"] - float(row["loglik"])

    # Compute proposal area metrics on the best model's simulation
    best_areas = best["sim_areas"]
    if len(best_areas) > 0:
        mean_area = float(np.mean(best_areas))
        p50_area = float(np.percentile(best_areas, 50))
        p90_area = float(np.percentile(best_areas, 90))
        # Correlation between simulated visible length and proposal area
        if len(best_areas) > 1 and np.std(best["sim_lengths"]) > 1e-8 and np.std(best_areas) > 1e-8:
            area_len_corr = float(np.corrcoef(best["sim_lengths"], best_areas)[0, 1])
        else:
            area_len_corr = 0.0
    else:
        mean_area, p50_area, p90_area, area_len_corr = float("nan"), float("nan"), float("nan"), float("nan")

    obs_unc = float(best["obs_unc"])
    obs_one = float(best["obs_one"])
    obs_two = float(best["obs_two"])
    mod_unc = float(best["mod_unc"])
    mod_one = float(best["mod_one"])
    mod_two = float(best["mod_two"])
    class_l1 = float(abs(obs_unc - mod_unc) + abs(obs_one - mod_one) + abs(obs_two - mod_two))
    q50_obs = float(np.percentile(obs_lengths, 50)) if len(obs_lengths) else float("nan")
    q90_obs = float(np.percentile(obs_lengths, 90)) if len(obs_lengths) else float("nan")
    q95_obs = float(np.percentile(obs_lengths, 95)) if len(obs_lengths) else float("nan")
    q50_mod = float(np.percentile(best["sim_lengths"], 50)) if len(best["sim_lengths"]) else float("nan")
    q90_mod = float(np.percentile(best["sim_lengths"], 90)) if len(best["sim_lengths"]) else float("nan")
    q95_mod = float(np.percentile(best["sim_lengths"], 95)) if len(best["sim_lengths"]) else float("nan")
    q90_ratio = q90_mod / q90_obs if q90_obs > 0.0 else float("nan")
    q95_ratio = q95_mod / q95_obs if q95_obs > 0.0 else float("nan")
    status, rejected, reason, warnings = determine_status(
        set_id,
        int(best["n_model_used"]),
        q90_ratio,
        q95_ratio,
        class_l1,
        profile_summary["weak_identifiability_flag"],
        window_mode,
    )
    class_diffs = {
        "uncensored": mod_unc - obs_unc,
        "one_end": mod_one - obs_one,
        "two_end": mod_two - obs_two,
    }
    dominant_key = max(class_diffs, key=lambda key: abs(class_diffs[key]))
    if abs(class_diffs[dominant_key]) < 0.10:
        dominant_class_error = "mixed"
    elif dominant_key == "uncensored":
        dominant_class_error = "uncensored_overpredicted" if class_diffs[dominant_key] > 0.0 else "uncensored_underpredicted"
    elif dominant_key == "one_end":
        dominant_class_error = "one_end_overpredicted" if class_diffs[dominant_key] > 0.0 else "one_end_underpredicted"
    else:
        dominant_class_error = "two_end_overpredicted" if class_diffs[dominant_key] > 0.0 else "two_end_underpredicted"

    for cls, label in [(0, "uncensored"), (1, "one_end_censored"), (2, "two_end_censored")]:
        pp_rows.append(
            {
                "set_id": set_id,
                "lmin_fit": float(lmin_fit),
                "kr_window_mc": best["kr"],
                "class": label,
                "observed_fraction": [obs_unc, obs_one, obs_two][cls],
                "model_fraction": [mod_unc, mod_one, mod_two][cls],
            }
        )

    n_unc = int(np.sum(obs_classes == 0))
    n_one = int(np.sum(obs_classes == 1))
    n_two = int(np.sum(obs_classes == 2))
    
    # Bootstrap CI calculation
    kr_boot_mean, kr_boot_std, kr_ci_low, kr_ci_high = float("nan"), float("nan"), float("nan"), float("nan")
    boundary_fraction = float("nan")
    recovery_ci_status = "unknown"
    
    if run_bootstrap and len(obs_lengths) > 0:
        boot_krs = []
        boot_rng = np.random.default_rng(12345 + set_id)
        for _ in range(n_bootstrap):
            idx_resampled = boot_rng.choice(len(obs_lengths), size=len(obs_lengths), replace=True)
            boot_lengths = obs_lengths[idx_resampled]
            boot_classes = obs_classes[idx_resampled]
            boot_counts = binned_counts(boot_lengths, boot_classes, edges)
            
            boot_best_lik = -np.inf
            boot_best_kr = kr_grid[0]
            boot_length_counts = np.sum(boot_counts, axis=1)
            boot_class_counts = np.sum(boot_counts, axis=0)
            for g_idx, g_probs in enumerate(grid_prob_summaries):
                lik = score_candidate(
                    loglik_from_tables(boot_counts, g_probs["joint"]),
                    loglik_from_marginal_counts(boot_length_counts, g_probs["length_only"]),
                    loglik_from_marginal_counts(boot_class_counts, g_probs["class_only"]),
                    likelihood_component,
                    class_likelihood_weight,
                )
                if lik > boot_best_lik:
                    boot_best_lik = lik
                    boot_best_kr = kr_grid[g_idx]
            boot_krs.append(boot_best_kr)
        
        boot_krs = np.array(boot_krs, dtype=np.float64)
        kr_boot_mean = float(np.mean(boot_krs))
        kr_boot_std = float(np.std(boot_krs))
        kr_ci_low = float(np.percentile(boot_krs, 2.5))
        kr_ci_high = float(np.percentile(boot_krs, 97.5))
        boundary_fraction = float(np.mean((boot_krs <= kr_grid[0] + 1e-5) | (boot_krs >= kr_grid[-1] - 1e-5)))
        
        if kr_true is not None and np.isfinite(kr_true):
            if kr_ci_low <= kr_true <= kr_ci_high:
                recovery_ci_status = "recovery_uncertain"
            else:
                recovery_ci_status = "systematic_bias"
    
    recovery_status = determine_recovery_status(float(best["kr"]), kr_true)
    adoption_status = determine_adoption_status(status, recovery_status)
    best_joint = max(candidate_rows, key=lambda row: row["loglik_joint"])
    best_length = max(candidate_rows, key=lambda row: row["loglik_length_only"])
    best_class = max(candidate_rows, key=lambda row: row["loglik_class_only"])
    fit_row = {
        "set_id": set_id,
        "model": f"window_mc_{window_mode}_v4_1",
        "window_mode": window_mode,
        "direction_mode": direction_mode,
        "center_weighting": center_weighting,
        "center_weighting_status": center_weighting_status(center_weighting),
        "likelihood_component": likelihood_component,
        "class_likelihood_weight": float(class_likelihood_weight),
        "main_class_likelihood_weight": float(class_likelihood_weight),
        "oracle_radius_mode": oracle_radius_mode,
        "weighted_probability_used": bool(center_weighting == "proposal_area"),
        "lmin_fit": float(lmin_fit),
        "rmin": float(rmin),
        "rmax": float(rmax),
        "kr_window_mc_hat": best["kr"],
        "kr_true": kr_true if kr_true is not None else float("nan"),
        "loglik": best["loglik"],
        "n_total": len(set_rows),
        "n_used": len(obs_lengths),
        "n_uncensored_used": n_unc,
        "n_one_end_censored_used": n_one,
        "n_two_end_censored_used": n_two,
        "censoring_ratio_used": (n_one + n_two) / len(obs_lengths) if len(obs_lengths) else float("nan"),
        "two_end_censoring_ratio_used": n_two / len(obs_lengths) if len(obs_lengths) else float("nan"),
        "mc_accepted_count": int(best["n_model_used"]),
        "q50_observed": q50_obs,
        "q90_observed": q90_obs,
        "q95_observed": q95_obs,
        "q50_model": q50_mod,
        "q90_model": q90_mod,
        "q95_model": q95_mod,
        "q90_ratio_model_observed": q90_ratio,
        "q95_ratio_model_observed": q95_ratio,
        "observed_uncensored_fraction": obs_unc,
        "model_uncensored_fraction": mod_unc,
        "observed_one_end_fraction": obs_one,
        "model_one_end_fraction": mod_one,
        "observed_two_end_fraction": obs_two,
        "model_two_end_fraction": mod_two,
        "class_fraction_l1_error": class_l1,
        "dominant_class_error": dominant_class_error,
        "profile_width_delta2": profile_summary["profile_width_delta2"],
        "weak_identifiability_flag": profile_summary["weak_identifiability_flag"],
        "mean_proposal_area": mean_area,
        "p50_proposal_area": p50_area,
        "p90_proposal_area": p90_area,
        "proposal_area_length_corr": area_len_corr,
        "kr_boot_mean": kr_boot_mean,
        "kr_boot_std": kr_boot_std,
        "kr_ci_low": kr_ci_low,
        "kr_ci_high": kr_ci_high,
        "bootstrap_boundary_fraction": boundary_fraction,
        "recovery_ci_status": recovery_ci_status,
        "fit_status": status,           # posterior predictive criterion
        "final_status": status,         # backward-compat alias
        "recovery_status": recovery_status,  # kr_true recovery criterion
        "adoption_status": adoption_status,  # combined decision
        "model_rejected": bool(rejected),
        "rejection_reason": reason,
        "kr_hat_length_only": float(best_length["kr"]),
        "kr_hat_class_only": float(best_class["kr"]),
        "kr_hat_joint": float(best_joint["kr"]),
        "kr_abs_error_length_only": abs(float(best_length["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "kr_abs_error_class_only": abs(float(best_class["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "kr_abs_error_joint": abs(float(best_joint["kr"]) - kr_true) if kr_true is not None and np.isfinite(kr_true) else float("nan"),
        "q90_ratio_length_only": float(best_length["q90_ratio"]),
        "q95_ratio_length_only": float(best_length["q95_ratio"]),
        "class_l1_joint": float(best_joint["class_l1"]),
        "fit_status_joint": determine_status(
            set_id,
            int(best_joint["n_model_used"]),
            float(best_joint["q90_ratio"]),
            float(best_joint["q95_ratio"]),
            float(best_joint["class_l1"]),
            profile_summary["weak_identifiability_flag"],
            window_mode,
        )[0],
        "recovery_status_joint": determine_recovery_status(float(best_joint["kr"]), kr_true),
        "bias_source": determine_bias_source(
            kr_true,
            float(best_length["kr"]),
            float(best_class["kr"]),
            float(best_joint["kr"]),
        ),
        "warning": "; ".join(warnings),
    }
    return fit_row, profile_rows, pp_rows, candidate_rows



def build_bbox_polygon_comparison_row(bbox_row: dict, polygon_row: dict) -> dict:
    return {
        "set_id": polygon_row["set_id"],
        "lmin_fit": polygon_row["lmin_fit"],
        "kr_bbox": bbox_row["kr_window_mc_hat"],
        "kr_polygon": polygon_row["kr_window_mc_hat"],
        "delta_kr": polygon_row["kr_window_mc_hat"] - bbox_row["kr_window_mc_hat"],
        "class_l1_bbox": bbox_row["class_fraction_l1_error"],
        "class_l1_polygon": polygon_row["class_fraction_l1_error"],
        "q90_ratio_bbox": bbox_row["q90_ratio_model_observed"],
        "q90_ratio_polygon": polygon_row["q90_ratio_model_observed"],
        "q95_ratio_bbox": bbox_row["q95_ratio_model_observed"],
        "q95_ratio_polygon": polygon_row["q95_ratio_model_observed"],
        "status_bbox": bbox_row["final_status"],
        "status_polygon": polygon_row["final_status"],
    }


def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def empirical_survival_curve(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    if len(lengths) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    grid = np.unique(np.sort(lengths))
    survival = np.asarray([np.mean(lengths > value) for value in grid], dtype=np.float64)
    return grid, survival


def kaplan_meier_survival_curve(lengths: np.ndarray, censoring_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    censoring_classes = np.atleast_1d(np.asarray(censoring_classes, dtype=np.int32))
    if len(lengths) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    grid = np.unique(np.sort(lengths))
    survival = np.empty(len(grid), dtype=np.float64)
    current_survival = 1.0
    for idx, value in enumerate(grid):
        at_risk = lengths >= value - 1e-12
        exact = np.abs(lengths - value) <= 1e-12
        n_i = int(np.sum(at_risk))
        d_i = int(np.sum(exact & (censoring_classes == 0)))
        if n_i > 0 and d_i > 0:
            current_survival *= (1.0 - d_i / n_i)
        survival[idx] = current_survival
    return grid, survival


def main() -> None:
    parser = argparse.ArgumentParser(description="Window-aware Monte Carlo likelihood for radius power-law candidates.")
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--target-set", nargs="+", type=int)
    parser.add_argument("--dfn-model", default="", help="DFN model/site label, e.g. forsmark or laxemar.")
    parser.add_argument("--rmin", type=float, default=0.5, help="Estimation rmin")
    parser.add_argument("--rmax", type=float, default=250.0, help="Estimation rmax")
    parser.add_argument(
        "--set-rmin-mode",
        choices=["global", "effective_generation", "table_r0"],
        default="effective_generation",
        help="Per-set lower-bound mode for radius support: global, effective_generation, or table_r0.",
    )
    parser.add_argument("--generation-rmin", type=float, default=0.5, help="Radius lower bound used during DFN generation.")
    parser.add_argument("--generation-rmax", type=float, default=250.0, help="Radius upper bound used during DFN generation.")
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m", help="P32 population label, e.g. P32_r_ge_0p5m.")
    parser.add_argument("--kr-min", type=float, default=1.5)
    parser.add_argument("--kr-max", type=float, default=5.5)
    parser.add_argument("--lmin-fit-values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75])
    parser.add_argument("--allow-rmin-mismatch", action="store_true", help="Allow estimation rmin to differ from DFN generation rmin.")
    parser.add_argument("--profile-grid-size", type=int, default=81)
    parser.add_argument("--mc-samples-per-grid", type=int, default=50000)
    parser.add_argument("--length-bin-count", type=int, default=40)
    parser.add_argument("--length-bin-mode", choices=["log", "linear"], default="log")
    parser.add_argument("--window-mode", choices=["polygon", "bbox"], default="polygon")
    parser.add_argument(
        "--direction-mode",
        choices=["empirical_trace", "orientation_conditioned"],
        default="empirical_trace",
        help="Trace direction generation mode: empirical_trace (default) or orientation_conditioned.",
    )
    parser.add_argument(
        "--site",
        choices=["forsmark", "laxemar"],
        default="",
        help="Site name for Fisher param lookup (required when --direction-mode=orientation_conditioned).",
    )
    parser.add_argument(
        "--kr-true-map",
        nargs="+",
        metavar="SET_ID:KR_TRUE",
        help="Known true kr values per set (e.g. 1:2.88 2:3.02). Used to compute recovery_status.",
    )
    parser.add_argument(
        "--center-weighting",
        choices=["unweighted", "proposal_area"],
        default="unweighted",
        help="Weighting mode for MC accepted samples: unweighted (default) or proposal_area.",
    )
    parser.add_argument(
        "--likelihood-component",
        choices=["joint", "length_only", "class_only"],
        default="joint",
        help="Likelihood objective for kr selection: joint (default), length_only, or class_only.",
    )
    parser.add_argument(
        "--class-likelihood-weight",
        nargs="+",
        type=float,
        default=[1.0],
        help="Class likelihood weights for sensitivity analysis. The first value is used for the main fit if needed.",
    )
    parser.add_argument(
        "--oracle-radius-mode",
        choices=["none", "observed_trace_radii"],
        default="none",
        help="Diagnostic-only radius source override. observed_trace_radii resamples radius_m from observed traces instead of sampling by kr.",
    )
    parser.add_argument(
        "--run-bootstrap",
        action="store_true",
        help="Run bootstrap resampling on observed traces to estimate kr confidence intervals.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--export-predicted-survival",
        action="store_true",
        help="Export MC-predicted trace-length survival curves for the fitted kr_hat values.",
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")
    generation_rmin = float(args.generation_rmin) if args.generation_rmin is not None else float(args.rmin)
    generation_rmax = float(args.generation_rmax) if args.generation_rmax is not None else float(args.rmax)
    p32_label = args.p32_label or f"P32_r_ge_{str(args.rmin).replace('.', 'p')}m"

    # rmin consistency guard
    h5_generation_rmin = None
    trace_rmin_metadata = {}
    if args.trace_h5:
        try:
            trace_rmin_metadata = load_trace_rmin_metadata_from_h5(args.trace_h5)
            if trace_rmin_metadata.get("generation_rmin") is not None:
                h5_generation_rmin = float(trace_rmin_metadata["generation_rmin"])
        except Exception as e:
            print(f"[WARNING] Failed to read generation_rmin from trace HDF5: {e}")

    gen_rmin_to_check = generation_rmin
    if h5_generation_rmin is not None:
        gen_rmin_to_check = h5_generation_rmin

    rmin_consistency_status = "matched"
    diagnostic_only = False
    warning_msg = None

    if gen_rmin_to_check is not None:
        if abs(gen_rmin_to_check - args.rmin) > 1e-5:
            if not args.allow_rmin_mismatch:
                raise ValueError(
                    f"mismatch: DFN generation_rmin ({gen_rmin_to_check}) does not match "
                    f"estimation rmin ({args.rmin}). Use --allow-rmin-mismatch to bypass."
                )
            else:
                rmin_consistency_status = "mismatch_diagnostic_only"
                diagnostic_only = True
                warning_msg = "generation_rmin and estimator rmin are inconsistent; kr recovery should not be interpreted"
                print(f"[WARNING] {warning_msg}")

    # Build kr_true_map
    _site_to_kr_true: Dict[str, Dict[int, float]] = {
        "forsmark": {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92},
        "laxemar":  {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.60},
    }
    site = getattr(args, "site", "") or ""
    direction_mode = getattr(args, "direction_mode", "empirical_trace")
    kr_true_map: Dict[int, float] = {}
    # Populate from --site built-in table
    if site:
        kr_true_map = dict(_site_to_kr_true.get(site, {}))
    # Override / supplement from --kr-true-map CLI
    if getattr(args, "kr_true_map", None):
        for token in args.kr_true_map:
            try:
                sid_str, kt_str = token.split(":")
                kr_true_map[int(sid_str)] = float(kt_str)
            except ValueError:
                print(f"[WARNING] Cannot parse --kr-true-map token: {token}")

    if direction_mode == "orientation_conditioned" and not site:
        raise ValueError("--site {forsmark,laxemar} is required when --direction-mode=orientation_conditioned")

    rows, polygon_yz = load_trace_data_from_h5(args.trace_h5) if args.trace_h5 else load_trace_data_from_csv(args.trace_csv)
    if polygon_yz is None:
        raise ValueError("Window polygon is required for window MC. Use --trace-h5 with /meta/tunnel_poly_yz.")
    set_rmin_lookup = build_set_rmin_lookup(site, float(gen_rmin_to_check), trace_rmin_metadata)

    target_sets = set(args.target_set) if args.target_set else None
    grouped = group_rows_by_set(rows, target_sets)
    if not grouped:
        raise ValueError("No matching rows for target sets.")

    kr_grid = np.linspace(args.kr_min, args.kr_max, args.profile_grid_size, dtype=np.float64)
    fit_rows: List[dict] = []
    profile_rows: List[dict] = []
    pp_rows: List[dict] = []
    comparison_rows: List[dict] = []
    weighting_comp_rows: List[dict] = []
    decomposition_rows: List[dict] = []
    class_weight_rows: List[dict] = []
    survival_rows: List[dict] = []
    main_class_weight = float(args.class_likelihood_weight[0])
    
    for set_id, set_rows in grouped.items():
        for lmin_fit in args.lmin_fit_values:
            kr_true = kr_true_map.get(set_id)
            set_likelihood_rmin, set_effective_generation_rmin, set_table_r0, set_support_status = resolve_set_likelihood_rmin(
                set_id,
                args.set_rmin_mode,
                float(args.rmin),
                set_rmin_lookup,
            )
            fit_row, profile, pp, candidate_rows = fit_set_lmin(
                set_id,
                set_rows,
                polygon_yz,
                kr_grid,
                set_likelihood_rmin,
                args.rmax,
                float(lmin_fit),
                args.mc_samples_per_grid,
                args.length_bin_count,
                args.length_bin_mode,
                args.window_mode,
                direction_mode=direction_mode,
                site=site,
                kr_true=kr_true,
                center_weighting=args.center_weighting,
                likelihood_component=args.likelihood_component,
                class_likelihood_weight=main_class_weight,
                oracle_radius_mode=args.oracle_radius_mode,
                run_bootstrap=args.run_bootstrap,
                n_bootstrap=args.n_bootstrap,
            )
            fit_row["rmin_consistency_status"] = rmin_consistency_status
            fit_row["diagnostic_only"] = bool(diagnostic_only)
            if warning_msg:
                fit_row["warning"] = (fit_row["warning"] + "; " if fit_row["warning"] else "") + warning_msg
            metadata = {
                "dfn_model": args.dfn_model,
                "generation_rmin": generation_rmin,
                "generation_rmax": generation_rmax,
                "estimation_rmin": float(args.rmin),
                "estimation_rmax": float(args.rmax),
                "likelihood_rmin": float(set_likelihood_rmin),
                "likelihood_rmax": float(args.rmax),
                "set_likelihood_rmin": float(set_likelihood_rmin),
                "set_effective_generation_rmin": float(set_effective_generation_rmin),
                "set_table_r0": float(set_table_r0) if np.isfinite(set_table_r0) else float("nan"),
                "set_rmin_mode": args.set_rmin_mode,
                "rmin_support_status": set_support_status,
                "lmin_fit_values": " ".join(str(v) for v in args.lmin_fit_values),
                "p32_label": p32_label,
            }
            fit_row.update(metadata)
            fit_row["adoption_status"] = determine_adoption_status(
                str(fit_row["fit_status"]),
                str(fit_row["recovery_status"]),
                set_support_status,
            )
            for row in profile:
                row.update(metadata)
            for row in pp:
                row.update(metadata)
            fit_rows.append(fit_row)
            profile_rows.extend(profile)
            pp_rows.extend(pp)
            if args.export_predicted_survival:
                best_candidate = max(candidate_rows, key=lambda row: row["loglik"])
                sim_lengths_best = np.asarray(best_candidate["sim_lengths"], dtype=np.float64)
                sim_classes_best = np.asarray(best_candidate["sim_classes"], dtype=np.int32)
                visible_lengths, visible_survival = empirical_survival_curve(sim_lengths_best)
                km_lengths, km_survival = kaplan_meier_survival_curve(sim_lengths_best, sim_classes_best)
                true_radii = np.asarray(best_candidate["sim_radii"], dtype=np.float64)
                true_chords = 2.0 * true_radii
                chord_lengths, chord_survival = empirical_survival_curve(true_chords)

                for length_value, surv in zip(visible_lengths, visible_survival):
                    survival_rows.append(
                        {
                            "site": site or args.dfn_model,
                            "set_id": set_id,
                            "lmin_fit": float(lmin_fit),
                            "length": float(length_value),
                            "mc_survival": float(surv),
                            "survival_mode": "mc_observed_visible_survival",
                            "kr_hat": float(fit_row["kr_window_mc_hat"]),
                            "set_likelihood_rmin": float(set_likelihood_rmin),
                            "set_effective_generation_rmin": float(set_effective_generation_rmin),
                            "center_weighting": args.center_weighting,
                            "window_mode": args.window_mode,
                            "direction_mode": direction_mode,
                            "n_mc_samples": int(args.mc_samples_per_grid),
                            "length_bin_count": int(args.length_bin_count),
                            "notes": "predicted_survival_from_fitted_window_mc_visible_lengths",
                        }
                    )
                for length_value, surv in zip(km_lengths, km_survival):
                    survival_rows.append(
                        {
                            "site": site or args.dfn_model,
                            "set_id": set_id,
                            "lmin_fit": float(lmin_fit),
                            "length": float(length_value),
                            "mc_survival": float(surv),
                            "survival_mode": "mc_km_emulated_survival",
                            "kr_hat": float(fit_row["kr_window_mc_hat"]),
                            "set_likelihood_rmin": float(set_likelihood_rmin),
                            "set_effective_generation_rmin": float(set_effective_generation_rmin),
                            "center_weighting": args.center_weighting,
                            "window_mode": args.window_mode,
                            "direction_mode": direction_mode,
                            "n_mc_samples": int(args.mc_samples_per_grid),
                            "length_bin_count": int(args.length_bin_count),
                            "notes": "kaplan_meier_applied_to_simulated_observed_lengths_and_censoring_classes",
                        }
                    )
                for length_value, surv in zip(chord_lengths, chord_survival):
                    survival_rows.append(
                        {
                            "site": site or args.dfn_model,
                            "set_id": set_id,
                            "lmin_fit": float(lmin_fit),
                            "length": float(length_value),
                            "mc_survival": float(surv),
                            "survival_mode": "mc_true_chord_survival",
                            "kr_hat": float(fit_row["kr_window_mc_hat"]),
                            "set_likelihood_rmin": float(set_likelihood_rmin),
                            "set_effective_generation_rmin": float(set_effective_generation_rmin),
                            "center_weighting": args.center_weighting,
                            "window_mode": args.window_mode,
                            "direction_mode": direction_mode,
                            "n_mc_samples": int(args.mc_samples_per_grid),
                            "length_bin_count": int(args.length_bin_count),
                            "notes": "diagnostic_true_chord_survival_before_window_clipping",
                        }
                    )
            decomposition_rows.append(
                {
                    "site": site or args.dfn_model,
                    "set_id": set_id,
                    "lmin_fit": float(lmin_fit),
                    "center_weighting": args.center_weighting,
                    "center_weighting_status": center_weighting_status(args.center_weighting),
                    "direction_mode": direction_mode,
                    "kr_true": kr_true if kr_true is not None else float("nan"),
                    "kr_hat_length_only": fit_row["kr_hat_length_only"],
                    "kr_hat_class_only": fit_row["kr_hat_class_only"],
                    "kr_hat_joint": fit_row["kr_hat_joint"],
                    "kr_abs_error_length_only": fit_row["kr_abs_error_length_only"],
                    "kr_abs_error_class_only": fit_row["kr_abs_error_class_only"],
                    "kr_abs_error_joint": fit_row["kr_abs_error_joint"],
                    "q90_ratio_length_only": fit_row["q90_ratio_length_only"],
                    "q95_ratio_length_only": fit_row["q95_ratio_length_only"],
                    "class_l1_joint": fit_row["class_l1_joint"],
                    "fit_status_joint": fit_row["fit_status_joint"],
                    "recovery_status_joint": fit_row["recovery_status_joint"],
                    "bias_source": fit_row["bias_source"],
                }
            )
            for class_weight in args.class_likelihood_weight:
                best_weighted = max(
                    candidate_rows,
                    key=lambda row: row["loglik_length_only"] + float(class_weight) * row["loglik_class_only"],
                )
                best_weighted_status, _, _, _ = determine_status(
                    set_id,
                    int(best_weighted["n_model_used"]),
                    float(best_weighted["q90_ratio"]),
                    float(best_weighted["q95_ratio"]),
                    float(best_weighted["class_l1"]),
                    fit_row["weak_identifiability_flag"],
                    args.window_mode,
                )
                weight_summary = build_component_row(
                    {
                        **best_weighted,
                        "fit_status": best_weighted_status,
                    },
                    kr_true,
                )
                class_weight_rows.append(
                    {
                        "site": site or args.dfn_model,
                        "set_id": set_id,
                        "lmin_fit": float(lmin_fit),
                        "class_weight": float(class_weight),
                        "kr_true": kr_true if kr_true is not None else float("nan"),
                        "kr_hat": weight_summary["kr_hat"],
                        "kr_abs_error": weight_summary["kr_abs_error"],
                        "q90_ratio": weight_summary["q90_ratio"],
                        "q95_ratio": weight_summary["q95_ratio"],
                        "class_l1": weight_summary["class_l1"],
                        "fit_status": weight_summary["fit_status"],
                        "recovery_status": weight_summary["recovery_status"],
                        "adoption_status": weight_summary["adoption_status"],
                    }
                )
            
            # If comparing center weighting, run the alternative and write comparison
            if args.center_weighting == "proposal_area":
                # Run the unweighted model as the counterpart for comparison
                unw_fit_row, _, _, _ = fit_set_lmin(
                    set_id,
                    set_rows,
                    polygon_yz,
                    kr_grid,
                    set_likelihood_rmin,
                    args.rmax,
                    float(lmin_fit),
                    args.mc_samples_per_grid,
                    args.length_bin_count,
                    args.length_bin_mode,
                    args.window_mode,
                    direction_mode=direction_mode,
                    site=site,
                    kr_true=kr_true,
                    center_weighting="unweighted",
                    likelihood_component=args.likelihood_component,
                    class_likelihood_weight=main_class_weight,
                    oracle_radius_mode=args.oracle_radius_mode,
                    run_bootstrap=False,
                )
                
                # Append both to comparison rows
                for r, weighting_type in [(unw_fit_row, "unweighted"), (fit_row, "proposal_area")]:
                    weighting_comp_rows.append({
                        "site": site or args.dfn_model,
                        "set_id": set_id,
                        "lmin_fit": float(lmin_fit),
                        "direction_mode": direction_mode,
                        "center_weighting": weighting_type,
                        "center_weighting_status": center_weighting_status(weighting_type),
                        "kr_true": kr_true if kr_true is not None else float("nan"),
                        "kr_hat": r["kr_window_mc_hat"],
                        "kr_abs_error": abs(r["kr_window_mc_hat"] - kr_true) if kr_true is not None else float("nan"),
                        "q90_ratio": r["q90_ratio_model_observed"],
                        "q95_ratio": r["q95_ratio_model_observed"],
                        "class_l1": r["class_fraction_l1_error"],
                        "fit_status": r["fit_status"],
                        "recovery_status": r["recovery_status"],
                        "adoption_status": r["adoption_status"],
                    })
            
            if args.window_mode == "polygon":
                bbox_fit_row, _, _, _ = fit_set_lmin(
                    set_id,
                    set_rows,
                    polygon_yz,
                    kr_grid,
                    set_likelihood_rmin,
                    args.rmax,
                    float(lmin_fit),
                    args.mc_samples_per_grid,
                    args.length_bin_count,
                    args.length_bin_mode,
                    "bbox",
                    direction_mode=direction_mode,
                    site=site,
                    kr_true=kr_true,
                    center_weighting=args.center_weighting,
                    likelihood_component=args.likelihood_component,
                    class_likelihood_weight=main_class_weight,
                    oracle_radius_mode=args.oracle_radius_mode,
                    run_bootstrap=False,
                )
                comparison_rows.append(build_bbox_polygon_comparison_row(bbox_fit_row, fit_row))

    os.makedirs(args.outdir, exist_ok=True)
    fit_csv = os.path.join(args.outdir, "window_mc_fit_by_set.csv")
    fit_json = os.path.join(args.outdir, "window_mc_fit_by_set.json")
    profile_csv = os.path.join(args.outdir, "window_mc_profile_likelihood.csv")
    pp_csv = os.path.join(args.outdir, "window_mc_posterior_predictive.csv")
    comparison_csv = os.path.join(args.outdir, "window_mc_bbox_vs_polygon_comparison.csv")
    weighting_comparison_csv = os.path.join(args.outdir, "window_mc_center_weighting_comparison.csv")
    decomposition_csv = os.path.join(args.outdir, "window_mc_likelihood_decomposition.csv")
    class_weight_csv = os.path.join(args.outdir, "window_mc_class_weight_sensitivity.csv")
    predicted_survival_csv = os.path.join(args.outdir, "window_mc_predicted_survival_curve.csv")
    
    write_csv(fit_rows, fit_csv)
    write_csv(profile_rows, profile_csv)
    write_csv(pp_rows, pp_csv)
    if comparison_rows:
        write_csv(comparison_rows, comparison_csv)
    if weighting_comp_rows:
        write_csv(weighting_comp_rows, weighting_comparison_csv)
    if decomposition_rows:
        write_csv(decomposition_rows, decomposition_csv)
    if class_weight_rows:
        write_csv(class_weight_rows, class_weight_csv)
    if survival_rows:
        write_csv(survival_rows, predicted_survival_csv)
        
    with open(fit_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_summary": {
                    "trace_h5": args.trace_h5,
                    "trace_csv": args.trace_csv,
                    "target_set": args.target_set,
                    "dfn_model": args.dfn_model,
                    "generation_rmin": generation_rmin,
                    "generation_rmax": generation_rmax,
                    "estimation_rmin": float(args.rmin),
                    "estimation_rmax": float(args.rmax),
                    "likelihood_rmin": float(args.rmin),
                    "likelihood_rmax": float(args.rmax),
                    "set_rmin_mode": args.set_rmin_mode,
                    "lmin_fit_values": args.lmin_fit_values,
                    "p32_label": p32_label,
                    "rmin": args.rmin,
                    "rmax": args.rmax,
                    "kr_min": args.kr_min,
                    "kr_max": args.kr_max,
                    "profile_grid_size": args.profile_grid_size,
                    "mc_samples_per_grid": args.mc_samples_per_grid,
                    "length_bin_count": args.length_bin_count,
                    "length_bin_mode": args.length_bin_mode,
                    "window_mode": args.window_mode,
                    "center_weighting": args.center_weighting,
                    "likelihood_component": args.likelihood_component,
                    "class_likelihood_weight": args.class_likelihood_weight,
                    "main_class_likelihood_weight": main_class_weight,
                    "sensitivity_class_likelihood_weights": " ".join(str(v) for v in args.class_likelihood_weight),
                    "oracle_radius_mode": args.oracle_radius_mode,
                    "run_bootstrap": args.run_bootstrap,
                    "n_bootstrap": args.n_bootstrap,
                    "warning": WINDOW_WARNING_POLYGON if args.window_mode == "polygon" else WINDOW_WARNING_BBOX,
                },
                "fit_rows": fit_rows,
            },
            f,
            indent=2,
        )

    print("[*] Window MC fit summary")
    for row in fit_rows:
        kr_true_val = row.get("kr_true")
        kr_true_str = f", kr_true={kr_true_val:.2f}" if kr_true_val is not None and np.isfinite(float(kr_true_val)) else ""
        boot_str = ""
        if args.run_bootstrap:
            boot_str = f", boot_mean={row['kr_boot_mean']:.3f} CI=[{row['kr_ci_low']:.2f}, {row['kr_ci_high']:.2f}]"
        print(
            f"    - Set {row['set_id']}, lmin={row['lmin_fit']:.3f}: "
            f"kr={row['kr_window_mc_hat']:.3f}{kr_true_str}{boot_str}, "
            f"fit={row['fit_status']}, recovery={row['recovery_status']}, adoption={row['adoption_status']}, "
            f"mc_used={row['mc_accepted_count']}, class_l1={row['class_fraction_l1_error']:.3f}"
        )
    print(f"[*] Fit CSV written to: {fit_csv}")
    print(f"[*] Fit JSON written to: {fit_json}")
    print(f"[*] Profile CSV written to: {profile_csv}")
    print(f"[*] Posterior predictive CSV written to: {pp_csv}")
    if comparison_rows:
        print(f"[*] Bbox-vs-polygon comparison CSV written to: {comparison_csv}")
    if weighting_comp_rows:
        print(f"[*] Center weighting comparison CSV written to: {weighting_comparison_csv}")
    if decomposition_rows:
        print(f"[*] Likelihood decomposition CSV written to: {decomposition_csv}")
    if class_weight_rows:
        print(f"[*] Class-weight sensitivity CSV written to: {class_weight_csv}")


if __name__ == "__main__":
    main()
