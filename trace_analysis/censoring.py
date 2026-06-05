"""Censoring classification against a tunnel-face observation polygon."""

from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd


def load_tunnel_polygon(polygon_path: str) -> np.ndarray:
    """Load a tunnel polygon from CSV or JSON as an array of [y, z] rows."""
    if polygon_path.lower().endswith(".csv"):
        df = pd.read_csv(polygon_path)
        if not {"y", "z"}.issubset(df.columns):
            raise ValueError("Tunnel polygon CSV must contain 'y' and 'z' columns.")
        polygon = df[["y", "z"]].to_numpy(dtype=float)
    elif polygon_path.lower().endswith(".json"):
        with open(polygon_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        points = data.get("points", data)
        polygon = np.asarray([[pt["y"], pt["z"]] for pt in points], dtype=float)
    else:
        raise ValueError("Unsupported tunnel polygon file. Use CSV or JSON.")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Tunnel polygon must contain at least three [y, z] points.")
    return polygon


def polygon_area(poly_yz: np.ndarray) -> float:
    """Compute polygon area in the Y-Z plane using the shoelace formula."""
    y = poly_yz[:, 0]
    z = poly_yz[:, 1]
    return float(0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1))))


def _point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def point_to_polygon_distance(p_y: float, p_z: float, poly_yz: np.ndarray) -> float:
    """Compute the shortest distance from a point to a closed polygon boundary."""
    point = np.array([p_y, p_z], dtype=float)
    distances = []
    for idx in range(len(poly_yz)):
        a = poly_yz[idx]
        b = poly_yz[(idx + 1) % len(poly_yz)]
        distances.append(_point_to_segment_distance(point, a, b))
    return float(min(distances))


def classify_trace_censoring(
    p0_y: float,
    p0_z: float,
    p1_y: float,
    p1_z: float,
    tunnel_polygon_yz: np.ndarray,
    boundary_tolerance: float,
) -> Dict[str, float | bool | int | str]:
    """Classify whether trace endpoints lie on the observation window boundary."""
    if tunnel_polygon_yz is None or len(tunnel_polygon_yz) < 3:
        return {
            "censoring_class": "Unknown",
            "is_p0_on_boundary": False,
            "is_p1_on_boundary": False,
            "distance_p0_to_boundary": np.nan,
            "distance_p1_to_boundary": np.nan,
        }

    dist0 = point_to_polygon_distance(p0_y, p0_z, tunnel_polygon_yz)
    dist1 = point_to_polygon_distance(p1_y, p1_z, tunnel_polygon_yz)
    touch0 = dist0 <= boundary_tolerance
    touch1 = dist1 <= boundary_tolerance
    if touch0 and touch1:
        censoring_class = 2
    elif touch0 or touch1:
        censoring_class = 1
    else:
        censoring_class = 0
    return {
        "censoring_class": censoring_class,
        "is_p0_on_boundary": bool(touch0),
        "is_p1_on_boundary": bool(touch1),
        "distance_p0_to_boundary": float(dist0),
        "distance_p1_to_boundary": float(dist1),
    }


def append_censoring_columns(
    qc_df: pd.DataFrame,
    tunnel_polygon_yz: np.ndarray,
    boundary_tolerance: float,
) -> pd.DataFrame:
    """Append censoring columns to a trace QC dataframe."""
    rows = []
    for _, row in qc_df.iterrows():
        result = classify_trace_censoring(
            p0_y=float(row["p0_y"]),
            p0_z=float(row["p0_z"]),
            p1_y=float(row["p1_y"]),
            p1_z=float(row["p1_z"]),
            tunnel_polygon_yz=tunnel_polygon_yz,
            boundary_tolerance=boundary_tolerance,
        )
        rows.append(result)
    censoring_df = pd.DataFrame(rows)
    return pd.concat([qc_df.reset_index(drop=True), censoring_df], axis=1)
