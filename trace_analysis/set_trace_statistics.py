"""Set-wise observed trace statistics on tunnel excavation faces."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

try:
    from trace_analysis.censoring import polygon_area
except ImportError:
    from censoring import polygon_area


def _axial_mean_and_dispersion(theta_deg: pd.Series) -> Tuple[float, float]:
    if theta_deg.empty:
        return np.nan, np.nan
    theta_rad = np.deg2rad(theta_deg.to_numpy(dtype=float))
    doubled = 2.0 * theta_rad
    c_mean = np.mean(np.cos(doubled))
    s_mean = np.mean(np.sin(doubled))
    mean_angle = 0.5 * np.arctan2(s_mean, c_mean)
    if mean_angle < 0.0:
        mean_angle += np.pi
    resultant = float(np.sqrt(c_mean**2 + s_mean**2))
    dispersion = float(1.0 - resultant)
    return float(np.degrees(mean_angle)), dispersion


def compute_set_observed_statistics(
    qc_df: pd.DataFrame,
    tunnel_polygon_yz: np.ndarray | None,
) -> pd.DataFrame:
    """Aggregate set-wise observed trace statistics using Y-Z trace representation.

    The observation window for set-wise intensity is the union of all analyzed
    excavation faces, not a single face. We therefore normalize by

        face_polygon_area * n_analyzed_faces

    so that total trace length collected across multiple faces is not divided by
    only one face area.
    """
    area = np.nan
    if tunnel_polygon_yz is not None:
        area = polygon_area(tunnel_polygon_yz)
    n_faces = int(qc_df["face_id"].nunique()) if "face_id" in qc_df.columns else 1
    total_observation_area = float(area * n_faces) if np.isfinite(area) else np.nan

    records = []
    for set_id, group in qc_df.groupby("set_id", sort=True):
        valid_group = group[group["valid_length"]]
        theta_mean, theta_dispersion = _axial_mean_and_dispersion(valid_group["theta_yz_deg"])
        total_length = float(valid_group["length_yz"].sum())
        observed_p21 = (
            float(total_length / total_observation_area)
            if np.isfinite(total_observation_area) and total_observation_area > 0
            else np.nan
        )
        type0_count = int((group["censoring_class"] == 0).sum()) if "censoring_class" in group else 0
        type1_count = int((group["censoring_class"] == 1).sum()) if "censoring_class" in group else 0
        type2_count = int((group["censoring_class"] == 2).sum()) if "censoring_class" in group else 0
        n_traces = int(len(group))
        records.append(
            {
                "set_id": int(set_id),
                "n_traces": n_traces,
                "n_valid_traces": int(len(valid_group)),
                "total_length_yz": total_length,
                "mean_length_yz": float(valid_group["length_yz"].mean()) if len(valid_group) else np.nan,
                "median_length_yz": float(valid_group["length_yz"].median()) if len(valid_group) else np.nan,
                "std_length_yz": float(valid_group["length_yz"].std(ddof=0)) if len(valid_group) else np.nan,
                "min_length_yz": float(valid_group["length_yz"].min()) if len(valid_group) else np.nan,
                "max_length_yz": float(valid_group["length_yz"].max()) if len(valid_group) else np.nan,
                "observed_P21": observed_p21,
                "censoring_type_0_count": type0_count,
                "censoring_type_1_count": type1_count,
                "censoring_type_2_count": type2_count,
                "censored_ratio": float((type1_count + type2_count) / max(n_traces, 1)),
                "theta_yz_mean_axial": theta_mean,
                "theta_yz_dispersion": theta_dispersion,
                "single_face_area": area,
                "observation_window_face_count": n_faces,
                "observation_window_area": total_observation_area,
            }
        )
    return pd.DataFrame.from_records(records)
