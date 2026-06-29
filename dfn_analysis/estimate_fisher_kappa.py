from typing import Optional, Tuple

import numpy as np

from dfn_analysis.estimate_mean_orientation import estimate_mean_normal_axial, normalize


def point_line_distance(point_xyz: np.ndarray, a_xyz: np.ndarray, b_xyz: np.ndarray, eps: float = 1e-12) -> float:
    point_xyz = np.asarray(point_xyz, dtype=np.float64)
    a_xyz = np.asarray(a_xyz, dtype=np.float64)
    b_xyz = np.asarray(b_xyz, dtype=np.float64)

    ab_xyz = b_xyz - a_xyz
    ab_norm = float(np.linalg.norm(ab_xyz))
    if ab_norm < eps:
        return 0.0
    return float(np.linalg.norm(np.cross(point_xyz - a_xyz, ab_xyz)) / ab_norm)


def estimate_trace_normal_3pt(
    points_xyz: np.ndarray,
    min_quality: float = 0.005,
    eps: float = 1e-12,
) -> dict:
    pts_xyz = np.asarray(points_xyz, dtype=np.float64)
    if pts_xyz.ndim != 2 or pts_xyz.shape[0] < 3 or pts_xyz.shape[1] != 3:
        return {"valid": False, "normal": None, "quality": 0.0, "reason": "not_enough_points"}

    p0_xyz = pts_xyz[0]
    p1_xyz = pts_xyz[-1]
    chord_length = float(np.linalg.norm(p1_xyz - p0_xyz))
    if chord_length < eps:
        return {"valid": False, "normal": None, "quality": 0.0, "reason": "zero_chord_length"}

    distances = np.array(
        [point_line_distance(point_xyz, p0_xyz, p1_xyz, eps=eps) for point_xyz in pts_xyz],
        dtype=np.float64,
    )
    max_index = int(np.argmax(distances))
    pm_xyz = pts_xyz[max_index]
    max_deviation = float(distances[max_index])
    quality = max_deviation / chord_length

    normal_xyz = normalize(np.cross(p1_xyz - p0_xyz, pm_xyz - p0_xyz), eps=eps)
    if normal_xyz is None:
        return {"valid": False, "normal": None, "quality": quality, "reason": "collinear_points"}
    if quality < min_quality:
        return {"valid": False, "normal": normal_xyz, "quality": quality, "reason": "low_noncollinearity"}

    return {
        "valid": True,
        "normal": normal_xyz,
        "quality": quality,
        "reason": "ok",
        "p0_xyz": p0_xyz,
        "p1_xyz": p1_xyz,
        "pm_xyz": pm_xyz,
        "chord_length": chord_length,
        "max_deviation": max_deviation,
    }


def estimate_fisher_k_axial(normals_xyz: np.ndarray, eps: float = 1e-12) -> dict:
    normals_xyz = np.asarray(normals_xyz, dtype=np.float64)
    if normals_xyz.ndim != 2 or normals_xyz.shape[0] == 0 or normals_xyz.shape[1] != 3:
        return {"valid": False, "kappa": np.nan, "mean_normal": None, "n": 0, "resultant_length": 0.0}

    mean_normal = estimate_mean_normal_axial(normals_xyz, eps=eps)
    n_normals = len(normals_xyz)
    if mean_normal is None or n_normals < 2:
        return {
            "valid": False,
            "kappa": np.nan,
            "mean_normal": mean_normal,
            "n": n_normals,
            "resultant_length": 0.0,
        }

    # Calculate resultant length using aligned vectors
    aligned = normals_xyz.copy()
    for idx in range(len(aligned)):
        if float(np.dot(aligned[idx], mean_normal)) < 0.0:
            aligned[idx] *= -1.0
    resultant_xyz = np.sum(aligned, axis=0)
    resultant_length = float(np.linalg.norm(resultant_xyz))

    if resultant_length < eps:
        return {
            "valid": False,
            "kappa": np.nan,
            "mean_normal": mean_normal,
            "n": n_normals,
            "resultant_length": resultant_length,
        }

    r_bar = resultant_length / n_normals
    if r_bar >= 1.0 - 1e-12:
        kappa = np.inf
    else:
        kappa = (3.0 * r_bar - r_bar**3) / max(1e-12, 1.0 - r_bar**2)

    return {
        "valid": True,
        "kappa": kappa,
        "mean_normal": mean_normal,
        "n": n_normals,
        "resultant_length": resultant_length,
    }
