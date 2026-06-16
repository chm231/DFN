"""Comparison metrics for DFN reconstruction validation."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from dfnrec.models import (
    ReconstructedDisc,
    FractureSetOrientation,
    FractureSetSizeIntensity,
    DFNParameterSet,
    Trace,
    Face,
)
from dfnrec.geometry.vector import axial_angle, normalize


def association_precision_recall(
    predicted_pairs: List[Tuple[str, str]],
    ground_truth_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute precision and recall for trace-disc association.

    Parameters
    ----------
    predicted_pairs : list of (trace_id, disc_id)
    ground_truth_pairs : list of (trace_id, disc_id)

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    pred_set = set(predicted_pairs)
    gt_set = set(ground_truth_pairs)
    tp = len(pred_set & gt_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def plane_normal_angular_error(
    estimated_discs: List[ReconstructedDisc],
    ground_truth_discs: List[ReconstructedDisc],
    id_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Mean angular error between estimated and ground-truth disc normals.

    Parameters
    ----------
    estimated_discs, ground_truth_discs : lists of ReconstructedDisc
    id_map : dict mapping estimated disc_id → ground_truth disc_id.
        If None, pairs by position in list.

    Returns
    -------
    dict: mean_deg, median_deg, max_deg, n
    """
    if id_map is None:
        pairs = list(zip(estimated_discs, ground_truth_discs))
    else:
        gt_by_id = {d.disc_id: d for d in ground_truth_discs}
        pairs = [(e, gt_by_id[id_map[e.disc_id]]) for e in estimated_discs if e.disc_id in id_map]

    angles_deg = []
    for est, gt in pairs:
        n_est = normalize(np.asarray(est.normal_xyz))
        n_gt = normalize(np.asarray(gt.normal_xyz))
        ang_rad = axial_angle(n_est, n_gt)
        angles_deg.append(math.degrees(ang_rad))

    if not angles_deg:
        return {"mean_deg": float("nan"), "median_deg": float("nan"), "max_deg": float("nan"), "n": 0}

    arr = np.array(angles_deg)
    return {
        "mean_deg": float(np.mean(arr)),
        "median_deg": float(np.median(arr)),
        "max_deg": float(np.max(arr)),
        "n": len(arr),
    }


def radius_map_relative_error(
    estimated_discs: List[ReconstructedDisc],
    ground_truth_discs: List[ReconstructedDisc],
    id_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Relative error in MAP radius estimates.

    Returns
    -------
    dict: mean_relative_error, median_relative_error, n
    """
    if id_map is None:
        pairs = list(zip(estimated_discs, ground_truth_discs))
    else:
        gt_by_id = {d.disc_id: d for d in ground_truth_discs}
        pairs = [(e, gt_by_id[id_map[e.disc_id]]) for e in estimated_discs if e.disc_id in id_map]

    rel_errs = []
    for est, gt in pairs:
        if gt.radius_m > 0:
            rel_errs.append(abs(est.radius_m - gt.radius_m) / gt.radius_m)

    if not rel_errs:
        return {"mean_relative_error": float("nan"), "median_relative_error": float("nan"), "n": 0}

    arr = np.array(rel_errs)
    return {
        "mean_relative_error": float(np.mean(arr)),
        "median_relative_error": float(np.median(arr)),
        "n": len(arr),
    }


def p32_error(
    estimated: FractureSetSizeIntensity,
    ground_truth: FractureSetSizeIntensity,
) -> Dict[str, float]:
    """Compare estimated vs. true P32_total and P32_eff.

    Returns
    -------
    dict: p32_total_error, p32_eff_error, k_r_error, relative_p32_total_error
    """
    result: Dict[str, float] = {}
    if estimated.P32_total is not None and ground_truth.P32_total is not None:
        err = estimated.P32_total - ground_truth.P32_total
        result["p32_total_error"] = err
        result["relative_p32_total_error"] = err / max(ground_truth.P32_total, 1e-9)
    if estimated.P32_eff is not None and ground_truth.P32_eff is not None:
        result["p32_eff_error"] = estimated.P32_eff - ground_truth.P32_eff
    if estimated.k_r is not None and ground_truth.k_r is not None:
        result["k_r_error"] = estimated.k_r - ground_truth.k_r
    return result


def non_observation_violation_count(
    hidden_fractures,
    faces: List[Face],
    L_min: float = 0.1,
) -> int:
    """Count hidden fractures that would have produced a visible trace.

    A non-observation constraint violation occurs when a conditional-stochastic
    fracture would produce a visible trace on one of the observation faces but
    was supposed to be absent from observations.

    Parameters
    ----------
    hidden_fractures : list of GeneratedHiddenFracture
    faces : list of Face
    L_min : float
        Minimum observable trace length [m].

    Returns
    -------
    int : number of violations (should be 0 for a correct sampler).
    """
    from dfnrec.geometry.disc_trace import predicted_visible_trace
    import numpy as np

    violations = 0
    for frac in hidden_fractures:
        for face in faces:
            result = predicted_visible_trace(
                center_xyz=np.asarray(frac.center_xyz),
                normal_xyz=np.asarray(frac.normal_xyz),
                radius_m=frac.radius_m,
                face_origin_xyz=np.asarray(face.origin_xyz),
                face_normal_xyz=np.asarray(face.normal_xyz),
                face_axis_u_xyz=np.asarray(face.axis_u_xyz),
                face_axis_v_xyz=np.asarray(face.axis_v_xyz),
                observation_window_uv=np.asarray(face.observation_window_polygon_uv),
            )
            if result.visible_length >= L_min:
                violations += 1
                break  # count each fracture at most once
    return violations


def compare_dfn_parameters(
    estimated: DFNParameterSet,
    ground_truth: DFNParameterSet,
) -> Dict[str, Any]:
    """Compare estimated vs. ground-truth DFN parameters for all sets.

    Returns a nested report dict keyed by set_id.
    """
    report: Dict[str, Any] = {}
    for sid in ground_truth.set_ids():
        set_report: Dict[str, Any] = {}
        # Orientation
        if sid in estimated.orientation and sid in ground_truth.orientation:
            est_ori = estimated.orientation[sid]
            gt_ori = ground_truth.orientation[sid]
            n_est = normalize(np.array([
                math.cos(math.radians(est_ori.mean_plunge_deg)) * math.sin(math.radians(est_ori.mean_trend_deg)),
                math.cos(math.radians(est_ori.mean_plunge_deg)) * math.cos(math.radians(est_ori.mean_trend_deg)),
                math.sin(math.radians(est_ori.mean_plunge_deg)),
            ]))
            n_gt = normalize(np.array([
                math.cos(math.radians(gt_ori.mean_plunge_deg)) * math.sin(math.radians(gt_ori.mean_trend_deg)),
                math.cos(math.radians(gt_ori.mean_plunge_deg)) * math.cos(math.radians(gt_ori.mean_trend_deg)),
                math.sin(math.radians(gt_ori.mean_plunge_deg)),
            ]))
            set_report["mean_pole_error_deg"] = math.degrees(axial_angle(n_est, n_gt))
            set_report["kappa_error"] = est_ori.kappa - gt_ori.kappa
        # Size / intensity
        if sid in estimated.size_intensity and sid in ground_truth.size_intensity:
            set_report["size_intensity"] = p32_error(
                estimated.size_intensity[sid],
                ground_truth.size_intensity[sid],
            )
        report[sid] = set_report
    return report
