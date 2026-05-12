"""
reconstruction_validator.py

Comprehensive Multi-Tier Validation Framework for 3D DFN Inverse Reconstruction.
Exposes modular APIs and a CLI to strictly evaluate:
  - Tier 1: Multi-Face Trace Matching Accuracy (Precision, Recall, F1-Score)
  - Tier 2: Deterministic Fracture Reconstruction Accuracy (SVD/MAP Geometric Errors)
  - Tier 2: SVD Stability Condition Indicators (Singular values, planarity, span)
  - Tier 2: Unobserved Face Validation (Out-of-sample Extrapolation on Face 3)
  - Tier 3: Statistical DFN & Multi-Tier Intensity Transition (5-Stage P21 metrics)
  - Tier 4: Voxel-Based Block Topology Preservation (Counts, KS-test, spearman risk rank)
"""
import os
import sys
import numpy as np
import h5py
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.stats import ks_2samp, spearmanr

# Add local path to import types
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from trace_reconstruction.trace_types import FaceTrace, ExcavationFace, ReconstructedPlane, TraceMatch


# ==============================================================================
# Tier 1: Multi-Face Trace Matching Validation
# ==============================================================================
def evaluate_trace_matching(
    obs_traces: List[FaceTrace],
    accepted_matches: List[TraceMatch]
) -> Dict[str, Any]:
    """
    Evaluates the precision, recall, and F1-score of pairwise multi-face matching
    by comparing predictions against ground-truth parent fracture IDs.
    """
    trace_lookup = {t.trace_id: t for t in obs_traces}
    
    # Identify True Positives, False Positives, False Negatives
    # A true match is a pair of trace IDs on adjacent faces that share the same non-None parent_fracture_id
    tp = 0
    fp = 0
    
    # Set of predicted matches: (id_prev, id_curr)
    pred_pairs = set()
    for m in accepted_matches:
        if m.accepted:
            pred_pairs.add((m.trace_id_prev, m.trace_id_curr))
            
            t0 = trace_lookup.get(m.trace_id_prev)
            t1 = trace_lookup.get(m.trace_id_curr)
            
            if t0 and t1 and t0.parent_fracture_id is not None and t1.parent_fracture_id is not None:
                if t0.parent_fracture_id == t1.parent_fracture_id:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1
                
    # Generate Ground-Truth adjacent matches
    gt_pairs = set()
    # Group traces by face_id
    grouped_by_face = {}
    for t in obs_traces:
        grouped_by_face.setdefault(t.face_id, []).append(t)
        
    faces_sorted = sorted(grouped_by_face.keys())
    for idx in range(len(faces_sorted) - 1):
        f0 = faces_sorted[idx]
        f1 = faces_sorted[idx + 1]
        
        traces_f0 = grouped_by_face[f0]
        traces_f1 = grouped_by_face[f1]
        
        # Cross-reference parent fracture IDs
        for t0 in traces_f0:
            if t0.parent_fracture_id is None:
                continue
            for t1 in traces_f1:
                if t1.parent_fracture_id == t0.parent_fracture_id:
                    gt_pairs.add((t0.trace_id, t1.trace_id))
                    
    # False Negatives are true matches that are NOT predicted
    fn = 0
    for gt_p in gt_pairs:
        if gt_p not in pred_pairs:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "gt_match_count": len(gt_pairs),
        "pred_match_count": len(pred_pairs),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# ==============================================================================
# Tier 2: SVD Stability & Condition Indicators
# ==============================================================================
def evaluate_svd_stability(track_traces: List[FaceTrace]) -> Dict[str, Any]:
    """
    Analyzes the 3D endpoint point cloud of a merged track to compute
    condition indicators for SVD flat fitting.
    """
    endpoints_3d = []
    for t in track_traces:
        endpoints_3d.append([t.x_face, t.p0_y, t.p0_z])
        endpoints_3d.append([t.x_face, t.p1_y, t.p1_z])
    endpoints_3d = np.array(endpoints_3d)
    
    centroid = np.mean(endpoints_3d, axis=0)
    shifted = endpoints_3d - centroid
    
    # Compute SVD singular values
    _, s, _ = np.linalg.svd(shifted)
    
    # Singular values are sorted in descending order: s0 >= s1 >= s2
    s0, s1, s2 = s[0], s[1], s[2] if len(s) > 2 else (s[0], s[1], 1e-9)
    if s2 < 1e-9:
        s2 = 1e-9
        
    sv_ratio_1 = float(s0 / s2)
    sv_ratio_2 = float(s1 / s2)
    planarity_ratio = float((s1 - s2) / s0) if s0 > 0 else 0.0
    spread_ratio = float(s2 / (s0 + s1 + s2)) if (s0 + s1 + s2) > 0 else 0.0
    
    x_coords = endpoints_3d[:, 0]
    face_span = float(np.max(x_coords) - np.min(x_coords))
    endpoint_count = len(endpoints_3d)
    
    # Establish a soft confidence score
    # SVD plane fitting is extremely stable if faces have a wide span (> 1.5m), 
    # more endpoints (>= 6), and a low spread_ratio (< 0.1)
    is_stable = True
    reasons = []
    if face_span < 1.5:
        is_stable = False
        reasons.append("span_too_narrow")
    if endpoint_count < 6:
        is_stable = False
        reasons.append("insufficient_endpoints")
    if sv_ratio_1 > 100.0:
        is_stable = False
        reasons.append("high_singular_anisotropy")
        
    confidence_flag = "stable" if is_stable else f"unstable({','.join(reasons)})"
    
    return {
        "singular_values": s.tolist(),
        "sv_ratio_1": sv_ratio_1,
        "sv_ratio_2": sv_ratio_2,
        "planarity_ratio": planarity_ratio,
        "spread_ratio": spread_ratio,
        "face_span": face_span,
        "endpoint_count": endpoint_count,
        "confidence_flag": confidence_flag
    }


# ==============================================================================
# Tier 2: Fracture Geometry Reconstruction Error
# ==============================================================================
def evaluate_deterministic_geometry(
    det_planes: List[ReconstructedPlane],
    tracks: List[List[FaceTrace]],
    gt_centers: np.ndarray,
    gt_normals: np.ndarray,
    gt_radii: np.ndarray
) -> Dict[str, Any]:
    """
    Compares reconstructed deterministic planes with original ground-truth DFN models.
    Retrieves matching GT fractures via majority voting of track trace parent_fracture_ids.
    """
    angular_errors = []
    center_errors = []
    radius_errors = []
    plane_dist_errors = []
    reproj_errors = []
    
    stability_logs = []
    
    from trace_reconstruction.forward_simulator import intersect_disc_with_face
    
    for dp, track in zip(det_planes, tracks):
        # 1. Majority vote for parent fracture ID
        parent_ids = [t.parent_fracture_id for t in track if t.parent_fracture_id is not None]
        if not parent_ids:
            continue
        
        gt_idx = max(set(parent_ids), key=parent_ids.count)
        if gt_idx >= len(gt_radii):
            continue
            
        gt_c = gt_centers[gt_idx]
        gt_n = gt_normals[gt_idx]
        gt_r = gt_radii[gt_idx]
        
        # 2. Compute Geometrical Errors
        dp_n = np.array([dp.normal_x, dp.normal_y, dp.normal_z])
        dp_c = np.array([dp.point_x, dp.point_y, dp.point_z])
        
        # Angular error between two normals (accounting for axial sign)
        dot = abs(np.dot(dp_n, gt_n))
        dot = min(1.0, max(-1.0, dot))
        ang_err = float(np.arccos(dot) * 180.0 / np.pi)
        angular_errors.append(ang_err)
        
        # Center distance error (Euclidean)
        c_err = float(np.linalg.norm(dp_c - gt_c))
        center_errors.append(c_err)
        
        # Radius relative error
        r_err = float(abs(dp.radius - gt_r) / gt_r)
        radius_errors.append(r_err)
        
        # Plane-to-plane distance (shortest 3D offset)
        plane_offset = float(abs(np.dot(gt_c - dp_c, dp_n)))
        plane_dist_errors.append(plane_offset)
        
        # Endpoint reprojection error
        trace_reproj_errs = []
        for t in track:
            # Re-project reconstructed plane back onto the face
            face_obj = ExcavationFace(face_id=t.face_id, x_face=t.x_face, tunnel_polygon_yz=np.array([[0,0]]), advance_step=0.0)
            reproj_traces = intersect_disc_with_face(
                dp.point_x, dp.point_y, dp.point_z,
                dp.normal_x, dp.normal_y, dp.normal_z,
                dp.radius, face_obj, start_trace_id=9999, set_id=1
            )
            if reproj_traces:
                # Calculate endpoint distance to original trace t
                rt = reproj_traces[0]
                dist_endpoints_p0 = np.linalg.norm(np.array([t.p0_y, t.p0_z]) - np.array([rt.p0_y, rt.p0_z]))
                dist_endpoints_p1 = np.linalg.norm(np.array([t.p1_y, t.p1_z]) - np.array([rt.p1_y, rt.p1_z]))
                trace_reproj_errs.append(0.5 * (dist_endpoints_p0 + dist_endpoints_p1))
            else:
                trace_reproj_errs.append(t.length)  # Severe failure penalty
                
        reproj_errors.append(np.mean(trace_reproj_errs))
        
        # 3. Log stability metrics alongside errors
        stab = evaluate_svd_stability(track)
        stab.update({
            "plane_id": dp.plane_id,
            "angular_error": ang_err,
            "center_error": c_err,
            "radius_error": r_err,
            "plane_dist_error": plane_offset,
            "reproj_error": reproj_errors[-1]
        })
        stability_logs.append(stab)
        
    return {
        "mean_angular_error": float(np.mean(angular_errors)) if angular_errors else 0.0,
        "median_angular_error": float(np.median(angular_errors)) if angular_errors else 0.0,
        "mean_center_error": float(np.mean(center_errors)) if center_errors else 0.0,
        "mean_radius_error": float(np.mean(radius_errors)) if radius_errors else 0.0,
        "mean_plane_dist_error": float(np.mean(plane_dist_errors)) if plane_dist_errors else 0.0,
        "mean_reproj_error": float(np.mean(reproj_errors)) if reproj_errors else 0.0,
        "stability_logs": stability_logs
    }


# ==============================================================================
# Tier 2: Unobserved Face Validation (Face 3 Extrapolation)
# ==============================================================================
def evaluate_unobserved_face_prediction(
    det_planes: List[ReconstructedPlane],
    faces: List[ExcavationFace],
    obs_traces: List[FaceTrace]
) -> Dict[str, Any]:
    """
    Validates generalizability by taking planes reconstructed ONLY from Face 1 and Face 2,
    extrapolating them to Face 3 (x_face=6.0m), and comparing the predicted traces
    with actual observations on Face 3.
    """
    from trace_reconstruction.forward_simulator import intersect_disc_with_face
    
    # 1. Filter planes reconstructed from Face 1 and Face 2 (i.e. x = 0m, 3m)
    # They should not contain Face 3 (x=6m) in their source tracks
    extrap_planes = []
    for dp in det_planes:
        # Check source traces face IDs
        # Face IDs are 1 and 2
        source_traces_faces = [t.face_id for t in obs_traces if t.trace_id in dp.source_trace_ids]
        if 3 not in source_traces_faces and (1 in source_traces_faces or 2 in source_traces_faces):
            extrap_planes.append(dp)
            
    if not extrap_planes or len(faces) < 3:
        return {"status": "skipped", "reason": "No valid planes or less than 3 faces."}
        
    face_3 = faces[2] # Face 3 is index 2 (x=6.0m)
    actual_face_3_traces = [t for t in obs_traces if t.face_id == 3]
    
    # 2. Intersect and predict traces on Face 3
    predicted_traces = []
    for dp in extrap_planes:
        pts = intersect_disc_with_face(
            dp.point_x, dp.point_y, dp.point_z,
            dp.normal_x, dp.normal_y, dp.normal_z,
            dp.radius, face_3, start_trace_id=20000 + dp.plane_id, set_id=dp.set_id or 1
        )
        predicted_traces.extend(pts)
        
    # 3. Match predicted traces with actual face 3 traces based on 2D position & angle
    matched_preds = 0
    angle_residuals = []
    length_diffs = []
    
    used_actual = set()
    for pt in predicted_traces:
        best_match = None
        best_dist = 1e9
        
        for at in actual_face_3_traces:
            if at.trace_id in used_actual:
                continue
            # Distance of midpoints
            dist = np.sqrt((pt.midpoint_y - at.midpoint_y)**2 + (pt.midpoint_z - at.midpoint_z)**2)
            # Match boundary: distance < 2.0m, angle diff < 15 deg
            ang_diff = abs(pt.orientation_2d - at.orientation_2d) * 180.0 / np.pi
            if ang_diff > 90.0:
                ang_diff = 180.0 - ang_diff
                
            if dist < 2.0 and ang_diff < 15.0 and dist < best_dist:
                best_dist = dist
                best_match = at
                
        if best_match:
            matched_preds += 1
            used_actual.add(best_match.trace_id)
            # Record metrics
            ang_diff = abs(pt.orientation_2d - best_match.orientation_2d) * 180.0 / np.pi
            if ang_diff > 90.0:
                ang_diff = 180.0 - ang_diff
            angle_residuals.append(ang_diff)
            length_diffs.append(abs(pt.length - best_match.length))
            
    match_rate = matched_preds / len(predicted_traces) if predicted_traces else 0.0
    recall_rate = matched_preds / len(actual_face_3_traces) if actual_face_3_traces else 0.0
    
    return {
        "status": "success",
        "n_extrapolated_planes": len(extrap_planes),
        "n_predicted_traces": len(predicted_traces),
        "n_actual_traces": len(actual_face_3_traces),
        "n_matched_traces": matched_preds,
        "prediction_precision": match_rate,
        "prediction_recall": recall_rate,
        "mean_angle_residual": float(np.mean(angle_residuals)) if angle_residuals else 0.0,
        "mean_length_residual": float(np.mean(length_diffs)) if length_diffs else 0.0
    }


# ==============================================================================
# Tier 4: Voxel-Based Block Topology Preservation Validation
# ==============================================================================
def evaluate_block_topology(
    gt_block_dir: str,
    rec_block_dir: str
) -> Dict[str, Any]:
    """
    Evaluates how well the reconstructed 3D DFN preserves block-detector topologies
    relative to the ground-truth 3D DFN block detector output.
    """
    gt_csv = os.path.join(gt_block_dir, "blocks.csv")
    rec_csv = os.path.join(rec_block_dir, "blocks.csv")
    
    if not os.path.exists(gt_csv) or not os.path.exists(rec_csv):
        return {
            "status": "skipped",
            "reason": f"Missing blocks.csv in one or both directories. GT={os.path.exists(gt_csv)}, REC={os.path.exists(rec_csv)}"
        }
        
    gt_df = pd.read_csv(gt_csv)
    rec_df = pd.read_csv(rec_csv)
    
    n_blocks_gt = len(gt_df)
    n_blocks_rec = len(rec_df)
    
    count_error = (n_blocks_rec - n_blocks_gt) / n_blocks_gt if n_blocks_gt > 0 else 0.0
    
    # Volume distribution analysis
    v_gt = gt_df["Volume"].values if n_blocks_gt > 0 else np.array([])
    v_rec = rec_df["Volume"].values if n_blocks_rec > 0 else np.array([])
    
    max_v_gt = np.max(v_gt) if n_blocks_gt > 0 else 0.0
    max_v_rec = np.max(v_rec) if n_blocks_rec > 0 else 0.0
    largest_volume_err = (max_v_rec - max_v_gt) / max_v_gt if max_v_gt > 0 else 0.0
    
    # 2-sample KS-test of volume distributions
    ks_stat = 1.0
    ks_p = 0.0
    if n_blocks_gt > 2 and n_blocks_rec > 2:
        ks_res = ks_2samp(v_gt, v_rec)
        ks_stat = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)
        
    # Check spatial center overlap
    # A reconstructed block is matched if its centroid is within 1.0m of a GT block
    matched_blocks = 0
    used_rec = set()
    for idx_gt, row_gt in gt_df.iterrows():
        c_gt = np.array([row_gt["Centroid_X"], row_gt["Centroid_Y"], row_gt["Centroid_Z"]])
        
        best_rec_idx = None
        best_dist = 1e9
        for idx_rec, row_rec in rec_df.iterrows():
            if idx_rec in used_rec:
                continue
            c_rec = np.array([row_rec["Centroid_X"], row_rec["Centroid_Y"], row_rec["Centroid_Z"]])
            dist = np.linalg.norm(c_gt - c_rec)
            if dist < 1.0 and dist < best_dist:
                best_dist = dist
                best_rec_idx = idx_rec
                
        if best_rec_idx is not None:
            matched_blocks += 1
            used_rec.add(best_rec_idx)
            
    overlap_iou = matched_blocks / (n_blocks_gt + n_blocks_rec - matched_blocks) if (n_blocks_gt + n_blocks_rec - matched_blocks) > 0 else 0.0
    
    # Spearman rank correlation of risks
    # We rank blocks by volume as a proxy of risk
    spearman_rho = 0.0
    spearman_p = 1.0
    if matched_blocks > 3:
        # Match up paired volumes
        matched_gt_v = []
        matched_rec_v = []
        used_rec_corr = set()
        for idx_gt, row_gt in gt_df.iterrows():
            c_gt = np.array([row_gt["Centroid_X"], row_gt["Centroid_Y"], row_gt["Centroid_Z"]])
            
            best_rec_idx = None
            best_dist = 1e9
            for idx_rec, row_rec in rec_df.iterrows():
                if idx_rec in used_rec_corr:
                    continue
                c_rec = np.array([row_rec["Centroid_X"], row_rec["Centroid_Y"], row_rec["Centroid_Z"]])
                dist = np.linalg.norm(c_gt - c_rec)
                if dist < 1.0 and dist < best_dist:
                    best_dist = dist
                    best_rec_idx = idx_rec
            if best_rec_idx is not None:
                matched_gt_v.append(row_gt["Volume"])
                matched_rec_v.append(rec_df.loc[best_rec_idx, "Volume"])
                used_rec_corr.add(best_rec_idx)
                
        rho, pval = spearmanr(matched_gt_v, matched_rec_v)
        spearman_rho = float(rho) if not np.isnan(rho) else 0.0
        spearman_p = float(pval) if not np.isnan(pval) else 1.0
        
    return {
        "status": "success",
        "n_gt_blocks": n_blocks_gt,
        "n_rec_blocks": n_blocks_rec,
        "count_error": count_error,
        "gt_largest_volume": max_v_gt,
        "rec_largest_volume": max_v_rec,
        "largest_volume_error": largest_volume_err,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "matched_block_count": matched_blocks,
        "overlap_iou": overlap_iou,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_p
    }


# ==============================================================================
# Multi-Tier Validation Master Logger
# ==============================================================================
def compile_validation_report(
    obs_traces: List[FaceTrace],
    accepted_matches: List[TraceMatch],
    det_planes: List[ReconstructedPlane],
    tracks: List[List[FaceTrace]],
    faces: List[ExcavationFace],
    gt_file: str,
    gt_block_dir: Optional[str] = None,
    rec_block_dir: Optional[str] = None,
    p21_transition: Optional[Dict[str, float]] = None
) -> str:
    """
    Compiles a comprehensive, academic-grade validation report across Tiers 1-4.
    """
    # 1. Load GT file arrays
    with h5py.File(gt_file, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        gt_radii = f['/fractures/radii'][:].ravel()
        gt_centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        gt_normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        
    # 2. Compute Tiers
    t1 = evaluate_trace_matching(obs_traces, accepted_matches)
    t2 = evaluate_deterministic_geometry(det_planes, tracks, gt_centers, gt_normals, gt_radii)
    t2_extrap = evaluate_unobserved_face_prediction(det_planes, faces, obs_traces)
    t4 = evaluate_block_topology(gt_block_dir, rec_block_dir) if gt_block_dir and rec_block_dir else {"status": "skipped"}
    
    # 3. Format Report
    report = []
    report.append("================================================================================")
    report.append("              [INVERSE DFN RECONSTRUCTION MULTI-TIER VALIDATION]")
    report.append("================================================================================")
    
    # Tier 1
    report.append(f"\n* Tier 1: Multi-Face Trace Matching Validation")
    report.append(f"  - Pred matched trace pairs: {t1['pred_match_count']} vs GT matched trace pairs: {t1['gt_match_count']}")
    report.append(f"  - True Positives: {t1['true_positives']} | False Positives: {t1['false_positives']} | False Negatives: {t1['false_negatives']}")
    report.append(f"  - Precision: {t1['precision']:.4f} ({t1['precision']*100:.2f}%)")
    report.append(f"  - Recall:    {t1['recall']:.4f} ({t1['recall']*100:.2f}%)")
    report.append(f"  - F1-Score:  {t1['f1_score']:.4f} ({t1['f1_score']*100:.2f}%)")
    
    # Tier 2 Deterministic
    report.append(f"\n* Tier 2: Deterministic Fracture Reconstruction Accuracy (SVD vs GT)")
    report.append(f"  - Evaluated deterministic planes: {len(det_planes)}")
    report.append(f"  - Mean Angular Normal Error:  {t2['mean_angular_error']:.3f}° | Median: {t2['median_angular_error']:.3f}°")
    report.append(f"  - Mean Center Offset Error:   {t2['mean_center_error']:.3f} m")
    report.append(f"  - Mean Radius Offset Error:   {t2['mean_radius_error']*100:.2f}%")
    report.append(f"  - Mean Plane distance error:  {t2['mean_plane_dist_error']:.3f} m")
    report.append(f"  - Mean Reprojected trace err: {t2['mean_reproj_error']:.3f} m")
    
    # SVD indicators
    report.append(f"\n* Tier 2: SVD Stability Condition Indicators (Soft-Filtering Audit)")
    stable_cnt = sum(1 for log in t2['stability_logs'] if "stable" in log['confidence_flag'] and "unstable" not in log['confidence_flag'])
    report.append(f"  - Stable SVD tracks: {stable_cnt}/{len(det_planes)} ({stable_cnt/len(det_planes)*100 if det_planes else 0:.1f}%)")
    for log in t2['stability_logs'][:5]:  # Show first 5 logs
        report.append(f"    [Track {log['plane_id']}] Span: {log['face_span']:.1f}m | Endpoints: {log['endpoint_count']} | SV Ratio 1: {log['sv_ratio_1']:.1f} | Flag: {log['confidence_flag']}")
        report.append(f"               -> Geometry Errors: Angular: {log['angular_error']:.2f}° | Center: {log['center_error']:.2f}m | Reproj: {log['reproj_error']:.2f}m")
        
    # Tier 2 Extrapolated
    report.append(f"\n* Tier 2: Unobserved Face Validation (Face 3 Extrapolation)")
    if t2_extrap['status'] == 'success':
        report.append(f"  - Reconstructed planes from Face 1 & 2 used: {t2_extrap['n_extrapolated_planes']}")
        report.append(f"  - Extrapolated predicted traces on Face 3:    {t2_extrap['n_predicted_traces']} vs Actual: {t2_extrap['n_actual_traces']}")
        report.append(f"  - Matched prediction count:                  {t2_extrap['n_matched_traces']}")
        report.append(f"  - Out-of-sample Precision:                   {t2_extrap['prediction_precision']:.4f} ({t2_extrap['prediction_precision']*100:.2f}%)")
        report.append(f"  - Out-of-sample Recall:                      {t2_extrap['prediction_recall']:.4f} ({t2_extrap['prediction_recall']*100:.2f}%)")
        report.append(f"  - Extrapolated angle residual mean:          {t2_extrap['mean_angle_residual']:.2f}°")
        report.append(f"  - Extrapolated length residual mean:         {t2_extrap['mean_length_residual']:.2f} m")
    else:
        report.append(f"  - [SKIPPED] Reason: {t2_extrap.get('reason', 'N/A')}")
        
    # Tier 3
    if p21_transition:
        report.append(f"\n* Tier 3: Statistical DFN & Multi-tier Intensity Transition")
        report.append(f"  - P21_obs (Observed Total):      {p21_transition.get('P21_obs', 0.0):.4f} m/m^2")
        p21_det = p21_transition.get('P21_det_only', 0.0)
        p21_obs = p21_transition.get('P21_obs', 1.0)
        p21_ratio = (p21_det / p21_obs) * 100 if p21_obs > 0 else 0.0
        report.append(f"  - P21_det_only (Deterministic):  {p21_det:.4f} m/m^2 (Ratio: {p21_ratio:.2f}%)")
        report.append(f"  - P21_residual_target (Target):  {p21_transition.get('P21_residual_target', 0.0):.4f} m/m^2")
        report.append(f"  - P21_stochastic (Stochastic):   {p21_transition.get('P21_stochastic', 0.0):.4f} m/m^2")
        report.append(f"  - P21_total (Combined Final):    {p21_transition.get('P21_total', 0.0):.4f} m/m^2")
        p21_err = p21_transition.get('p21_error', 0.0)
        report.append(f"  - Combined P21 error:            {p21_err:.4f} ({p21_err*100:.2f}%)")
        
    # Tier 4
    report.append(f"\n* Tier 4: Block Topology Preservation Metrics")
    if t4['status'] == 'success':
        report.append(f"  - GT Block Count: {t4['n_gt_blocks']} vs Reconstructed Block Count: {t4['n_rec_blocks']} (Error: {t4['count_error']*100:.2f}%)")
        report.append(f"  - GT Largest block volume: {t4['gt_largest_volume']:.3f} m^3 vs Reconstructed: {t4['rec_largest_volume']:.3f} m^3 (Error: {t4['largest_volume_error']*100:.2f}%)")
        report.append(f"  - Volume KS-Test p-value:  {t4['ks_pvalue']:.5f} (KS stat: {t4['ks_statistic']:.4f}) -> {'Pass' if t4['ks_pvalue'] > 0.05 else 'Fail (Significantly different volume distributions)'}")
        report.append(f"  - Keyblock spatial centroid IoU overlap: {t4['overlap_iou']:.4f} ({t4['overlap_iou']*100:.2f}%)")
        report.append(f"  - Block Risk Ranking Correlation (Spearman): rho = {t4['spearman_rho']:.4f} | p-value: {t4['spearman_pvalue']:.5f}")
    else:
        report.append(f"  - [SKIPPED] Reason: {t4.get('reason', 'N/A')}")
        
    report.append("================================================================================")
    
    return "\n".join(report)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-tier DFN Inverse Reconstruction Validator")
    parser.add_argument("--gt-file", required=True, help="Ground Truth HDF5 DFN file")
    parser.add_argument("--gt-block-dir", help="Ground Truth block result directory")
    parser.add_argument("--rec-block-dir", help="Reconstructed block result directory")
    
    args = parser.parse_args()
    print("[*] Running reconstruction_validator as standalone...")
