"""
[Phase 4: Bayes Factor Face Association]
Implements pair-wise log Bayes Factor (BF_ij) calculations (H1 same fracture vs H0 independent),
Hungarian matching with 3-sigma physical gates, and third-face visibility-gated absence penalties.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .trace_types import FaceTrace, ExcavationFace, TraceMatch


def get_candidate_plane_normal(t0: FaceTrace, t1: FaceTrace) -> np.ndarray:
    """
    Computes the exact candidate 3D plane normal defined by two parallel/semi-parallel traces.
    t0 on x = x0, t1 on x = x1.
    """
    dx = t1.x_face - t0.x_face
    
    # Average 2D trace direction unit vector in YZ plane
    theta_avg = 0.5 * (t0.orientation_2d + t1.orientation_2d)
    d_avg = np.array([0.0, np.cos(theta_avg), np.sin(theta_avg)])
    
    # Vector connecting midpoints
    v_mid = np.array([dx, t1.midpoint_y - t0.midpoint_y, t1.midpoint_z - t0.midpoint_z])
    
    # Plane normal is perpendicular to trace direction and connection vector
    n_raw = np.cross(d_avg, v_mid)
    n_len = np.linalg.norm(n_raw)
    
    if n_len > 1e-9:
        n = n_raw / n_len
    else:
        # Fallback: simple normal with no x-component
        n = np.array([0.0, -np.sin(theta_avg), np.cos(theta_avg)])
        
    # Ensure normal points to positive x hemisphere
    if n[0] < 0:
        n = -n
    return n


def check_physical_gate(
    t0: FaceTrace,
    t1: FaceTrace,
    max_angle_deg: float = 20.0,
    max_midpoint_dist: float = 1.8
) -> bool:
    """
    Applies 3-sigma physical gates as pre-filters before Bayes Factor matching.
    Returns True if the pair passes the gates.
    """
    # 1. Orientation angle difference (handling axial wrapping)
    d_theta = abs(t0.orientation_2d - t1.orientation_2d)
    if d_theta > np.pi / 2.0:
        d_theta = np.pi - d_theta
        
    if d_theta > np.radians(max_angle_deg):
        return False
        
    # 2. Euclidean distance between midpoints in YZ
    dy = t0.midpoint_y - t1.midpoint_y
    dz = t0.midpoint_z - t1.midpoint_z
    dist_yz = np.sqrt(dy**2 + dz**2)
    
    if dist_yz > max_midpoint_dist:
        return False
        
    return True


def compute_log_bayes_factor(
    t0: FaceTrace,
    t1: FaceTrace,
    set_stats: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
    sigma_theta: float = 0.087,  # Orientation noise (~5 deg in radians)
    sigma_d: float = 0.15,      # Plane-normal midpoint residual noise (~15 cm)
    bg_log_likelihood: float = -2.0  # Constant H0 log-likelihood
) -> float:
    """
    Computes log BF_ij = ln p(obs | H1) - ln p(obs | H0).
    Explicitly accounts for:
    1. Orientation similarity
    2. Coplanarity/alignment of midpoints
    3. Structural set orientation prior (VMF) if set_stats is available
    4. Distance persistence compared to estimated sizes
    """
    # Average or expected set parameters
    set_id = t0.set_id if t0.set_id == t1.set_id else None
    
    # Candidate plane normal
    n_plane = get_candidate_plane_normal(t0, t1)
    
    # 1. Orientation consistency
    d_theta = abs(t0.orientation_2d - t1.orientation_2d)
    if d_theta > np.pi / 2.0:
        d_theta = np.pi - d_theta
    ln_p_orient = - (d_theta**2) / (2 * sigma_theta**2) - np.log(np.sqrt(2 * np.pi) * sigma_theta)
    
    # 2. Spatial alignment / coplanarity
    # Point-to-plane residual for t1 relative to plane defined by n_plane passing through t0's midpoint
    v_mid = np.array([t1.x_face - t0.x_face, t1.midpoint_y - t0.midpoint_y, t1.midpoint_z - t0.midpoint_z])
    plane_dist = abs(np.dot(v_mid, n_plane))
    ln_p_spatial = - (plane_dist**2) / (2 * sigma_d**2) - np.log(np.sqrt(2 * np.pi) * sigma_d)
    
    # 3. Structural normal prior (VMF on sphere)
    ln_p_prior = 0.0
    if set_id is not None and set_stats is not None and set_id in set_stats:
        mean_normal, kappa = set_stats[set_id]
        cos_angle = abs(np.dot(n_plane, mean_normal))
        # VMF PDF is proportional to exp(kappa * cos_angle)
        # We normalize dynamically
        ln_p_prior = kappa * cos_angle - np.log(2 * np.pi * (np.exp(kappa) - np.exp(-kappa)) / kappa + 1e-9)
        
    # 4. Persistence probability (H1 requires diameter >= midpoint distance)
    dist_3d = np.linalg.norm(v_mid)
    # Simple size decay: trace length indicates candidate radius. If separation is much larger, penalize.
    # We use a loose lognormal or exponential size prior model
    expected_size = max(t0.length, t1.length) * 1.5
    if dist_3d > 2.0 * expected_size:
        ln_p_persist = -3.0 * (dist_3d / (2.0 * expected_size))
    else:
        ln_p_persist = 0.0
        
    ln_p_h1 = ln_p_orient + ln_p_spatial + ln_p_prior + ln_p_persist
    ln_p_h0 = bg_log_likelihood
    
    return float(ln_p_h1 - ln_p_h0)


def find_plane_polygon_intersection(
    normal: np.ndarray,
    center: np.ndarray,
    x_face: float,
    poly_yz: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Intersects a 3D plane with the tunnel polygon boundary at face x = x_face.
    Returns the two intersection points if the plane cuts the face.
    """
    if len(poly_yz) == 0:
        return None
        
    # Find equation of plane intersection line in the plane x = x_face:
    # n_x * (x_face - cx) + n_y * (y - cy) + n_z * (z - cz) = 0
    # => n_y * y + n_z * z = n_x * (cx - x_face) + n_y * cy + n_z * cz
    rhs = normal[0] * (center[0] - x_face) + normal[1] * center[1] + normal[2] * center[2]
    ny, nz = normal[1], normal[2]
    
    intersections = []
    n_points = len(poly_yz)
    
    for i in range(n_points):
        v0 = poly_yz[i]
        v1 = poly_yz[(i + 1) % n_points]
        
        # Line segment: P(t) = v0 + t * (v1 - v0), t in [0, 1]
        # Intersection: ny * (v0_y + t * dy) + nz * (v0_z + t * dz) = rhs
        dy = v1[0] - v0[0]
        dz = v1[1] - v0[1]
        
        denom = ny * dy + nz * dz
        if abs(denom) < 1e-9:
            continue
            
        t = (rhs - ny * v0[0] - nz * v0[1]) / denom
        if 0.0 <= t <= 1.0:
            p_inter = v0 + t * np.array([dy, dz])
            # Check for duplicate point
            if not any(np.linalg.norm(p_inter - p) < 1e-4 for p in intersections):
                intersections.append(p_inter)
                
    if len(intersections) == 2:
        return intersections[0], intersections[1]
    return None


def apply_absence_penalization(
    matches: List[TraceMatch],
    traces_f0: List[FaceTrace],
    traces_f1: List[FaceTrace],
    face_2: ExcavationFace,
    traces_f2: List[FaceTrace],
    set_stats: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
    min_visible_length: float = 0.30,
    p_detect: float = 0.90
) -> List[TraceMatch]:
    """
    Evaluates 3-face absence information:
    If a matched trace pair (Face 0, Face 1) mathematically intersects Face 2
    with visible length > min_visible_length, but NO corresponding trace is observed on Face 2,
    penalize its Bayes Factor score.
    """
    t0_by_id = {t.trace_id: t for t in traces_f0}
    t1_by_id = {t.trace_id: t for t in traces_f1}
    
    penalty = np.log(1.0 - p_detect)
    
    for m in matches:
        if not m.accepted:
            continue
            
        t0 = t0_by_id.get(m.trace_id_prev)
        t1 = t1_by_id.get(m.trace_id_curr)
        
        if t0 is None or t1 is None:
            continue
            
        # 1. Define candidate plane
        n_plane = get_candidate_plane_normal(t0, t1)
        cx = 0.5 * (t0.x_face + t1.x_face)
        cy = 0.5 * (t0.midpoint_y + t1.midpoint_y)
        cz = 0.5 * (t0.midpoint_z + t1.midpoint_z)
        center = np.array([cx, cy, cz])
        
        # 2. Intersect plane with Face 2 boundary polygon
        inter = find_plane_polygon_intersection(n_plane, center, face_2.x_face, face_2.tunnel_polygon_yz)
        if inter is None:
            continue
            
        p_a, p_b = inter
        candidate_len = np.linalg.norm(p_a - p_b)
        
        # 3. Visibility test: is the intersection trace long enough?
        if candidate_len >= min_visible_length:
            # Face 2 should contain this trace. Let's check if any observed trace on Face 2 matches it.
            mid_cand = 0.5 * (p_a + p_b)
            orient_cand = np.arctan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
            if orient_cand > np.pi / 2.0:
                orient_cand -= np.pi
            elif orient_cand < -np.pi / 2.0:
                orient_cand += np.pi
                
            has_matching_trace = False
            for t2 in traces_f2:
                # Orientation gate (axial angle difference < 20 deg)
                d_theta = abs(t2.orientation_2d - orient_cand)
                if d_theta > np.pi / 2.0:
                    d_theta = np.pi - d_theta
                    
                if d_theta <= np.radians(20.0):
                    # Spatial gate (distance from observed trace to candidate intersection midpoint < 1.5 m)
                    dist = np.sqrt((t2.midpoint_y - mid_cand[0])**2 + (t2.midpoint_z - mid_cand[1])**2)
                    if dist < 1.5:
                        has_matching_trace = True
                        break
                        
            if not has_matching_trace:
                # Candidate disc failed to appear on Face 2 -> Oversized radius rejection / Absence penalization
                m.log_bayes_factor += penalty
                # If penalization drops score below 0, we reject the match!
                if m.log_bayes_factor < 0.0:
                    m.accepted = False
                    
    return matches


def match_faces_hungarian(
    traces_prev: List[FaceTrace],
    traces_curr: List[FaceTrace],
    set_stats: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
    max_angle_deg: float = 20.0,
    max_midpoint_dist: float = 1.8
) -> List[TraceMatch]:
    """
    Matches traces between consecutive faces using log Bayes Factor scoring,
    gated by physical 3-sigma constraints, solved globally via the Hungarian algorithm.
    """
    if not traces_prev or not traces_curr:
        return []
        
    n_prev = len(traces_prev)
    n_curr = len(traces_curr)
    
    # Cost matrix for Hungarian solver (minimize cost, so we use -BF)
    # Unmatched/gated cells set to a very high cost
    cost_matrix = np.full((n_prev, n_curr), 1e6)
    bf_matrix = np.full((n_prev, n_curr), -1e6)
    
    for i, t_p in enumerate(traces_prev):
        for j, t_c in enumerate(traces_curr):
            # Same orientation set check (set mismatch is a hard gate)
            if t_p.set_id != t_c.set_id:
                continue
                
            # Physical 3-sigma gate pre-filter
            if not check_physical_gate(t_p, t_c, max_angle_deg, max_midpoint_dist):
                continue
                
            # Compute log Bayes Factor
            log_bf = compute_log_bayes_factor(t_p, t_c, set_stats)
            bf_matrix[i, j] = log_bf
            
            # Match is only viable if BF > 1 (i.e. log_bf > 0)
            if log_bf > 0.0:
                cost_matrix[i, j] = -log_bf
                
    # Solve global matching via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_ind, col_ind):
        log_bf = bf_matrix[r, c]
        is_accepted = cost_matrix[r, c] < 1e5  # Valid match
        
        matches.append(TraceMatch(
            face_id_prev=traces_prev[r].face_id,
            face_id_curr=traces_curr[c].face_id,
            trace_id_prev=traces_prev[r].trace_id,
            trace_id_curr=traces_curr[c].trace_id,
            log_bayes_factor=log_bf,
            accepted=is_accepted
        ))
        
    return matches
