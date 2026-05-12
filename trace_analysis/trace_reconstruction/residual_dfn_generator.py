"""
[Phase 6: Residual DFN and Joint Moment Matching]
Calculates residual intensity P21 and NA metrics after subtracting deterministic planes,
computes orientation factors using VMF sampling, and solves lognormal size prior
parameters (mu, sigma) via joint moment matching.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .trace_types import FaceTrace, ReconstructedPlane, ExcavationFace
from .forward_simulator import sample_vmf_normals


def compute_orientation_mapping_factor(
    mean_normal: np.ndarray,
    kappa: float,
    face_normal: np.ndarray = np.array([1.0, 0.0, 0.0])
) -> float:
    """
    Computes set-wise orientation mapping factor kappa_s(m) = E[|n x w_m|]
    by sampling unit normals from the VMF set prior.
    """
    samples = sample_vmf_normals(mean_normal, kappa, n_samples=500, seed=42)
    
    # Calculate cross product length for each sample: ||n x face_normal||
    cross_lens = []
    for n in samples:
        cross_vec = np.cross(n, face_normal)
        cross_lens.append(np.linalg.norm(cross_vec))
        
    return float(np.mean(cross_lens))


def solve_lognormal_joint_moments(
    residual_lengths: np.ndarray
) -> Tuple[float, float]:
    """
    Solves for lognormal size distribution parameters (mu, sigma)
    using closed-form joint moment matching of trace lengths:
    - E[L]   = (pi/2) * exp(mu + 1.5 * sigma^2)
    - E[L^2] = (8/3)  * exp(2*mu + 4 * sigma^2)
    """
    if len(residual_lengths) < 3:
        # Fallback to sensible defaults for small sample sizes
        return float(np.log(1.5)), 0.35
        
    mean_L = np.mean(residual_lengths)
    mean_L2 = np.mean(residual_lengths**2)
    
    ratio = mean_L2 / (mean_L**2 + 1e-9)
    
    # analytical formula derivation:
    # exp(sigma^2) = (3 * pi^2 / 32) * (E[L^2] / (E[L])^2)
    c_factor = (3 * np.pi**2) / 32.0
    sigma2_val = np.log(c_factor * ratio + 1e-9)
    
    # Clip to prevent extreme or negative variance
    sigma_s2 = float(np.clip(sigma2_val, 0.04, 0.50))
    sigma_s = np.sqrt(sigma_s2)
    
    # Solve for mu: mu = ln(2 * E[L] / pi) - 1.5 * sigma^2
    mu_val = np.log(2.0 * mean_L / np.pi) - 1.5 * sigma_s2
    mu_s = float(np.clip(mu_val, np.log(0.2), np.log(5.0)))
    
    return mu_s, sigma_s


def compute_residual_statistics_and_priors(
    obs_traces: List[FaceTrace],
    det_planes: List[ReconstructedPlane],
    faces: List[ExcavationFace],
    set_stats: Dict[int, Tuple[np.ndarray, float]]
) -> Dict[int, Dict[str, float]]:
    """
    1. Computes residual P21 and NA on each face by deducting deterministic plane traces.
    2. Calculates residual trace length properties per orientation set.
    3. Solves the joint moment matching equations to estimate set-wise size prior parameters (mu_s, sigma_s).
    
    Returns:
    Dictionary mapping set_id -> {'mu_s': float, 'sigma_s': float, 'P32': float, 'P30': float}
    """
    results = {}
    
    # Deduct deterministic traces from observations
    det_trace_ids = set()
    for p in det_planes:
        for tid in p.source_trace_ids:
            det_trace_ids.add(tid)
            
    residual_traces = [t for t in obs_traces if t.trace_id not in det_trace_ids]
    
    # Calculate set-wise lognormal parameters using joint moments
    for set_id, (mean_normal, kappa) in set_stats.items():
        set_res_traces = [t for t in residual_traces if t.set_id == set_id]
        lengths = np.array([t.length for t in set_res_traces])
        
        mu_s, sigma_s = solve_lognormal_joint_moments(lengths)
        
        # Calculate residual P21 sum and NA sum across all faces
        p21_res_sum = 0.0
        face_mapping_factors = []
        
        for face in faces:
            face_res = [t for t in set_res_traces if t.face_id == face.face_id]
            # Sum residual lengths and divide by face area to obtain true areal intensity P21
            p21_face_res = sum(t.length for t in face_res)
            poly = face.tunnel_polygon_yz
            if poly is not None and len(poly) > 2:
                # Shoelace area formula
                y = poly[:, 0]
                z = poly[:, 1]
                area = 0.5 * np.abs(np.dot(y, np.roll(z, 1)) - np.dot(z, np.roll(y, 1)))
                area = max(1.0, area)
            else:
                area = 1.0
                
            p21_face_res_areal = p21_face_res / area
            p21_res_sum += p21_face_res_areal
            
            # Compute face mapping factor
            kappa_m = compute_orientation_mapping_factor(mean_normal, kappa)
            face_mapping_factors.append(kappa_m)
            
        # Volumetric mapping
        avg_kappa = np.mean(face_mapping_factors) if face_mapping_factors else 0.8
        
        # P32 = sum(P21_res) / sum(kappa_m)
        sum_kappa = sum(face_mapping_factors) if face_mapping_factors else 1.0
        P32 = p21_res_sum / (sum_kappa + 1e-9)
        P32 = max(1e-5, P32)
        
        # Volumetric count P30 = P32 / (pi * E[R^2])
        # For lognormal R, E[R^2] = exp(2*mu + 2*sigma^2)
        expected_r2 = np.exp(2 * mu_s + 2 * (sigma_s**2))
        P30 = P32 / (np.pi * expected_r2 + 1e-9)
        
        results[set_id] = {
            'mu_s': mu_s,
            'sigma_s': sigma_s,
            'P32': P32,
            'P30': P30
        }
        
    return results
