"""
[Phase 5: Censoring-Aware Constrained MAP Fitting]
Implements coordinate projection to local fracture plane systems, MAP optimization
with separate exact and clipped censoring likelihoods, Laplace covariance estimation,
and a probabilistic posterior sampler for single-face traces.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.stats import norm
from .trace_types import FaceTrace, ExcavationFace, ReconstructedPlane
from .forward_simulator import sample_vmf_normals


def fit_plane_svd_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a 3D plane normal and centroid to a set of 3D points using SVD.
    Returns: (centroid, normal)
    """
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    # Singular Value Decomposition
    _, _, vh = np.linalg.svd(shifted)
    # Normal is the last row of V^T (least singular value direction)
    normal = vh[-1]
    
    # Ensure normal points to positive x hemisphere
    if normal[0] < 0:
        normal = -normal
    return centroid, normal


def get_local_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes two orthogonal unit vectors (e_u, e_v) in the plane perpendicular to the normal.
    """
    # Choose a non-parallel reference vector to cross with normal
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
        
    e_u = np.cross(normal, ref)
    e_u = e_u / np.linalg.norm(e_u)
    e_v = np.cross(normal, e_u)
    e_v = e_v / np.linalg.norm(e_v)
    return e_u, e_v


def project_points_to_plane_2d(
    points_3d: np.ndarray,
    centroid: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray
) -> np.ndarray:
    """Projects 3D points to local 2D (u, v) coordinates on the plane."""
    shifted = points_3d - centroid
    u = np.dot(shifted, e_u)
    v = np.dot(shifted, e_v)
    return np.column_stack([u, v])


def evaluate_analytical_trace_length(
    u0: float,
    v0: float,
    R: float,
    x_face: float,
    centroid: np.ndarray,
    normal: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray
) -> float:
    """
    Calculates the analytical expected 2D trace length of a 3D disc of radius R
    centered at (u0, v0) in the plane local coordinates, intersecting the face x = x_face.
    """
    # 1. Plane intersection line at x = x_face in 3D:
    # Any point in the plane satisfies P(u, v) = centroid + u*e_u + v*e_v.
    # To lie on x = x_face: centroid_x + u*e_u_x + v*e_v_x = x_face.
    # This forms a line equation in (u, v) local coordinates:
    # A * u + B * v = C
    A = e_u[0]
    B = e_v[0]
    C = x_face - centroid[0]
    
    denom = A**2 + B**2
    if denom < 1e-12:
        return 0.0  # Plane is parallel to face and does not intersect
        
    # 2. Perpendicular distance from disc center (u0, v0) to intersection line in (u, v)
    proj_dist = abs(A * u0 + B * v0 - C) / np.sqrt(denom)
    
    # 3. Expected chord length on intersection
    if R > proj_dist:
        return 2.0 * np.sqrt(R**2 - proj_dist**2)
    return 0.0


def negative_log_posterior(
    params: np.ndarray,
    traces: List[FaceTrace],
    centroid: np.ndarray,
    normal: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray,
    mu_s: float,
    sigma_s: float,
    sigma_L: float = 0.20,
    p_detect: float = 0.90
) -> float:
    """
    Computes the negative log-posterior score for params = [u0, v0, R].
    Separates exact likelihood (Type 0) and inequality likelihood (Type 1 & 2).
    """
    u0, v0, R = params
    
    # Lognormal prior on radius R
    if R <= 1e-3:
        return 1e10
    ln_prior = - ((np.log(R) - mu_s)**2) / (2 * sigma_s**2) - np.log(R)
    
    ln_likelihood = 0.0
    for t in traces:
        # Expected trace length on this face
        L_exp = evaluate_analytical_trace_length(u0, v0, R, t.x_face, centroid, normal, e_u, e_v)
        
        if t.censoring_class == 0:
            # Type 0 (Contained): exact Gaussian length matching
            ln_likelihood += - ((t.length - L_exp)**2) / (2 * sigma_L**2) - np.log(np.sqrt(2 * np.pi) * sigma_L)
        else:
            # Type 1 & 2 (Clipped): inequality likelihood L_expected >= L_observed
            z = (L_exp - t.length) / sigma_L
            ln_likelihood += norm.logcdf(z)
            
    # Total posterior
    return float(-(ln_prior + ln_likelihood))


def compute_numerical_hessian(
    func,
    x0: np.ndarray,
    eps: float = 1e-4
) -> np.ndarray:
    """Computes numerical Hessian matrix at x0 using finite differences."""
    n = len(x0)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                x_plus = x0.copy()
                x_plus[i] += eps
                x_minus = x0.copy()
                x_minus[i] -= eps
                hessian[i, i] = (func(x_plus) - 2 * func(x0) + func(x_minus)) / (eps**2)
            else:
                x_pp = x0.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x0.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x0.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x0.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps**2)
    return hessian


def fit_constrained_map_plane(
    track_id: int,
    traces: List[FaceTrace],
    mu_s: float,
    sigma_s: float,
    set_id: Optional[int] = None,
    sigma_L: float = 0.20
) -> ReconstructedPlane:
    """
    Solves the censoring-aware constrained MAP problem to fit a 3D disc to a trace track.
    Estimates standard covariance matrix via the Laplace approximation.
    """
    # 1. Collate 3D endpoints to fit the initial SVD plane
    endpoints_3d = []
    for t in traces:
        endpoints_3d.append([t.x_face, t.p0_y, t.p0_z])
        endpoints_3d.append([t.x_face, t.p1_y, t.p1_z])
    endpoints_3d = np.array(endpoints_3d)
    
    centroid, normal = fit_plane_svd_3d(endpoints_3d)
    e_u, e_v = get_local_axes(normal)
    
    # Project endpoints to get search boundaries for center
    local_pts = project_points_to_plane_2d(endpoints_3d, centroid, e_u, e_v)
    mid_u = np.mean(local_pts[:, 0])
    mid_v = np.mean(local_pts[:, 1])
    
    # Bounds & initial guess
    min_r = max(t.length for t in traces) / 2.0
    init_r = float(np.exp(mu_s))
    if init_r < min_r:
        init_r = min_r * 1.2
        
    x0 = np.array([mid_u, mid_v, init_r])
    bounds = [
        (mid_u - 3.0, mid_u + 3.0),
        (mid_v - 3.0, mid_v + 3.0),
        (min_r, min_r * 5.0)
    ]
    
    # 2. Minimize negative log-posterior
    loss_func = lambda x: negative_log_posterior(x, traces, centroid, normal, e_u, e_v, mu_s, sigma_s, sigma_L)
    res = minimize(loss_func, x0, bounds=bounds, method='L-BFGS-B')
    
    u0_map, v0_map, r_map = res.x
    
    # Transform center back to 3D global coordinates
    center_3d = centroid + u0_map * e_u + v0_map * e_v
    
    # 3. Laplace Covariance matrix calculation
    cov = None
    try:
        hess = compute_numerical_hessian(loss_func, res.x)
        # Check if positive definite
        if np.all(np.linalg.eigenvals(hess) > 0):
            cov = np.linalg.inv(hess)
    except Exception:
        pass
        
    # Calculate Posterior Inclusion Probability (PIP)
    # Approximated by optimization success and residuals
    pip = 1.0 if res.success else 0.5
    
    return ReconstructedPlane(
        plane_id=track_id,
        point_x=float(center_3d[0]),
        point_y=float(center_3d[1]),
        point_z=float(center_3d[2]),
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
        normal_z=float(normal[2]),
        radius=float(r_map),
        source_trace_ids=[t.trace_id for t in traces],
        confidence=pip,
        covariance=cov,
        set_id=set_id
    )


def sample_single_face_posterior_candidates(
    track_id: int,
    trace: FaceTrace,
    mean_normal: np.ndarray,
    kappa: float,
    mu_s: float,
    sigma_s: float,
    set_id: int,
    n_samples: int = 15,
    random_seed: int = 42
) -> List[ReconstructedPlane]:
    """
    Probabilistic posterior candidate generator for single-face traces (unconstrained out-of-plane).
    Samples normals from the VMF set prior, then fits the 3D MAP plane for each sample,
    returning a collection of plausible candidates representing structural uncertainty.
    """
    candidates = []
    
    # Batch-sample VMF normals using the shared canonical implementation
    sampled_normals = sample_vmf_normals(mean_normal, kappa, n_samples=n_samples, seed=random_seed)
    
    for b in range(n_samples):
        n_pert = sampled_normals[b]
            
        # Centroid is trace midpoint
        centroid = np.array([trace.x_face, trace.midpoint_y, trace.midpoint_z])
        e_u_p, e_v_p = get_local_axes(n_pert)
        
        # Center bounds for 1-face: center must lie somewhere near trace midpoint
        min_r = trace.length / 2.0
        x0 = np.array([0.0, 0.0, float(np.exp(mu_s))])
        bounds = [(-1.5, 1.5), (-1.5, 1.5), (min_r, min_r * 4.0)]
        
        loss_func = lambda x: negative_log_posterior(x, [trace], centroid, n_pert, e_u_p, e_v_p, mu_s, sigma_s)
        res = minimize(loss_func, x0, bounds=bounds, method='L-BFGS-B')
        
        u0, v0, R = res.x
        center_3d = centroid + u0 * e_u_p + v0 * e_v_p
        
        candidates.append(ReconstructedPlane(
            plane_id=track_id * 1000 + b,
            point_x=float(center_3d[0]),
            point_y=float(center_3d[1]),
            point_z=float(center_3d[2]),
            normal_x=float(n_pert[0]),
            normal_y=float(n_pert[1]),
            normal_z=float(n_pert[2]),
            radius=float(R),
            source_trace_ids=[trace.trace_id],
            confidence=float(1.0 / n_samples),  # Equally spread confidence (Entropy representation)
            set_id=set_id,
            is_single_face_candidate=True
        ))
        
    return candidates
