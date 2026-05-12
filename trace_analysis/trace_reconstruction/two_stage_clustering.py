"""
[Phase 3: Two-Stage Spherical Set Clustering]
Refines 3D normal orientation sets using hemispherical projection,
and unit-sphere normalized pure-numpy K-Means with BIC model selection.
Ensures 100% dependency-free portability.
"""
import numpy as np
from typing import List, Tuple, Dict
from .trace_types import ReconstructedPlane
from .trace_preprocessor import kmeans_pure_numpy, evaluate_kmeans_bic


def map_to_upper_hemisphere(normals: np.ndarray) -> np.ndarray:
    """
    Maps 3D normals to the upper hemisphere (normal_x >= 0)
    to treat axial directional data correctly.
    """
    mapped = normals.copy()
    for i in range(len(mapped)):
        n = mapped[i]
        # Ensure x-coordinate is positive. If very close to 0, use y or z.
        if n[0] < -1e-7:
            mapped[i] = -n
        elif abs(n[0]) <= 1e-7:
            if n[1] < -1e-7:
                mapped[i] = -n
            elif abs(n[1]) <= 1e-7:
                if n[2] < 0:
                    mapped[i] = -n
    return mapped


def estimate_vmf_concentration(normals: np.ndarray, mean: np.ndarray) -> float:
    """
    Estimates the Fisher concentration parameter kappa for a set of unit vectors
    relative to their spherical mean direction.
    """
    if len(normals) <= 1:
        return 10.0  # Safe default concentration
        
    # Mean resultant length R_bar
    cos_angles = np.dot(normals, mean)
    R_bar = np.mean(cos_angles)
    R_bar = np.clip(R_bar, 0.0, 0.999) # Prevent divide by zero/inf
    
    # Standard approximation of concentration parameter kappa (VMF)
    if R_bar < 0.05:
        kappa = 2.0 * R_bar
    elif R_bar > 0.99:
        kappa = 1.0 / (1.0 - R_bar)
    else:
        kappa = (R_bar * (3.0 - R_bar**2)) / (1.0 - R_bar**2)
        
    return float(np.clip(kappa, 1.0, 500.0))


def cluster_reconstructed_normals_3d(
    planes: List[ReconstructedPlane],
    max_k: int = 4,
    random_seed: int = 42
) -> Tuple[int, Dict[int, Tuple[np.ndarray, float]]]:
    """
    Performs the second stage of clustering: groups the reconstructed 3D normals of
    multi-face planes on the hemisphere using a unit-sphere normalized pure-numpy K-Means.
    
    Uses BIC model selection to identify the optimal number of 3D sets K.
    Modifies planes in-place by updating their set_id.
    
    Returns:
    - Optimal set count K
    - Set-wise statistics: Dictionary mapping set_id -> (mean_normal, concentration_kappa)
    """
    if not planes:
        return 0, {}
        
    normals_raw = np.array([[p.normal_x, p.normal_y, p.normal_z] for p in planes])
    X = map_to_upper_hemisphere(normals_raw)
    
    best_bic = float('inf')
    best_labels = None
    best_k = 1
    
    # Restrict K based on sample size
    actual_max_k = min(max_k, max(1, len(planes) // 3))
    actual_max_k = max(1, actual_max_k)
    
    for k in range(1, actual_max_k + 1):
        _, labels, rss = kmeans_pure_numpy(X, k, seed=random_seed)
        bic = evaluate_kmeans_bic(X, k, labels, rss)
        
        if bic < best_bic:
            best_bic = bic
            best_labels = labels
            best_k = k
            
    # Update set_id (1-indexed) in-place
    for p, label in zip(planes, best_labels):
        p.set_id = int(label + 1)
        
    # Calculate set-wise stats (mean direction, VMF kappa)
    set_stats = {}
    for label in range(best_k):
        set_id = label + 1
        idx = (best_labels == label)
        set_normals = X[idx]
        
        if len(set_normals) == 0:
            continue
            
        # Spherical mean (vector sum, then normalized)
        sum_vec = np.sum(set_normals, axis=0)
        sum_len = np.linalg.norm(sum_vec)
        if sum_len > 1e-9:
            mean_vec = sum_vec / sum_len
        else:
            mean_vec = np.array([1.0, 0.0, 0.0])
            
        kappa = estimate_vmf_concentration(set_normals, mean_vec)
        set_stats[set_id] = (mean_vec, kappa)
        
    return best_k, set_stats
