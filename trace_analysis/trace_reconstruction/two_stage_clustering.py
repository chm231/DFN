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


def calculate_kappa_tensor_aligned(normals: np.ndarray, mean_dir: np.ndarray = None) -> float:
    """
    Calculates the Fisher concentration parameter K using the M.L.M. simplified formula:
    K ≈ (M - 1) / (M - |r_n|)
    where unit normal vectors are aligned dynamically to the dominant axis of the orientation tensor.
    """
    M = len(normals)
    if M <= 1:
        return 10.0  # Safe default concentration

    # 1. Force unit vector normalization with numerical safety
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    n = normals / norms_safe
    zero_mask = (norms.ravel() <= 1e-12)
    if np.any(zero_mask):
        n[zero_mask] = np.array([1.0, 0.0, 0.0])

    # 2. Compute orientation tensor: T = (1/M) * sum(n_i * n_i^T)
    T = np.dot(n.T, n) / M

    # 3. Solve eigenvalues and eigenvectors to find the dominant axis (PCA)
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    dominant_axis = eigenvectors[:, -1]

    # 4. Dynamic hemisphere alignment: flip n_i if dot(n_i, dominant_axis) < 0
    dots = np.dot(n, dominant_axis)
    flip_mask = (dots < 0)
    aligned_n = n.copy()
    aligned_n[flip_mask] = -aligned_n[flip_mask]

    # 5. Resultant vector summation & Fisher Kappa computation
    r_n = np.sum(aligned_n, axis=0)
    r_n_mag = np.linalg.norm(r_n)

    denominator = M - r_n_mag
    if denominator < 1e-6:
        kappa = 500.0
    else:
        kappa = (M - 1) / denominator

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
            
        kappa = calculate_kappa_tensor_aligned(set_normals, mean_vec)
        set_stats[set_id] = (mean_vec, kappa)
        
    return best_k, set_stats
