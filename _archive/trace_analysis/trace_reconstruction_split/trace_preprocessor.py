"""
[Phase 2: Preprocessing and Doubled-Angle 2D Clustering]
Implements endpoint boundary-proximity censoring classification,
and 2D K-Means clustering on a doubled-angle representation with pure-numpy BIC model selection.
Ensures 100% dependency-free portability.
"""
import numpy as np
from typing import List, Tuple, Optional
from .trace_types import FaceTrace, ExcavationFace


def compute_point_to_polygon_distance(p_y: float, p_z: float, poly_yz: np.ndarray) -> float:
    """
    Computes the shortest perpendicular/vertex distance from 2D point (p_y, p_z)
    to a closed polygon defined by poly_yz (Shape N x 2).
    """
    if len(poly_yz) == 0:
        return float('inf')
        
    min_dist = float('inf')
    n_points = len(poly_yz)
    
    # Loop over all segments of the closed polygon
    for i in range(n_points):
        v0 = poly_yz[i]
        v1 = poly_yz[(i + 1) % n_points]
        
        # Segment vector
        d = v1 - v0
        len_sq = np.sum(d**2)
        
        if len_sq < 1e-12:
            # Degenerate segment
            dist = np.sqrt((p_y - v0[0])**2 + (p_z - v0[1])**2)
        else:
            # Projection factor t bounded to [0, 1]
            t = ((p_y - v0[0]) * d[0] + (p_z - v0[1]) * d[1]) / len_sq
            t = np.clip(t, 0.0, 1.0)
            
            # Closest point on segment
            closest = v0 + t * d
            dist = np.sqrt((p_y - closest[0])**2 + (p_z - closest[1])**2)
            
        if dist < min_dist:
            min_dist = dist
            
    return float(min_dist)


def classify_censoring(traces: List[FaceTrace], face: ExcavationFace, tolerance: float = 0.10):
    """
    Classifies censoring class (Type 0: Contained, Type 1: One-end clipped, Type 2: Both-end clipped)
    by checking whether the trace endpoints lie near the tunnel polygon boundary within tolerance.
    Modifies traces in-place.
    """
    poly = face.tunnel_polygon_yz
    for t in traces:
        if t.face_id != face.face_id:
            continue
            
        dist0 = compute_point_to_polygon_distance(t.p0_y, t.p0_z, poly)
        dist1 = compute_point_to_polygon_distance(t.p1_y, t.p1_z, poly)
        
        touch0 = dist0 <= tolerance
        touch1 = dist1 <= tolerance
        
        if touch0 and touch1:
            t.censoring_class = 2
        elif touch0 or touch1:
            t.censoring_class = 1
        else:
            t.censoring_class = 0


def kmeans_pure_numpy(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pure NumPy implementation of K-Means clustering.
    Returns: (centroids, labels, RSS)
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    
    if n_samples <= k:
        # Trivial clustering when samples are few
        labels = np.arange(n_samples) % k
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            idx = (labels == i)
            if np.any(idx):
                centroids[i] = np.mean(X[idx], axis=0)
        rss = 0.0
        for i in range(n_samples):
            rss += np.sum((X[i] - centroids[labels[i]])**2)
        return centroids, labels, rss
        
    # K-Means++ initialization
    centroids = [X[rng.choice(n_samples)]]
    for _ in range(1, k):
        dists = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
        probs = dists / (np.sum(dists) + 1e-12)
        centroids.append(X[rng.choice(n_samples, p=probs)])
    centroids = np.array(centroids)
    
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        old_labels = labels.copy()
        
        # Expectation step: assign to nearest centroid
        for i, x in enumerate(X):
            labels[i] = np.argmin(np.sum((centroids - x)**2, axis=1))
            
        # Maximization step: recompute centroids
        for j in range(k):
            idx = (labels == j)
            if np.any(idx):
                centroids[j] = np.mean(X[idx], axis=0)
                
        if np.all(labels == old_labels):
            break
            
    # Compute RSS (Residual Sum of Squares)
    rss = 0.0
    for i, x in enumerate(X):
        rss += np.sum((x - centroids[labels[i]])**2)
        
    return centroids, labels, rss


def evaluate_kmeans_bic(X: np.ndarray, k: int, labels: np.ndarray, rss: float) -> float:
    """
    Computes analytical BIC score for K-Means clustering:
    We assume clusters are spherical Gaussians.
    """
    N, D = X.shape
    if N <= k:
        return float('inf')
        
    # Variance MLE estimate
    variance = rss / (N * D + 1e-9)
    variance = max(1e-9, variance)
    
    # Log-likelihood of data
    log_likelihood = - (N * D / 2.0) * np.log(2.0 * np.pi * variance) - (N * D / 2.0)
    
    # Free parameters: k * D centroids, and (k-1) mixture weights
    n_params = k * D + (k - 1)
    
    return float(np.log(N) * n_params - 2.0 * log_likelihood)


def cluster_axial_traces_doubled_gmm(
    traces: List[FaceTrace],
    max_k: int = 4,
    random_seed: int = 42
) -> int:
    """
    Clusters trace orientations using GMM-equivalent K-Means on the Doubled-Angle representation:
    x = cos(2*theta), y = sin(2*theta)
    
    Uses BIC (Bayesian Information Criterion) to automatically select the optimal number of sets K.
    Assigns set_id to traces in-place (starting from 1).
    
    Returns the optimal set count K.
    
    Academic Note:
    Circular von Mises Mixture Models are theoretically more natural for circular directional data,
    but the Doubled-Angle Cartesian projection + K-Means/GMM provides robust convergence and analytical BIC stability.
    """
    if not traces:
        return 0
        
    angles = np.array([t.orientation_2d for t in traces])
    
    # Project axial angles to the unit circle via doubled angle
    X = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)])
    
    best_bic = float('inf')
    best_labels = None
    best_k = 1
    
    # Restrict max_k based on total trace sample size to prevent over-fitting on sparse samples
    actual_max_k = min(max_k, max(1, len(traces) // 5))
    actual_max_k = max(1, actual_max_k)
    
    for k in range(1, actual_max_k + 1):
        _, labels, rss = kmeans_pure_numpy(X, k, seed=random_seed)
        bic = evaluate_kmeans_bic(X, k, labels, rss)
        
        if bic < best_bic:
            best_bic = bic
            best_labels = labels
            best_k = k
            
    # Assign set_id (1-indexed) in-place
    for t, label in zip(traces, best_labels):
        t.set_id = int(label + 1)
        
    return best_k
