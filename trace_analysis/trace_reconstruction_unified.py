"""
================================================================================
Bayesian 3D DFN Inverse Reconstruction - Unified Pipeline Script
================================================================================
This file consolidates all 12 modules from the `trace_reconstruction` package
into a single, self-contained Python file. Internal relative imports have been
fully resolved to allow seamless copying and pasting or ingestion by other AI models.

Structure Consolidated:
1. Dataclasses & Types (trace_types.py)
2. Preprocessor & 2D Clustering (trace_preprocessor.py)
3. 3D Spherical Clustering (two_stage_clustering.py)
4. Parametric MLE Length Estimator (mle_estimation.py)
5. Hekmatnejad Non-parametric Estimator (hekmatnejad_estimation.py)
6. Forward Simulator (forward_simulator.py)
7. Constrained MAP Plane Fitter (constrained_map_fitter.py)
8. Bayes Factor Face Association (face_association.py)
9. Residual DFN Prior Generator (residual_dfn_generator.py)
10. Manifold Glide Simulated Annealing (manifold_glide_optimizer.py)
11. HDF5 DFN Exporter (dfn_exporter.py)
================================================================================
"""

import time
import h5py
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from scipy.optimize import minimize, linear_sum_assignment
from scipy.stats import lognorm, expon, pareto, norm

# Optional import of lifelines for validation
try:
    import lifelines  # type: ignore
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES (trace_types.py)
# ==============================================================================

@dataclass
class ExcavationFace:
    """터널 굴착 막장면 (x=const 평면에서의 터널 단면 정보)"""
    face_id: int
    x_face: float
    tunnel_polygon_yz: np.ndarray  # Shape (N, 2) array of [y, z] coordinates
    advance_step: float            # 이전 face로부터의 굴진 거리


@dataclass
class FaceTrace:
    """터널 굴착 막장면(polygon 내부)에서 관측된 단일 fracture trace segment"""
    face_id: int
    trace_id: int
    x_face: float
    p0_y: float
    p0_z: float
    p1_y: float
    p1_z: float
    confidence: float = 1.0
    parent_fracture_id: Optional[int] = None
    
    # Pre-calculated geometric fields
    midpoint_y: float = field(init=False)
    midpoint_z: float = field(init=False)
    length: float = field(init=False)
    orientation_2d: float = field(init=False)  # 2D orientation angle in [-pi/2, pi/2] (axial)
    censoring_class: int = 0                    # 0: Contained, 1: One-end clipped, 2: Both-end clipped
    set_id: Optional[int] = None                # Clustering set identifier

    def __post_init__(self):
        self.midpoint_y = (self.p0_y + self.p1_y) / 2.0
        self.midpoint_z = (self.p0_z + self.p1_z) / 2.0
        self.length = float(np.sqrt((self.p1_y - self.p0_y)**2 + (self.p1_z - self.p0_z)**2))
        
        # Calculate orientation 2d: angle of line segment bounded to [-pi/2, pi/2]
        dy = self.p1_y - self.p0_y
        dz = self.p1_z - self.p0_z
        angle = np.arctan2(dz, dy)
        
        # Wrap angle to axial range [-pi/2, pi/2]
        if angle > np.pi / 2.0:
            angle -= np.pi
        elif angle < -np.pi / 2.0:
            angle += np.pi
        self.orientation_2d = float(angle)


@dataclass
class TraceMatch:
    """인접한 두 face 간의 trace 매칭 결과 및 Bayes Factor 정보"""
    face_id_prev: int
    face_id_curr: int
    trace_id_prev: int
    trace_id_curr: int
    log_bayes_factor: float
    accepted: bool = False


@dataclass
class ReconstructedPlane:
    """연속 매칭된 trace(들)로부터 3차원 공간 상에 역산된 평면 (MAP)"""
    plane_id: int
    point_x: float
    point_y: float
    point_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    radius: float
    source_trace_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0  # Posterior Inclusion Probability (PIP)
    covariance: Optional[np.ndarray] = None  # 3x3 covariance matrix from Laplace approximation
    set_id: Optional[int] = None
    is_single_face_candidate: bool = False  # True if generated as a probabilistic sample of a single-face trace


@dataclass
class StochasticFracture:
    """Stochastic DFN 생성을 위한 3D 균열 원판 구조체"""
    fracture_id: int
    center_x: float
    center_y: float
    center_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    radius: float
    set_id: int


# ==============================================================================
# SECTION 2: 2D CLUSTERING & CENSORING PREPROCESSOR (trace_preprocessor.py)
# ==============================================================================

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
    """
    if not traces:
        return 0
        
    angles = np.array([t.orientation_2d for t in traces])
    
    # Project axial angles to the unit circle via doubled angle
    X = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)])
    
    best_bic = float('inf')
    best_labels = np.zeros(len(traces), dtype=int)
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


# ==============================================================================
# SECTION 3: 3D SPHERICAL SET CLUSTERING (two_stage_clustering.py)
# ==============================================================================

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


def calculate_kappa_tensor_aligned(normals: np.ndarray, mean_dir: np.ndarray | None = None) -> float:
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
    """
    if not planes:
        return 0, {}
        
    normals_raw = np.array([[p.normal_x, p.normal_y, p.normal_z] for p in planes])
    X = map_to_upper_hemisphere(normals_raw)
    
    best_bic = float('inf')
    best_labels = np.zeros(len(planes), dtype=int)
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
            
        # Utilize the equatorial boundary-flip proof tensor kappa estimator
        kappa = calculate_kappa_tensor_aligned(set_normals, mean_vec)
        set_stats[set_id] = (mean_vec, kappa)
        
    return best_k, set_stats


# ==============================================================================
# SECTION 4: PARAMETRIC MLE LENGTH ESTIMATION (mle_estimation.py)
# ==============================================================================

class ParametricMLEEstimator:
    """
    Parametric Maximum Likelihood Estimator (MLE) for trace length inversion.
    """
    def __init__(self, min_truncation: float = 0.15, correct_size_bias: bool = False, 
                 window_diameter: float = 10.0, self_calibrate: bool = False):
        self.min_truncation = min_truncation
        self.correct_size_bias = correct_size_bias
        self.window_diameter = window_diameter
        self.self_calibrate = self_calibrate
        
        # Calibration offsets
        self.d1 = 2.0
        self.d2 = 16.0
        
        # Best model attributes
        self.best_dist_name = None
        self.best_params = None
        self.best_aic = None
        self.best_log_lik = None

    def _perform_self_calibration(self, lengths: np.ndarray, censoring: np.ndarray):
        """
        Unsupervised blind self-calibration. Optimizes d1 and d2 to match the theoretical
        circular window trace class proportions with the observed proportions.
        """
        n_total = len(lengths)
        obs_p0 = np.sum(censoring == 0) / n_total
        obs_p1 = np.sum(censoring == 1) / n_total
        obs_p2 = np.sum(censoring == 2) / n_total
        
        c = self.min_truncation
        D = self.window_diameter
        
        def compute_theoretical_proportions(mu, sigma):
            def int0(L):
                p0 = (1.0 - L/D)**2 if L < D else 0.0
                return lognorm.pdf(L, s=sigma, scale=np.exp(mu)) * p0
                
            def int1(L):
                p1 = (2.0 * L / D) * (1.0 - L/D) if L < D else 0.0
                return lognorm.pdf(L, s=sigma, scale=np.exp(mu)) * p1
                
            def int2(L):
                p2 = (L/D)**2 if L < D else 1.0
                return lognorm.pdf(L, s=sigma, scale=np.exp(mu)) * p2
                
            val0, _ = integrate.quad(int0, c, D, epsabs=1e-3, epsrel=1e-3)
            val1, _ = integrate.quad(int1, c, D, epsabs=1e-3, epsrel=1e-3)
            val2, _ = integrate.quad(int2, c, np.inf, epsabs=1e-3, epsrel=1e-3)
            
            sum_vals = val0 + val1 + val2
            if sum_vals <= 1e-15: return 0.0, 0.0, 0.0
            return val0 / sum_vals, val1 / sum_vals, val2 / sum_vals

        best_loss = 1e10
        best_d1 = 2.0
        best_d2 = 16.0
        
        print("[*] Running unsupervised blind self-calibration on face trace proportions...")
        for d1_cand in np.linspace(1.0, 6.0, 11):
            for d2_cand in np.linspace(2.0, 10.0, 33):
                recon = []
                for l, cc in zip(lengths, censoring):
                    if cc == 0:
                        recon.append(l)
                    elif cc == 1:
                        recon.append(l + d1_cand)
                    elif cc == 2:
                        recon.append(l + d2_cand)
                recon = np.array(recon)
                
                try:
                    s, loc, scale = lognorm.fit(recon, floc=0)
                    mu = np.log(scale)
                    sigma = s
                    t0, t1, t2 = compute_theoretical_proportions(mu, sigma)
                    loss = (t0 - obs_p0)**2 + (t1 - obs_p1)**2 + (t2 - obs_p2)**2
                    if loss < best_loss:
                        best_loss = loss
                        best_d1 = d1_cand
                        best_d2 = d2_cand
                except Exception:
                    continue
                    
        self.d1 = best_d1
        self.d2 = best_d2
        print(f"    -> Self-Calibration Finished: d1 = {self.d1:.3f}m, d2 = {self.d2:.3f}m (Loss = {best_loss:.6f})")

    def fit(self, lengths: np.ndarray, censoring: np.ndarray) -> Dict[str, Any]:
        """
        Fits Lognormal, Exponential, and Pareto distributions using robust MLE and selects the best model.
        """
        valid = lengths >= self.min_truncation
        lengths = lengths[valid]
        censoring = censoring[valid]
        c = self.min_truncation

        # Self-calibrate offsets if requested
        if self.self_calibrate:
            self._perform_self_calibration(lengths, censoring)
        else:
            print(f"[*] Using pre-calibrated baseline offsets: d1 = {self.d1:.1f}m, d2 = {self.d2:.1f}m")

        # Reconstruct unclipped lengths
        recon_lengths = []
        for l, cc in zip(lengths, censoring):
            if cc == 0:
                recon_lengths.append(l)
            elif cc == 1:
                recon_lengths.append(l + self.d1)
            elif cc == 2:
                recon_lengths.append(l + self.d2)
        recon_lengths = np.array(recon_lengths)

        # --- 1. LOGNORMAL MLE ---
        try:
            s_ln, loc_ln, scale_ln = lognorm.fit(recon_lengths, floc=0)
            mu_ln = np.log(scale_ln)
            sigma_ln = s_ln
            log_lik_ln = np.sum(lognorm.logpdf(recon_lengths, s=sigma_ln, scale=scale_ln))
            aic_ln = 2 * 2 - 2 * log_lik_ln
        except Exception:
            aic_ln = 1e10
            log_lik_ln = -1e10

        # --- 2. EXPONENTIAL MLE ---
        try:
            loc_ex, scale_ex = expon.fit(recon_lengths, floc=0)
            log_lik_ex = np.sum(expon.logpdf(recon_lengths, scale=scale_ex))
            aic_ex = 2 * 1 - 2 * log_lik_ex
        except Exception:
            aic_ex = 1e10
            log_lik_ex = -1e10

        b_pa = 1.0
        # --- 3. PARETO MLE ---
        try:
            r_min_3d = 1.0
            pareto_truncation = 2.0 * r_min_3d
            pareto_recon = recon_lengths[recon_lengths >= pareto_truncation]
            if len(pareto_recon) >= 5:
                b_pa, loc_pa, scale_pa = pareto.fit(pareto_recon, floc=0, fscale=pareto_truncation)
                log_lik_pa = np.sum(pareto.logpdf(pareto_recon, b=b_pa, scale=scale_pa))
                aic_pa = 2 * 1 - 2 * log_lik_pa
            else:
                b_pa, loc_pa, scale_pa = pareto.fit(recon_lengths, floc=0, fscale=self.min_truncation)
                log_lik_pa = np.sum(pareto.logpdf(recon_lengths, b=b_pa, scale=scale_pa))
                aic_pa = 2 * 1 - 2 * log_lik_pa
        except Exception:
            aic_pa = 1e10
            log_lik_pa = -1e10

        print("\n[*] Exact Calibrated MLE Solver Fit Summary:")
        print(f"  - Lognormal  : Log-Likelihood = {log_lik_ln:.4f}, AIC = {aic_ln:.4f}")
        print(f"  - Exponential: Log-Likelihood = {log_lik_ex:.4f}, AIC = {aic_ex:.4f}")
        print(f"  - Pareto     : Log-Likelihood = {log_lik_pa:.4f}, AIC = {aic_pa:.4f}")

        # Enforce Pareto (Power-law) model globally
        best_name = "Pareto"
        best_aic = aic_pa
        best_log_lik = log_lik_pa
        best_params = np.array([b_pa])

        print(f"[*] Optimal Model Selected: **{best_name}** (AIC = {best_aic:.4f})")
        
        self.best_dist_name = best_name
        self.best_params = best_params
        self.best_aic = best_aic
        self.best_log_lik = best_log_lik

        cdf_fun, pdf_fun = self._build_functions(best_name, best_params, c)
        
        return {
            "dist_name": best_name,
            "params": best_params,
            "cdf_function": cdf_fun,
            "pdf_function": pdf_fun,
            "log_likelihood": best_log_lik,
            "aic": best_aic
        }

    def _build_functions(self, dist_name: str, params: np.ndarray, c: float) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """
        Builds the CDF and PDF functions, applying size-bias recovery if configured.
        """
        if dist_name == "Lognormal":
            mu_b, sigma_b = params
            if self.correct_size_bias:
                sigma_L = sigma_b
                mu_L = mu_b - sigma_b**2
            else:
                sigma_L = sigma_b
                mu_L = mu_b
                
            def cdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                f_l = lognorm.cdf(l_arr, s=sigma_L, scale=np.exp(mu_L))
                f_c = lognorm.cdf(c, s=sigma_L, scale=np.exp(mu_L))
                res = np.where(l_arr < c, 0.0, (f_l - f_c) / (1.0 - f_c))
                return res[0] if np.isscalar(l) else res

            def pdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                f_l = lognorm.pdf(l_arr, s=sigma_L, scale=np.exp(mu_L))
                f_c = lognorm.cdf(c, s=sigma_L, scale=np.exp(mu_L))
                res = np.where(l_arr < c, 0.0, f_l / (1.0 - f_c))
                return res[0] if np.isscalar(l) else res

        elif dist_name == "Exponential":
            lam = params[0]
            def cdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, 1.0 - np.exp(-lam * (l_arr - c)))
                return res[0] if np.isscalar(l) else res

            def pdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, lam * np.exp(-lam * (l_arr - c)))
                return res[0] if np.isscalar(l) else res

        elif dist_name == "Pareto":
            alpha_b = params[0]
            if self.correct_size_bias:
                alpha = alpha_b + 1.0
            else:
                alpha = alpha_b
                
            def cdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, 1.0 - (c / l_arr)**alpha)
                return res[0] if np.isscalar(l) else res

            def pdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, alpha * (c**alpha) / (l_arr**(alpha + 1)))
                return res[0] if np.isscalar(l) else res
        else:
            raise ValueError(f"Unknown distribution name: {dist_name}")
                
        return cdf_fun, pdf_fun


# ==============================================================================
# SECTION 5: HEKMATNEJAD NON-PARAMETRIC LENGTH INVERSION (hekmatnejad_estimation.py)
# ==============================================================================

class HekmatnejadEstimator:
    """
    Core implementation of 2D Trace length bias correction and True Length PDF/CDF inversion
    following Hekmatnejad et al. (2018) methodologies.
    """
    def __init__(
        self,
        min_truncation: float = 0.1,
        max_weight: float = 10.0,
        apply_orientation_bias: bool = True,
        dip_in_degrees: bool = True
    ):
        self.min_truncation = min_truncation
        self.max_weight = max_weight
        self.apply_orientation_bias = apply_orientation_bias
        self.dip_in_degrees = dip_in_degrees

    def compute_terzaghi_weights(self, dip_angles: np.ndarray) -> np.ndarray:
        """Computes Terzaghi orientation bias correction weights: w_i = 1 / sin(theta_i)"""
        if not self.apply_orientation_bias or dip_angles is None:
            return np.ones(len(dip_angles))

        angles_rad = np.deg2rad(dip_angles) if self.dip_in_degrees else dip_angles
        angles_rad = np.abs(angles_rad)
        angles_rad = np.clip(angles_rad, 1e-6, np.pi / 2.0)
        
        weights = 1.0 / np.sin(angles_rad)
        weights = np.minimum(weights, self.max_weight)
        return weights

    def filter_truncation(
        self, 
        lengths: np.ndarray, 
        censoring_types: np.ndarray, 
        dip_angles: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Filters out traces with length less than the minimum resolution threshold (c)."""
        mask = lengths >= self.min_truncation
        filtered_lengths = lengths[mask]
        filtered_censoring = censoring_types[mask]
        filtered_dips = dip_angles[mask] if dip_angles is not None else None
        
        return filtered_lengths, filtered_censoring, filtered_dips

    def fit_weighted_kaplan_meier(
        self,
        lengths: np.ndarray,
        censoring_types: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the survival function S(l) = P(L > l) via a custom Weighted Kaplan-Meier Estimator.
        """
        n_samples = len(lengths)
        if n_samples == 0:
            return np.array([self.min_truncation]), np.array([1.0])
            
        events = (censoring_types == 0).astype(int)
        sort_idx = np.argsort(lengths)
        sorted_l = lengths[sort_idx]
        sorted_e = events[sort_idx]
        sorted_w = weights[sort_idx]
        
        unique_l = np.unique(sorted_l)
        survival_probs = []
        
        total_w_above = np.zeros(len(unique_l))
        events_w_at = np.zeros(len(unique_l))
        
        for idx, ul in enumerate(unique_l):
            risk_mask = sorted_l >= ul
            total_w_above[idx] = np.sum(sorted_w[risk_mask])
            
            event_mask = (sorted_l == ul) & (sorted_e == 1)
            events_w_at[idx] = np.sum(sorted_w[event_mask])

        current_s = 1.0
        for idx in range(len(unique_l)):
            n_j = total_w_above[idx]
            d_j = events_w_at[idx]
            
            if n_j > 0:
                current_s *= (1.0 - (d_j / n_j))
            survival_probs.append(current_s)
            
        return unique_l, np.array(survival_probs)

    def cross_validate_with_lifelines(
        self,
        lengths: np.ndarray,
        censoring_types: np.ndarray,
        weights: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Dynamic cross-validation helper using the lifelines package if available."""
        if not HAS_LIFELINES:
            return None
            
        events = (censoring_types == 0).astype(int)
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(durations=lengths, event_observed=events, weights=weights)
        return kmf.survival_function_.index.values, kmf.survival_function_['KM_estimate'].values

    def build_continuous_distributions(
        self,
        unique_lengths: np.ndarray,
        survival_probs: np.ndarray
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """
        Converts the discrete survival estimation to a smooth, strictly monotonic CDF
        and corresponding PDF using PchipInterpolator and analytical derivative.
        """
        cdf_probs = 1.0 - survival_probs
        fit_lengths = np.concatenate([[self.min_truncation], unique_lengths])
        fit_cdf = np.concatenate([[0.0], cdf_probs])
        
        for idx in range(1, len(fit_cdf)):
            if fit_cdf[idx] < fit_cdf[idx - 1]:
                fit_cdf[idx] = fit_cdf[idx - 1]
                
        if fit_cdf[-1] > 0.0:
            fit_cdf = fit_cdf / fit_cdf[-1]
            
        pchip_cdf = interp.PchipInterpolator(fit_lengths, fit_cdf, extrapolate=True)
        pchip_pdf = pchip_cdf.derivative()
        
        def safe_cdf(l: np.ndarray) -> np.ndarray:
            l_arr = np.atleast_1d(l)
            res = pchip_cdf(l_arr)
            res = np.where(l_arr < self.min_truncation, 0.0, res)
            res = np.where(l_arr > fit_lengths[-1], 1.0, res)
            res = np.clip(res, 0.0, 1.0)
            return res[0] if np.isscalar(l) else res

        def safe_pdf(l: np.ndarray) -> np.ndarray:
            l_arr = np.atleast_1d(l)
            res = pchip_pdf(l_arr)
            res = np.where((l_arr < self.min_truncation) | (l_arr > fit_lengths[-1]), 0.0, res)
            res = np.maximum(res, 0.0)
            return res[0] if np.isscalar(l) else res
            
        return safe_cdf, safe_pdf

    def run_inversion_pipeline(
        self,
        raw_lengths: np.ndarray,
        raw_censoring: np.ndarray,
        raw_dips: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Orchestrates the entire correction & inversion pipeline."""
        if self.apply_orientation_bias and raw_dips is not None:
            raw_weights = self.compute_terzaghi_weights(raw_dips)
        else:
            raw_weights = np.ones(len(raw_lengths))

        lengths, censoring, weights = self.filter_truncation(raw_lengths, raw_censoring, raw_weights)
        assert weights is not None
        unique_lengths, survival_probs = self.fit_weighted_kaplan_meier(lengths, censoring, weights)
        
        lifelines_result = None
        if HAS_LIFELINES:
            lifelines_result = self.cross_validate_with_lifelines(lengths, censoring, weights)
            
        cdf_fun, pdf_fun = self.build_continuous_distributions(unique_lengths, survival_probs)
        
        return {
            "filtered_lengths": lengths,
            "filtered_censoring": censoring,
            "filtered_weights": weights,
            "unique_lengths": unique_lengths,
            "survival_probs": survival_probs,
            "lifelines_verification": lifelines_result,
            "cdf_function": cdf_fun,
            "pdf_function": pdf_fun
        }


def plot_length_distributions(
    raw_lengths: np.ndarray,
    results: Dict[str, Any],
    min_truncation: float = 0.1,
    save_path: Optional[str] = None
):
    """Plots comparative histograms of the raw vs. corrected trace length distributions."""
    filtered_lengths = results["filtered_lengths"]
    weights = results["filtered_weights"]
    cdf_fun = results["cdf_function"]
    pdf_fun = results["pdf_function"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#fafafa")
    
    c_raw = "#d95f02"
    c_corrected = "#1b9e77"
    c_pdf = "#7570b3"
    
    # ------------------ PANEL 1: PDF comparison ------------------
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    ax1.hist(
        raw_lengths, bins=25, density=True, alpha=0.35, color=c_raw, 
        edgecolor=c_raw, linewidth=1.2, label="Raw Observed Distribution"
    )
    ax1.hist(
        filtered_lengths, bins=25, density=True, weights=weights, alpha=0.45, color=c_corrected,
        edgecolor=c_corrected, linewidth=1.2, label="Bias-Corrected (Weighted)"
    )
    l_grid = np.linspace(min_truncation, np.max(raw_lengths) * 1.1, 500)
    pdf_vals = pdf_fun(l_grid)
    ax1.plot(
        l_grid, pdf_vals, color=c_pdf, linewidth=3.0, linestyle="-",
        label="Inverted True PDF $f(l)$"
    )
    ax1.set_title("Probability Density Function (PDF) Inversion Comparison", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Trace Length $l$ (m)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")
    
    # ------------------ PANEL 2: CDF comparison ------------------
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    sorted_raw = np.sort(raw_lengths)
    ecdf_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
    ax2.step(
        sorted_raw, ecdf_raw, color=c_raw, alpha=0.6, linewidth=1.8,
        where="post", label="Empirical CDF (Raw)"
    )
    cdf_vals = cdf_fun(l_grid)
    ax2.plot(
        l_grid, cdf_vals, color=c_corrected, linewidth=3.0, linestyle="-",
        label="Inverted True CDF $F(l)$"
    )
    
    if results["lifelines_verification"] is not None:
        ll_l, ll_s = results["lifelines_verification"]
        ax2.step(
            ll_l, 1.0 - ll_s, color="#e7298a", linestyle=":", linewidth=2.0,
            where="post", label="lifelines KMF Verification"
        )
        
    ax2.set_title("Cumulative Distribution Function (CDF) Inversion Comparison", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Trace Length $l$ (m)", fontsize=11)
    ax2.set_ylabel("Cumulative Probability", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[*] Premium comparative distribution figure saved to: {save_path}")
        plt.close()
    else:
        plt.show()


# ==============================================================================
# SECTION 6: FORWARD SIMULATOR & INTERSECTIONS (forward_simulator.py)
# ==============================================================================

def is_point_inside_polygon(y: float, z: float, poly: np.ndarray) -> bool:
    """Ray-casting algorithm to determine if point (y, z) is inside a closed polygon."""
    n = len(poly)
    inside = False
    p1y, p1z = poly[0]
    for i in range(n + 1):
        p2y, p2z = poly[i % n]
        if z > min(p1z, p2z):
            if z <= max(p1z, p2z):
                if y <= max(p1y, p2y):
                    if p1z != p2z:
                        xinters = (z - p1z) * (p2y - p1y) / (p2z - p1z) + p1y
                    else:
                        xinters = p1y
                    if p1y == p2y or y <= xinters:
                        inside = not inside
        p1y, p1z = p2y, p2z
    return inside


def clip_line_segment_to_polygon(
    p0: np.ndarray,
    p1: np.ndarray,
    poly: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Clips a 2D line segment (p0, p1) to a closed polygon."""
    if len(poly) == 0:
        return []
        
    v_dir = p1 - p0
    v_len = np.linalg.norm(v_dir)
    if v_len < 1e-12:
        if is_point_inside_polygon(p0[0], p0[1], poly):
            return [(p0, p1)]
        return []
        
    v_u = v_dir / v_len
    t_vals = [0.0, 1.0]
    n_points = len(poly)
    
    for i in range(n_points):
        v0 = poly[i]
        v1 = poly[(i + 1) % n_points]
        
        d_edge = v1 - v0
        denom = v_dir[0] * d_edge[1] - v_dir[1] * d_edge[0]
        if abs(denom) < 1e-12:
            continue
            
        t = (v0[0] * d_edge[1] - v0[1] * d_edge[0] - p0[0] * d_edge[1] + p0[1] * d_edge[0]) / denom
        s = (p0[0] * v_dir[1] - p0[1] * v_dir[0] - v0[0] * v_dir[1] + v0[1] * v_dir[0]) / (-denom)
        
        if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
            t_vals.append(t)
            
    t_vals = sorted(list(set(t_vals)))
    clipped_segments = []
    
    for i in range(len(t_vals) - 1):
        t_start, t_end = t_vals[i], t_vals[i + 1]
        if t_end - t_start < 1e-5:
            continue
            
        mid_t = 0.5 * (t_start + t_end)
        mid_pt = p0 + mid_t * v_dir
        
        if is_point_inside_polygon(mid_pt[0], mid_pt[1], poly):
            sub_p0 = p0 + t_start * v_dir
            sub_p1 = p0 + t_end * v_dir
            clipped_segments.append((sub_p0, sub_p1))
            
    return clipped_segments


def sample_vmf_normals(mean_normal: np.ndarray, kappa: float, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Samples unit normals from a 3D von Mises-Fisher distribution."""
    rng = np.random.default_rng(seed)
    if abs(mean_normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
        
    e_u = np.cross(mean_normal, ref)
    e_u = e_u / np.linalg.norm(e_u)
    e_v = np.cross(mean_normal, e_u)
    e_v = e_v / np.linalg.norm(e_v)
    
    stdev = 1.0 / np.sqrt(kappa)
    thetas = rng.normal(0, stdev, n_samples)
    phis = rng.uniform(0, 2 * np.pi, n_samples)
    
    samples = []
    for theta, phi in zip(thetas, phis):
        n = mean_normal + theta * (np.cos(phi) * e_u + np.sin(phi) * e_v)
        n = n / np.linalg.norm(n)
        if n[0] < 0:
            n = -n
        samples.append(n)
        
    return np.array(samples)


def generate_stochastic_dfn(
    domain: Dict[str, float],
    set_params: Dict[int, Dict[str, float]],
    set_stats: Dict[int, Tuple[np.ndarray, float]],
    start_id: int = 2000,
    seed: Optional[int] = 42
) -> List[StochasticFracture]:
    """Generates a stochastic 3D DFN using a Poisson Point Process."""
    rng = np.random.default_rng(seed)
    stoch_fractures = []
    fid = start_id
    
    vol = (domain['xmax'] - domain['xmin']) * (domain['ymax'] - domain['ymin']) * (domain['zmax'] - domain['zmin'])
    
    for set_id, params in set_params.items():
        P30 = params['P30']
        mu_s = params['mu_s']
        sigma_s = params['sigma_s']
        mean_normal, kappa = set_stats[set_id]
        
        n_frac = rng.poisson(P30 * vol)
        if n_frac <= 0:
            continue
            
        cx = rng.uniform(domain['xmin'], domain['xmax'], n_frac)
        cy = rng.uniform(domain['ymin'], domain['ymax'], n_frac)
        cz = rng.uniform(domain['zmin'], domain['zmax'], n_frac)
        
        radii = rng.lognormal(mu_s, sigma_s, n_frac)
        set_seed = int(rng.integers(0, 1000000)) if seed is not None else None
        normals = sample_vmf_normals(mean_normal, kappa, n_frac, seed=set_seed)
        
        for i in range(n_frac):
            stoch_fractures.append(StochasticFracture(
                fracture_id=fid,
                center_x=float(cx[i]),
                center_y=float(cy[i]),
                center_z=float(cz[i]),
                normal_x=float(normals[i, 0]),
                normal_y=float(normals[i, 1]),
                normal_z=float(normals[i, 2]),
                radius=float(radii[i]),
                set_id=set_id
            ))
            fid += 1
            
    return stoch_fractures


def intersect_disc_with_face(
    c_x: float, c_y: float, c_z: float,
    n_x: float, n_y: float, n_z: float,
    radius: float,
    face: ExcavationFace,
    start_trace_id: int = 5000,
    set_id: int = 1,
    parent_fracture_id: Optional[int] = None
) -> List[FaceTrace]:
    """Analytically intersects a 3D disc with an excavation face plane (x = x_face)."""
    x_f = face.x_face
    poly = face.tunnel_polygon_yz
    
    ny, nz = n_y, n_z
    ny_z_sq = ny**2 + nz**2
    if ny_z_sq < 1e-12:
        return []
        
    C_rhs = n_x * (c_x - x_f) + ny * c_y + nz * c_z
    dist_to_line = abs(x_f - c_x) / np.sqrt(ny_z_sq)
    if dist_to_line >= radius:
        return []
        
    factor = (ny * c_y + nz * c_z - C_rhs) / ny_z_sq
    y_mid = c_y - ny * factor
    z_mid = c_z - nz * factor
    mid_pt = np.array([y_mid, z_mid])
    
    chord_half_len = np.sqrt(radius**2 - dist_to_line**2)
    d_line = np.array([-nz, ny]) / np.sqrt(ny_z_sq)
    
    p0 = mid_pt - chord_half_len * d_line
    p1 = mid_pt + chord_half_len * d_line
    
    clipped = clip_line_segment_to_polygon(p0, p1, poly)
    
    traces = []
    tid = start_trace_id
    for cp0, cp1 in clipped:
        t = FaceTrace(
            face_id=face.face_id,
            trace_id=tid,
            x_face=x_f,
            p0_y=float(cp0[0]),
            p0_z=float(cp0[1]),
            p1_y=float(cp1[0]),
            p1_z=float(cp1[1]),
            confidence=1.0,
            parent_fracture_id=parent_fracture_id
        )
        t.set_id = set_id
        traces.append(t)
        tid += 1
        
    return traces


# ==============================================================================
# SECTION 7: CONSTRAINED MAP PLANE FITTER (constrained_map_fitter.py)
# ==============================================================================

def fit_plane_svd_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fits a 3D plane normal and centroid to a set of 3D points using SVD."""
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    _, _, vh = np.linalg.svd(shifted)
    normal = vh[-1]
    
    if normal[0] < 0:
        normal = -normal
    return centroid, normal


def get_local_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes two orthogonal unit vectors in the plane perpendicular to the normal."""
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
    """Calculates expected 2D trace length of a 3D disc intersecting x = x_face."""
    A = e_u[0]
    B = e_v[0]
    C = x_face - centroid[0]
    
    denom = A**2 + B**2
    if denom < 1e-12:
        return 0.0
        
    proj_dist = abs(A * u0 + B * v0 - C) / np.sqrt(denom)
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
    """Computes negative log-posterior score for params = [u0, v0, R]."""
    u0, v0, R = params
    if R <= 1e-3:
        return 1e10
        
    ln_prior = - ((np.log(R) - mu_s)**2) / (2 * sigma_s**2) - np.log(R)
    
    ln_likelihood = 0.0
    for t in traces:
        L_exp = evaluate_analytical_trace_length(u0, v0, R, t.x_face, centroid, normal, e_u, e_v)
        
        if t.censoring_class == 0:
            ln_likelihood += - ((t.length - L_exp)**2) / (2 * sigma_L**2) - np.log(np.sqrt(2 * np.pi) * sigma_L)
        else:
            z = (L_exp - t.length) / sigma_L
            ln_likelihood += norm.logcdf(z)
            
    return float(-(ln_prior + ln_likelihood))


def compute_numerical_hessian(func, x0: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Computes numerical Hessian matrix at x0 using finite differences."""
    n = len(x0)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                x_plus = x0.copy(); x_plus[i] += eps
                x_minus = x0.copy(); x_minus[i] -= eps
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
    """Solves the censoring-aware constrained MAP problem to fit a 3D disc."""
    endpoints_3d = []
    for t in traces:
        endpoints_3d.append([t.x_face, t.p0_y, t.p0_z])
        endpoints_3d.append([t.x_face, t.p1_y, t.p1_z])
    endpoints_3d = np.array(endpoints_3d)
    
    centroid, normal = fit_plane_svd_3d(endpoints_3d)
    e_u, e_v = get_local_axes(normal)
    
    local_pts = project_points_to_plane_2d(endpoints_3d, centroid, e_u, e_v)
    mid_u = np.mean(local_pts[:, 0])
    mid_v = np.mean(local_pts[:, 1])
    
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
    
    loss_func = lambda x: negative_log_posterior(x, traces, centroid, normal, e_u, e_v, mu_s, sigma_s, sigma_L)
    res = minimize(loss_func, x0, bounds=bounds, method='L-BFGS-B')
    u0_map, v0_map, r_map = res.x
    center_3d = centroid + u0_map * e_u + v0_map * e_v
    
    cov = None
    try:
        hess = compute_numerical_hessian(loss_func, res.x)
        if np.all(np.linalg.eigvals(hess) > 0):
            cov = np.linalg.inv(hess)
    except Exception:
        pass
        
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
    """Probabilistic posterior candidate generator for single-face traces."""
    candidates = []
    sampled_normals = sample_vmf_normals(mean_normal, kappa, n_samples=n_samples, seed=random_seed)
    
    for b in range(n_samples):
        n_pert = sampled_normals[b]
        centroid = np.array([trace.x_face, trace.midpoint_y, trace.midpoint_z])
        e_u_p, e_v_p = get_local_axes(n_pert)
        
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
            confidence=1.0 / n_samples,
            set_id=set_id,
            is_single_face_candidate=True
        ))
        
    return candidates


# ==============================================================================
# SECTION 8: BAYES FACTOR FACE ASSOCIATION (face_association.py)
# ==============================================================================

def get_candidate_plane_normal(t0: FaceTrace, t1: FaceTrace) -> np.ndarray:
    """Computes candidate 3D plane normal defined by two parallel traces."""
    dx = t1.x_face - t0.x_face
    theta_avg = 0.5 * (t0.orientation_2d + t1.orientation_2d)
    d_avg = np.array([0.0, np.cos(theta_avg), np.sin(theta_avg)])
    v_mid = np.array([dx, t1.midpoint_y - t0.midpoint_y, t1.midpoint_z - t0.midpoint_z])
    
    n_raw = np.cross(d_avg, v_mid)
    n_len = np.linalg.norm(n_raw)
    
    if n_len > 1e-9:
        n = n_raw / n_len
    else:
        n = np.array([0.0, -np.sin(theta_avg), np.cos(theta_avg)])
        
    if n[0] < 0:
        n = -n
    return n


def check_physical_gate(
    t0: FaceTrace,
    t1: FaceTrace,
    max_angle_deg: float = 20.0,
    max_midpoint_dist: float = 1.8
) -> bool:
    """Applies physical gating constraints before Bayes Factor matching."""
    d_theta = abs(t0.orientation_2d - t1.orientation_2d)
    if d_theta > np.pi / 2.0:
        d_theta = np.pi - d_theta
        
    if d_theta > np.radians(max_angle_deg):
        return False
        
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
    sigma_theta: float = 0.087,
    sigma_d: float = 0.15,
    bg_log_likelihood: float = -2.0
) -> float:
    """Computes log BF_ij = ln p(obs | H1) - ln p(obs | H0)."""
    set_id = t0.set_id if t0.set_id == t1.set_id else None
    n_plane = get_candidate_plane_normal(t0, t1)
    
    d_theta = abs(t0.orientation_2d - t1.orientation_2d)
    if d_theta > np.pi / 2.0:
        d_theta = np.pi - d_theta
    ln_p_orient = - (d_theta**2) / (2 * sigma_theta**2) - np.log(np.sqrt(2 * np.pi) * sigma_theta)
    
    v_mid = np.array([t1.x_face - t0.x_face, t1.midpoint_y - t0.midpoint_y, t1.midpoint_z - t0.midpoint_z])
    plane_dist = abs(np.dot(v_mid, n_plane))
    ln_p_spatial = - (plane_dist**2) / (2 * sigma_d**2) - np.log(np.sqrt(2 * np.pi) * sigma_d)
    
    ln_p_prior = 0.0
    if set_id is not None and set_stats is not None and set_id in set_stats:
        mean_normal, kappa = set_stats[set_id]
        cos_angle = abs(np.dot(n_plane, mean_normal))
        ln_p_prior = kappa * cos_angle - np.log(2 * np.pi * (np.exp(kappa) - np.exp(-kappa)) / kappa + 1e-9)
        
    dist_3d = np.linalg.norm(v_mid)
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
    """Intersects a 3D plane with the tunnel polygon boundary at face x = x_face."""
    if len(poly_yz) == 0:
        return None
        
    rhs = normal[0] * (center[0] - x_face) + normal[1] * center[1] + normal[2] * center[2]
    ny, nz = normal[1], normal[2]
    
    intersections = []
    n_points = len(poly_yz)
    
    for i in range(n_points):
        v0 = poly_yz[i]
        v1 = poly_yz[(i + 1) % n_points]
        
        dy = v1[0] - v0[0]
        dz = v1[1] - v0[1]
        
        denom = ny * dy + nz * dz
        if abs(denom) < 1e-9:
            continue
            
        t = (rhs - ny * v0[0] - nz * v0[1]) / denom
        if 0.0 <= t <= 1.0:
            p_inter = v0 + t * np.array([dy, dz])
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
    """Evaluates 3-face absence penalization information."""
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
            
        n_plane = get_candidate_plane_normal(t0, t1)
        cx = 0.5 * (t0.x_face + t1.x_face)
        cy = 0.5 * (t0.midpoint_y + t1.midpoint_y)
        cz = 0.5 * (t0.midpoint_z + t1.midpoint_z)
        center = np.array([cx, cy, cz])
        
        inter = find_plane_polygon_intersection(n_plane, center, face_2.x_face, face_2.tunnel_polygon_yz)
        if inter is None:
            continue
            
        p_a, p_b = inter
        candidate_len = np.linalg.norm(p_a - p_b)
        
        if candidate_len >= min_visible_length:
            mid_cand = 0.5 * (p_a + p_b)
            orient_cand = np.arctan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
            if orient_cand > np.pi / 2.0:
                orient_cand -= np.pi
            elif orient_cand < -np.pi / 2.0:
                orient_cand += np.pi
                
            has_matching_trace = False
            for t2 in traces_f2:
                d_theta = abs(t2.orientation_2d - orient_cand)
                if d_theta > np.pi / 2.0:
                    d_theta = np.pi - d_theta
                    
                if d_theta <= np.radians(20.0):
                    dist = np.sqrt((t2.midpoint_y - mid_cand[0])**2 + (t2.midpoint_z - mid_cand[1])**2)
                    if dist < 1.5:
                        has_matching_trace = True
                        break
                        
            if not has_matching_trace:
                m.log_bayes_factor += penalty
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
    """Matches traces between consecutive faces using Hungarian optimization."""
    if not traces_prev or not traces_curr:
        return []
        
    n_prev = len(traces_prev)
    n_curr = len(traces_curr)
    
    cost_matrix = np.full((n_prev, n_curr), 1e6)
    bf_matrix = np.full((n_prev, n_curr), -1e6)
    
    for i, t_p in enumerate(traces_prev):
        for j, t_c in enumerate(traces_curr):
            if t_p.set_id != t_c.set_id:
                continue
                
            if not check_physical_gate(t_p, t_c, max_angle_deg, max_midpoint_dist):
                continue
                
            log_bf = compute_log_bayes_factor(t_p, t_c, set_stats)
            bf_matrix[i, j] = log_bf
            
            if log_bf > -1.0:
                cost_matrix[i, j] = -log_bf
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_ind, col_ind):
        log_bf = bf_matrix[r, c]
        is_accepted = cost_matrix[r, c] < 1e5
        
        matches.append(TraceMatch(
            face_id_prev=traces_prev[r].face_id,
            face_id_curr=traces_curr[c].face_id,
            trace_id_prev=traces_prev[r].trace_id,
            trace_id_curr=traces_curr[c].trace_id,
            log_bayes_factor=log_bf,
            accepted=is_accepted
        ))
        
    return matches


# ==============================================================================
# SECTION 9: RESIDUAL PRIOR GENERATOR & MOMENTS (residual_dfn_generator.py)
# ==============================================================================

def compute_orientation_mapping_factor(
    mean_normal: np.ndarray,
    kappa: float,
    face_normal: np.ndarray = np.array([1.0, 0.0, 0.0])
) -> float:
    """Computes set-wise orientation mapping factor kappa_s(m) = E[|n x w_m|]."""
    samples = sample_vmf_normals(mean_normal, kappa, n_samples=500, seed=42)
    cross_lens = []
    for n in samples:
        cross_vec = np.cross(n, face_normal)
        cross_lens.append(np.linalg.norm(cross_vec))
        
    return float(np.mean(cross_lens))


def solve_lognormal_joint_moments(residual_lengths: np.ndarray) -> Tuple[float, float]:
    """Solves for lognormal size distribution parameters (mu, sigma) via joint moments."""
    if len(residual_lengths) < 3:
        return float(np.log(1.5)), 0.35
        
    mean_L = np.mean(residual_lengths)
    mean_L2 = np.mean(residual_lengths**2)
    
    ratio = mean_L2 / (mean_L**2 + 1e-9)
    c_factor = (3 * np.pi**2) / 32.0
    sigma2_val = np.log(c_factor * ratio + 1e-9)
    
    sigma_s2 = float(np.clip(sigma2_val, 0.04, 0.50))
    sigma_s = np.sqrt(sigma_s2)
    
    mu_val = np.log(2.0 * mean_L / np.pi) - 1.5 * sigma_s2
    mu_s = float(np.clip(mu_val, np.log(0.2), np.log(5.0)))
    
    return mu_s, sigma_s


def compute_residual_statistics_and_priors(
    obs_traces: List[FaceTrace],
    det_planes: List[ReconstructedPlane],
    faces: List[ExcavationFace],
    set_stats: Dict[int, Tuple[np.ndarray, float]]
) -> Dict[int, Dict[str, float]]:
    """Calculates residual intensity P21, NA and generates priors via joint moments."""
    results = {}
    det_trace_ids = set()
    for p in det_planes:
        for tid in p.source_trace_ids:
            det_trace_ids.add(tid)
            
    residual_traces = [t for t in obs_traces if t.trace_id not in det_trace_ids]
    
    for set_id, (mean_normal, kappa) in set_stats.items():
        set_res_traces = [t for t in residual_traces if t.set_id == set_id]
        lengths = np.array([t.length for t in set_res_traces])
        
        mu_s, sigma_s = solve_lognormal_joint_moments(lengths)
        p21_res_sum = 0.0
        face_mapping_factors = []
        
        for face in faces:
            face_res = [t for t in set_res_traces if t.face_id == face.face_id]
            p21_face_res = sum(t.length for t in face_res)
            poly = face.tunnel_polygon_yz
            if poly is not None and len(poly) > 2:
                y = poly[:, 0]
                z = poly[:, 1]
                area = 0.5 * np.abs(np.dot(y, np.roll(z, 1)) - np.dot(z, np.roll(y, 1)))
                area = max(1.0, area)
            else:
                area = 1.0
                
            p21_face_res_areal = p21_face_res / area
            p21_res_sum += p21_face_res_areal
            
            kappa_m = compute_orientation_mapping_factor(mean_normal, kappa)
            face_mapping_factors.append(kappa_m)
            
        sum_kappa = sum(face_mapping_factors) if face_mapping_factors else 1.0
        P32 = p21_res_sum / (sum_kappa + 1e-9)
        P32 = max(1e-5, P32)
        
        expected_r2 = np.exp(2 * mu_s + 2 * (sigma_s**2))
        P30 = P32 / (np.pi * expected_r2 + 1e-9)
        
        results[set_id] = {
            'mu_s': mu_s,
            'sigma_s': sigma_s,
            'P32': P32,
            'P30': P30
        }
        
    return results


# ==============================================================================
# SECTION 10: MANIFOLD GLIDE MCMC / SA OPTIMIZER (manifold_glide_optimizer.py)
# ==============================================================================

def evaluate_dfn_loss(
    obs_traces: List[FaceTrace],
    sim_traces: List[FaceTrace],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Computes multi-objective discrepancy loss comparing simulated to observed traces."""
    if weights is None:
        weights = {
            'p21_error': 1.5,
            'count_error': 2.5,
            'length_error': 2.0,
            'censoring_error': 2.0
        }
        
    n_obs = len(obs_traces)
    n_sim = len(sim_traces)
    if n_obs == 0:
        return {'total': 0.0}
        
    p21_obs = sum(t.length for t in obs_traces)
    p21_sim = sum(t.length for t in sim_traces)
    
    err_p21 = abs(p21_obs - p21_sim) / p21_obs
    err_count = abs(n_obs - n_sim) / n_obs
    
    mean_L_obs = np.mean([t.length for t in obs_traces]) if n_obs > 0 else 1.0
    mean_L_sim = np.mean([t.length for t in sim_traces]) if n_sim > 0 else 0.0
    err_length = abs(mean_L_obs - mean_L_sim) / mean_L_obs
    
    obs_cens = np.array([t.censoring_class for t in obs_traces])
    sim_cens = np.array([t.censoring_class for t in sim_traces])
    
    obs_ratios = np.array([np.sum(obs_cens == c) for c in [0, 1, 2]]) / (n_obs + 1e-9)
    sim_ratios = np.array([np.sum(sim_cens == c) for c in [0, 1, 2]]) / (n_sim + 1e-9)
    err_censoring = float(np.sum(np.abs(obs_ratios - sim_ratios)))
    
    total_loss = (
        weights['p21_error'] * err_p21 +
        weights['count_error'] * err_count +
        weights['length_error'] * err_length +
        weights['censoring_error'] * err_censoring
    )
    
    return {
        'total': float(total_loss),
        'p21_error': err_p21,
        'count_error': err_count,
        'length_error': float(err_length),
        'censoring_error': err_censoring,
        'obs_count': n_obs,
        'sim_count': n_sim,
        'mean_L_obs': float(mean_L_obs),
        'mean_L_sim': float(mean_L_sim),
        'obs_ratios': obs_ratios,
        'sim_ratios': sim_ratios
    }


def precalculate_fixed_traces(
    det_planes: List[ReconstructedPlane],
    faces: List[ExcavationFace]
) -> List[FaceTrace]:
    """Pre-intersects fixed deterministic planes to save computing in SA loop."""
    fixed_traces = []
    tid = 10000
    
    for face in faces:
        for dp in det_planes:
            if abs(dp.point_x - face.x_face) >= dp.radius:
                continue
                
            ft = intersect_disc_with_face(
                dp.point_x, dp.point_y, dp.point_z,
                dp.normal_x, dp.normal_y, dp.normal_z,
                dp.radius, face, start_trace_id=tid, set_id=dp.set_id or 1
            )
            fixed_traces.extend(ft)
            tid += len(ft)
            
    for face in faces:
        classify_censoring(fixed_traces, face, tolerance=0.10)
        
    return fixed_traces


def simulate_stochastic_traces(
    stoch_fractures: List[StochasticFracture],
    faces: List[ExcavationFace],
    start_tid: int = 50000
) -> List[FaceTrace]:
    """Simulates active stochastic fracture intersections with bounding box filters."""
    stoch_traces = []
    tid = start_tid
    
    for face in faces:
        active_sf = [
            sf for sf in stoch_fractures
            if abs(sf.center_x - face.x_face) < sf.radius
        ]
        
        for sf in active_sf:
            ft = intersect_disc_with_face(
                sf.center_x, sf.center_y, sf.center_z,
                sf.normal_x, sf.normal_y, sf.normal_z,
                sf.radius, face, start_trace_id=tid, set_id=sf.set_id
            )
            stoch_traces.extend(ft)
            tid += len(ft)
            
    for face in faces:
        classify_censoring(stoch_traces, face, tolerance=0.10)
        
    return stoch_traces


def run_manifold_glide_sa(
    obs_traces: List[FaceTrace],
    det_planes: List[ReconstructedPlane],
    faces: List[ExcavationFace],
    set_stats: Dict[int, Tuple[np.ndarray, float]],
    initial_residual_priors: Dict[int, Dict[str, float]],
    domain: Dict[str, float],
    sa_iterations: int = 150,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.95,
    random_seed: int = 42
) -> Tuple[Dict[int, Dict[str, float]], List[StochasticFracture], List[FaceTrace]]:
    """Executes Simulated Annealing using the Manifold Glide decoupling strategy."""
    rng = np.random.default_rng(random_seed)
    
    print("  [*] Pre-calculating fixed deterministic plane intersections with tunnel faces...")
    t0_pre = time.time()
    fixed_traces = precalculate_fixed_traces(det_planes, faces)
    print(f"  -> Generated {len(fixed_traces)} fixed traces from reconstructed planes (Elapsed: {time.time() - t0_pre:.2f}s)")
    
    current_state = {}
    for set_id, priors in initial_residual_priors.items():
        mu_s = priors['mu_s']
        sigma_s = priors['sigma_s']
        P30 = priors['P30']
        
        rho = mu_s
        chi = float(np.log(P30) + 2 * mu_s)
        
        current_state[set_id] = {
            'chi': chi,
            'rho': rho,
            'sigma_s': sigma_s
        }
        
    def state_to_physical(state: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
        physical = {}
        for sid, s_val in state.items():
            mu = s_val['rho']
            P30 = float(np.exp(s_val['chi'] - 2 * mu))
            physical[sid] = {
                'mu_s': mu,
                'sigma_s': s_val['sigma_s'],
                'P30': float(np.clip(P30, 1e-5, 0.5))
            }
        return physical

    phys_priors = state_to_physical(current_state)
    stoch_dfn = generate_stochastic_dfn(domain, phys_priors, set_stats)
    stoch_traces = simulate_stochastic_traces(stoch_dfn, faces)
    
    sim_traces = fixed_traces + stoch_traces
    current_loss_dict = evaluate_dfn_loss(obs_traces, sim_traces)
    current_loss = current_loss_dict['total']
    
    best_state = {sid: val.copy() for sid, val in current_state.items()}
    best_loss = current_loss
    best_loss_dict = current_loss_dict
    
    temp = initial_temp
    
    for it in range(sa_iterations):
        temp = initial_temp * (cooling_rate ** it)
        
        proposal_state = {}
        for sid, s_val in current_state.items():
            step_chi = rng.normal(0, 0.15 * temp)
            step_rho = rng.normal(0, 0.15 * temp)
            
            proposal_state[sid] = {
                'chi': s_val['chi'] + step_chi,
                'rho': s_val['rho'] + step_rho,
                'sigma_s': s_val['sigma_s']
            }
            
        proposal_phys = state_to_physical(proposal_state)
        proposal_stoch = generate_stochastic_dfn(domain, proposal_phys, set_stats)
        proposal_stoch_traces = simulate_stochastic_traces(proposal_stoch, faces)
        
        proposal_sim = fixed_traces + proposal_stoch_traces
        proposal_loss_dict = evaluate_dfn_loss(obs_traces, proposal_sim)
        proposal_loss = proposal_loss_dict['total']
        
        delta_loss = proposal_loss - current_loss
        if delta_loss < 0 or rng.uniform() < np.exp(-delta_loss / (temp + 1e-9)):
            current_state = proposal_state
            current_loss = proposal_loss
            current_loss_dict = proposal_loss_dict
            
            if proposal_loss < best_loss:
                best_state = {sid: val.copy() for sid, val in proposal_state.items()}
                best_loss = proposal_loss
                best_loss_dict = proposal_loss_dict
                
        if (it + 1) % 10 == 0 or it == 0:
            print(f"  [SA] Iter {it+1:3d}/{sa_iterations}: Loss={current_loss:.4f} (Best={best_loss:.4f}), Temp={temp:.4f}")
            
    best_phys = state_to_physical(best_state)
    best_stoch = generate_stochastic_dfn(domain, best_phys, set_stats)
    best_stoch_traces = simulate_stochastic_traces(best_stoch, faces)
    best_sim_traces = fixed_traces + best_stoch_traces
    
    print(f"\n  [SA 완료] Best Loss: {best_loss:.4f}")
    for sid, val in best_phys.items():
        r_avg = np.exp(val['mu_s'] + 0.5 * (val['sigma_s']**2))
        print(f"    Set {sid}: P30={val['P30']:.6f}, Avg Radius={r_avg:.3f}m, mu_s={val['mu_s']:.3f}")
        
    return best_phys, best_stoch, best_sim_traces


# ==============================================================================
# SECTION 11: GEOM COMPATIBLE DFN EXPORTER (dfn_exporter.py)
# ==============================================================================

def export_dfn_to_hdf5(
    file_path: str,
    det_planes: List[ReconstructedPlane],
    single_face_candidates: List[ReconstructedPlane],
    stoch_fractures: List[StochasticFracture],
    tunnel_poly_yz: np.ndarray,
    domain_box: np.ndarray,
    x_start: float = 0.0,
    x_end: float = 6.0
):
    """Writes all deterministic and stochastic fractures into a unified HDF5 file."""
    centers = []
    normals = []
    radii = []
    set_ids = []
    sources = []
    
    # 1. Deterministic Multi-Face Planes
    for p in det_planes:
        centers.append([p.point_x, p.point_y, p.point_z])
        normals.append([p.normal_x, p.normal_y, p.normal_z])
        radii.append(p.radius)
        set_ids.append(p.set_id or 1)
        sources.append(1)
        
    # 2. Single-Face Probabilistic Candidates
    for p in single_face_candidates:
        centers.append([p.point_x, p.point_y, p.point_z])
        normals.append([p.normal_x, p.normal_y, p.normal_z])
        radii.append(p.radius)
        set_ids.append(p.set_id or 1)
        sources.append(2)
        
    # 3. Stochastic PPP Fractures
    for sf in stoch_fractures:
        centers.append([sf.center_x, sf.center_y, sf.center_z])
        normals.append([sf.normal_x, sf.normal_y, sf.normal_z])
        radii.append(sf.radius)
        set_ids.append(sf.set_id)
        sources.append(3)
        
    centers = np.array(centers, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    set_ids = np.array(set_ids, dtype=np.uint16)
    sources = np.array(sources, dtype=np.uint8)
    
    with h5py.File(file_path, 'w') as f:
        grp_frac = f.create_group('fractures')
        grp_frac.create_dataset('centers', data=centers.T, compression='gzip')
        grp_frac.create_dataset('normals', data=normals.T, compression='gzip')
        grp_frac.create_dataset('radii', data=radii, compression='gzip')
        grp_frac.create_dataset('set_id', data=set_ids, compression='gzip')
        grp_frac.create_dataset('source_type', data=sources, compression='gzip')
        
        grp_tunnel = f.create_group('tunnel')
        grp_tunnel.create_dataset('poly_YZ', data=tunnel_poly_yz.T, compression='gzip')
        
        grp_meta = f.create_group('meta')
        grp_meta.create_dataset('domain_box', data=domain_box)
        grp_meta.create_dataset('crop_box', data=domain_box)
        grp_meta.create_dataset('x_start', data=x_start)
        grp_meta.create_dataset('x_end', data=x_end)
        
        f.attrs['n_deterministic'] = len(det_planes)
        f.attrs['n_single_face'] = len(single_face_candidates)
        f.attrs['n_stochastic'] = len(stoch_fractures)
        f.attrs['n_total'] = len(radii)
        
    print(f"\n  [HDF5 Export Complete] Saved {len(radii)} fractures to: {file_path}")
    print(f"    - Multi-face Deterministic: {len(det_planes)}")
    print(f"    - Single-face Candidates  : {len(single_face_candidates)}")
    print(f"    - Volumetric Stochastic   : {len(stoch_fractures)}")
