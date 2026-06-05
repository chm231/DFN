"""
Hekmatnejad et al. (2018) Trace Length Bias Correction & True Length Inversion Module.
Implements non-parametric true length distribution reconstruction via Terzaghi weighting,
truncation filtering, and a custom Weighted Kaplan-Meier Estimator.
"""
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional, Dict, Any

# Optional import of lifelines for validation
try:
    import lifelines
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


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
        """
        Parameters:
        -----------
        min_truncation : float
            Detection limit (c). Traces with length < min_truncation are filtered out.
        max_weight : float
            Maximum upper bound (w_max) for Terzaghi orientation weighting to prevent singularity.
        apply_orientation_bias : bool
            Whether to apply Terzaghi weighting based on dip_angle.
        dip_in_degrees : bool
            True if dip angles are provided in degrees, False if in radians.
        """
        self.min_truncation = min_truncation
        self.max_weight = max_weight
        self.apply_orientation_bias = apply_orientation_bias
        self.dip_in_degrees = dip_in_degrees

    def compute_terzaghi_weights(self, dip_angles: np.ndarray) -> np.ndarray:
        """
        Computes Terzaghi orientation bias correction weights: w_i = 1 / sin(theta_i)
        """
        if not self.apply_orientation_bias or dip_angles is None:
            return np.ones(len(dip_angles))

        # Convert to radians if necessary
        angles_rad = np.deg2rad(dip_angles) if self.dip_in_degrees else dip_angles
        
        # Ensure angles are in valid physical range (0, pi/2]
        angles_rad = np.abs(angles_rad)
        angles_rad = np.clip(angles_rad, 1e-6, np.pi / 2.0)
        
        # Calculate weights
        weights = 1.0 / np.sin(angles_rad)
        
        # Apply cap to prevent extreme sensitivity on low-angle intersections
        weights = np.minimum(weights, self.max_weight)
        return weights

    def filter_truncation(
        self, 
        lengths: np.ndarray, 
        censoring_types: np.ndarray, 
        dip_angles: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Filters out traces with length less than the minimum resolution threshold (c).
        """
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
        Supports continuous custom weights (Terzaghi weights).
        
        Censoring Type Mapping:
        ----------------------
        0: Uncensored (Event observed, trace ends naturally on the face)
        1: Right-censored (One end clipped by face boundary)
        2: Interval/Right-censored (Both ends clipped by face boundary)
        
        Returns:
        --------
        unique_lengths : np.ndarray
            Sorted unique trace lengths.
        survival_probs : np.ndarray
            Reconstructed survival function values corresponding to unique_lengths.
        """
        n_samples = len(lengths)
        if n_samples == 0:
            return np.array([self.min_truncation]), np.array([1.0])
            
        # Event is observed only if censoring_type is 0 (Uncensored)
        events = (censoring_types == 0).astype(int)
        
        # Sort all arrays by length in ascending order
        sort_idx = np.argsort(lengths)
        sorted_l = lengths[sort_idx]
        sorted_e = events[sort_idx]
        sorted_w = weights[sort_idx]
        
        # Identify unique lengths and aggregate events/risk weights
        unique_l = np.unique(sorted_l)
        survival_probs = []
        
        # Pre-calculate cumulative weights from right to left for efficient risk-set summation
        # total_w_above[i] represents sum(w_k) for all k with l_k >= unique_l[i]
        total_w_above = np.zeros(len(unique_l))
        events_w_at = np.zeros(len(unique_l))
        
        for idx, ul in enumerate(unique_l):
            # Risk set: all indices where length is >= current unique length
            risk_mask = sorted_l >= ul
            total_w_above[idx] = np.sum(sorted_w[risk_mask])
            
            # Event set: indices where length is exactly current unique length AND event was observed
            event_mask = (sorted_l == ul) & (sorted_e == 1)
            events_w_at[idx] = np.sum(sorted_w[event_mask])

        # Step-by-step product formulation
        current_s = 1.0
        for idx in range(len(unique_l)):
            n_j = total_w_above[idx]  # Sum of weights in risk set
            d_j = events_w_at[idx]    # Sum of weights of observed events
            
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
        """
        Dynamic cross-validation helper using the lifelines package if available.
        """
        if not HAS_LIFELINES:
            return None
            
        events = (censoring_types == 0).astype(int)
        kmf = lifelines.KaplanMeierFitter()
        # Fit Kaplan-Meier using lifelines with Terzaghi weights
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
        
        Returns:
        --------
        cdf_fun : Callable[[np.ndarray], np.ndarray]
            Smooth CDF function mapping l -> F(l)
        pdf_fun : Callable[[np.ndarray], np.ndarray]
            Smooth PDF function mapping l -> f(l)
        """
        # CDF definition: F(l) = 1 - S(l)
        cdf_probs = 1.0 - survival_probs
        
        # Enforce physical boundary conditions: F(c) = 0
        # Insert boundary point at the minimum resolution limit
        fit_lengths = np.concatenate([[self.min_truncation], unique_lengths])
        fit_cdf = np.concatenate([[0.0], cdf_probs])
        
        # Ensure strict monotonicity by cleaning any tiny numerical fluctuations
        for idx in range(1, len(fit_cdf)):
            if fit_cdf[idx] < fit_cdf[idx - 1]:
                fit_cdf[idx] = fit_cdf[idx - 1]
                
        # Scale to ensure the maximum CDF value reaches 1.0 as l extends to infinity
        if fit_cdf[-1] > 0.0:
            fit_cdf = fit_cdf / fit_cdf[-1]
            
        # 1. Monotonic cubic interpolation (Pchip) for CDF
        pchip_cdf = interp.PchipInterpolator(fit_lengths, fit_cdf, extrapolate=True)
        
        # 2. Analytical derivative for PDF
        pchip_pdf = pchip_cdf.derivative()
        
        # Wrap CDF and PDF calls to handle physical bounds safely
        def safe_cdf(l: np.ndarray) -> np.ndarray:
            l_arr = np.atleast_1d(l)
            res = pchip_cdf(l_arr)
            # Clip physically impossible bounds
            res = np.where(l_arr < self.min_truncation, 0.0, res)
            # Extrapolate beyond maximum observed length to exactly 1.0 (prevent polynomial collapse)
            res = np.where(l_arr > fit_lengths[-1], 1.0, res)
            res = np.clip(res, 0.0, 1.0)
            return res[0] if np.isscalar(l) else res

        def safe_pdf(l: np.ndarray) -> np.ndarray:
            l_arr = np.atleast_1d(l)
            res = pchip_pdf(l_arr)
            # Clip PDF to be non-negative, and exactly 0.0 outside truncation or maximum bounds
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
        """
        Orchestrates the entire correction & inversion pipeline:
        Step 1 -> Step 2 -> Step 3 -> Step 4
        """
        # Step 1: Orientation weights calculation
        if self.apply_orientation_bias and raw_dips is not None:
            raw_weights = self.compute_terzaghi_weights(raw_dips)
        else:
            raw_weights = np.ones(len(raw_lengths))

        # Step 2: Truncation filtering (removes lengths < c)
        lengths, censoring, weights = self.filter_truncation(raw_lengths, raw_censoring, raw_weights)
        
        # Step 3: Weighted Kaplan-Meier estimation
        unique_lengths, survival_probs = self.fit_weighted_kaplan_meier(lengths, censoring, weights)
        
        # Dynamic verification if lifelines is present
        lifelines_result = None
        if HAS_LIFELINES:
            lifelines_result = self.cross_validate_with_lifelines(lengths, censoring, weights)
            
        # Step 4: CDF and PDF smooth transformation
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
    """
    Plots professional comparative histograms of the raw vs. corrected trace length distributions,
    overlaying the continuous non-parametric true CDF and PDF curves.
    """
    filtered_lengths = results["filtered_lengths"]
    weights = results["filtered_weights"]
    cdf_fun = results["cdf_function"]
    pdf_fun = results["pdf_function"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#fafafa")
    
    # Elegant custom HSL tailored colors
    c_raw = "#d95f02"       # Vibrant terracotta orange
    c_corrected = "#1b9e77" # Deep premium teal
    c_pdf = "#7570b3"       # Sleek indigo
    
    # ------------------ PANEL 1: PDF comparison ------------------
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    
    # Density histogram of raw data
    ax1.hist(
        raw_lengths, bins=25, density=True, alpha=0.35, color=c_raw, 
        edgecolor=c_raw, linewidth=1.2, label="Raw Observed Distribution"
    )
    
    # Density histogram of corrected and truncated data (using weights)
    ax1.hist(
        filtered_lengths, bins=25, density=True, weights=weights, alpha=0.45, color=c_corrected,
        edgecolor=c_corrected, linewidth=1.2, label="Bias-Corrected (Weighted)"
    )
    
    # Overlay Continuous Inverted PDF
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
    
    # Empirical CDF of raw data
    sorted_raw = np.sort(raw_lengths)
    ecdf_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
    ax2.step(
        sorted_raw, ecdf_raw, color=c_raw, alpha=0.6, linewidth=1.8,
        where="post", label="Empirical CDF (Raw)"
    )
    
    # Reconstructed Continuous CDF
    cdf_vals = cdf_fun(l_grid)
    ax2.plot(
        l_grid, cdf_vals, color=c_corrected, linewidth=3.0, linestyle="-",
        label="Inverted True CDF $F(l)$"
    )
    
    # If lifelines verification is present, plot it for visual comparison
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
    
    # Enhance visual appeal
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[*] Premium comparative distribution figure saved to: {save_path}")
        plt.close()
    else:
        plt.show()
