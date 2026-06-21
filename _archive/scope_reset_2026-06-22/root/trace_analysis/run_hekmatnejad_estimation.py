"""
Synthetic data simulation and validation runner for Hekmatnejad et al. (2018)
non-parametric trace length distribution inversion.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Insert parent directories
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

from trace_reconstruction_unified import HekmatnejadEstimator


def generate_synthetic_biased_traces(
    n_population: int = 5000,
    mu: float = 0.8,
    sigma: float = 0.45,
    c_truncation: float = 0.15,
    r_tunnel: float = 4.5,
    seed: int = 101
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic trace dataset that rigorously incorporates:
    1. Geometrical Lognormal True Length Distribution.
    2. Terzaghi Orientation Sampling Bias (low-dip planes are less likely to intersect).
    3. Truncation Limit (traces shorter than c_truncation are invisible).
    4. Circular Tunnel Window Censoring (traces extending beyond boundary are clipped).
    
    Returns:
        true_lengths_sampled : np.ndarray
            The actual uncensored, untruncated lengths that were successfully observed.
        observed_lengths : np.ndarray
            The clipped lengths visible inside the tunnel.
        censoring_types : np.ndarray
            0: Uncensored, 1: One-end clipped, 2: Both-end clipped.
        dip_angles : np.ndarray
            Dip angles of the successfully sampled planes.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Sample true lengths from lognormal distribution
    # Lognormal parameters: mean of ln(L) = mu, std of ln(L) = sigma
    candidate_lengths = rng.lognormal(mean=mu, sigma=sigma, size=n_population)
    
    # 2. Assign random dip angles (in degrees, [5, 85])
    # Planes with higher dip angles (closer to perpendicular to face) have higher probability of being sampled.
    candidate_dips = rng.uniform(5.0, 85.0, size=n_population)
    
    # Accept-reject sampling for Terzaghi Orientation Bias
    # Sampling probability is proportional to sin(dip_angle)
    sampling_probs = np.sin(np.deg2rad(candidate_dips))
    accept_mask = rng.uniform(0.0, 1.0, size=n_population) <= sampling_probs
    
    lengths_orient_biased = candidate_lengths[accept_mask]
    dips_orient_biased = candidate_dips[accept_mask]
    
    # 3. Simulate circular tunnel face boundary censoring (Windowing Effect)
    # Circular tunnel face centered at (0, 0) with radius r_tunnel
    observed_lengths = []
    censoring_types = []
    true_lengths_observed = []
    dip_angles_observed = []
    
    for l, dip in zip(lengths_orient_biased, dips_orient_biased):
        # Place midpoint uniformly in a square bounding the circle, then reject if outside
        # to ensure uniform spatial density on the excavation face.
        while True:
            mid_y = rng.uniform(-r_tunnel, r_tunnel)
            mid_z = rng.uniform(-r_tunnel, r_tunnel)
            if mid_y**2 + mid_z**2 <= r_tunnel**2:
                break
                
        # Generate random 2D trace orientation angle (theta)
        theta = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        dy = np.cos(theta)
        dz = np.sin(theta)
        
        # Original endpoints of the trace
        p0_y = mid_y - (l / 2.0) * dy
        p0_z = mid_z - (l / 2.0) * dz
        p1_y = mid_y + (l / 2.0) * dy
        p1_z = mid_z + (l / 2.0) * dz
        
        # Check boundary intersection with circle y^2 + z^2 = r_tunnel^2
        # Parametric equation of segment: p(t) = mid + t * d, t in [-l/2, l/2]
        # Intersection: (mid_y + t*dy)^2 + (mid_z + t*dz)^2 = r_tunnel^2
        # t^2 (dy^2 + dz^2) + 2*t*(mid_y*dy + mid_z*dz) + (mid_y^2 + mid_z^2 - r_tunnel^2) = 0
        # Since dy^2 + dz^2 = 1 (unit direction):
        # t^2 + 2*b*t + c = 0 where b = mid_y*dy + mid_z*dz, c = mid_y^2 + mid_z^2 - r_tunnel^2
        b = mid_y * dy + mid_z * dz
        c_coeff = mid_y**2 + mid_z**2 - r_tunnel**2
        discriminant = b**2 - c_coeff
        
        t0, t1 = -l/2.0, l/2.0
        
        if discriminant > 0:
            sqrt_disc = np.sqrt(discriminant)
            root1 = -b - sqrt_disc
            root2 = -b + sqrt_disc
            
            # The intersection points are at parametric coordinates root1 and root2
            # Clip the segment endpoints to circular boundary
            t0_clipped = max(-l/2.0, root1)
            t1_clipped = min(l/2.0, root2)
            
            # Recompute observed endpoints
            clip_p0_y = mid_y + t0_clipped * dy
            clip_p0_z = mid_z + t0_clipped * dz
            clip_p1_y = mid_y + t1_clipped * dy
            clip_p1_z = mid_z + t1_clipped * dz
            
            obs_l = np.sqrt((clip_p1_y - clip_p0_y)**2 + (clip_p1_z - clip_p0_z)**2)
            
            # Classify censoring based on clipping action
            clipped_p0 = (t0_clipped > -l/2.0)
            clipped_p1 = (t1_clipped < l/2.0)
            
            if clipped_p0 and clipped_p1:
                cens_type = 2  # Both-end clipped
            elif clipped_p0 or clipped_p1:
                cens_type = 1  # One-end clipped
            else:
                cens_type = 0  # Uncensored
        else:
            # No intersection (mathematical edge case, should not happen as mid is inside circular boundary)
            obs_l = l
            cens_type = 0
            
        # 4. Apply Truncation limit: if visible length is below c_truncation, it cannot be observed
        if obs_l >= c_truncation:
            observed_lengths.append(obs_l)
            censoring_types.append(cens_type)
            true_lengths_observed.append(l)
            dip_angles_observed.append(dip)
            
    return (
        np.array(true_lengths_observed),
        np.array(observed_lengths),
        np.array(censoring_types),
        np.array(dip_angles_observed)
    )


def main():
    print("=" * 80)
    print(" Hekmatnejad et al. (2018) Trace Length Inversion Pipeline Validation")
    print("=" * 80)
    
    # Configure parameters
    mu_true = 0.6          # True ln(L) mean
    sigma_true = 0.4       # True ln(L) std (Average length = exp(0.6 + 0.5*0.16) = 1.97m)
    c_truncation = 0.2     # 20cm resolution limit
    r_tunnel = 5.0         # 5m radius tunnel face
    
    output_dir = os.path.join(_here, "storage", "output", "hekmatnejad_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Generating biased synthetic trace database...")
    print(f"    - True Lognormal parameters: mu = {mu_true}, sigma = {sigma_true}")
    print(f"    - Truncation limit (c)     : {c_truncation} m")
    print(f"    - Circular Tunnel radius (R): {r_tunnel} m")
    
    true_l, obs_l, censoring, dips = generate_synthetic_biased_traces(
        n_population=6000,
        mu=mu_true,
        sigma=sigma_true,
        c_truncation=c_truncation,
        r_tunnel=r_tunnel,
        seed=42
    )
    
    print(f"\n[*] Successfully generated trace observations:")
    print(f"    - Total Visible Traces       : {len(obs_l)} items")
    print(f"    - Uncensored Traces (Type 0) : {np.sum(censoring == 0)} items ({np.mean(censoring == 0)*100:.1f}%)")
    print(f"    - One-end Clipped (Type 1)   : {np.sum(censoring == 1)} items ({np.mean(censoring == 1)*100:.1f}%)")
    print(f"    - Both-end Clipped (Type 2)  : {np.sum(censoring == 2)} items ({np.mean(censoring == 2)*100:.1f}%)")
    
    # Initialize the Estimator
    estimator = HekmatnejadEstimator(
        min_truncation=c_truncation,
        max_weight=10.0,
        apply_orientation_bias=True,
        dip_in_degrees=True
    )
    
    print(f"\n[*] Executing the Hekmatnejad Inverse Reconstruction Pipeline...")
    results = estimator.run_inversion_pipeline(obs_l, censoring, dips)
    
    # Extract results
    cdf_fun = results["cdf_function"]
    pdf_fun = results["pdf_function"]
    
    # Calculate some quantitative metrics
    l_grid = np.linspace(c_truncation, np.max(obs_l), 1000)
    
    # Analytical True Lognormal CDF and PDF for strict validation
    # Note: Because of orientation bias and truncation, we want to see how well we reconstruct 
    # the true length distribution of the physical fractures that intersect the tunnel.
    from scipy.stats import lognorm
    # scipy lognorm parameterization: s = std, scale = exp(mean)
    true_lognorm_dist = lognorm(s=sigma_true, scale=np.exp(mu_true))
    
    # Let's compute RMSE between our reconstructed CDF and the True sampled CDF
    # For a fair comparison, the true population ECDF above truncation limit is computed
    sorted_true = np.sort(true_l)
    ecdf_true_vals = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
    
    # Reconstructed CDF at sorted true lengths
    recon_cdf_at_true = cdf_fun(sorted_true)
    cdf_rmse = np.sqrt(np.mean((recon_cdf_at_true - ecdf_true_vals)**2))
    
    print(f"\n" + "-" * 50)
    print(" INVERSION ACCURACY METRICS")
    print("-" * 50)
    print(f"  * Reconstructed CDF vs True sampled CDF RMSE: {cdf_rmse:.5f}")
    
    # Check if the PDF integrates to ~1.0
    dx = l_grid[1] - l_grid[0]
    pdf_integral = np.sum(pdf_fun(l_grid)) * dx
    print(f"  * Reconstructed PDF Area Under Curve (AUC)  : {pdf_integral:.4f}")
    print("-" * 50)
    
    # Generate high-quality visual comparative plot
    plot_path = os.path.join(output_dir, "hekmatnejad_inversion_validation.png")
    
    # Override standard plotting to also show the Ground-Truth analytical CDF and PDF curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#fafafa")
    
    c_raw = "#d95f02"       # terracotta
    c_corrected = "#1b9e77" # premium teal
    c_pdf = "#7570b3"       # indigo
    c_true = "#252525"      # dark gray (Ground Truth)
    
    # PANEL 1: PDF comparison
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    ax1.hist(obs_l, bins=30, density=True, alpha=0.25, color=c_raw, edgecolor=c_raw, label="Observed Traces (Biased/Clipped)")
    ax1.hist(results["filtered_lengths"], bins=30, density=True, weights=results["filtered_weights"], alpha=0.4, color=c_corrected, edgecolor=c_corrected, label="Weighted Bias-Corrected")
    
    l_plot = np.linspace(c_truncation, np.max(obs_l) * 1.05, 500)
    ax1.plot(l_plot, pdf_fun(l_plot), color=c_pdf, linewidth=3.0, label="Inverted True PDF $f(l)$")
    
    # True analytical lognormal PDF conditioned on L >= c_truncation
    true_pdf_cond = true_lognorm_dist.pdf(l_plot) / (1.0 - true_lognorm_dist.cdf(c_truncation))
    ax1.plot(l_plot, true_pdf_cond, color=c_true, linewidth=2.0, linestyle="--", label="Ground Truth Analytical PDF")
    
    ax1.set_title("Probability Density Function (PDF) Inversion", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Trace Length $l$ (m)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")
    
    # PANEL 2: CDF comparison
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    ax2.step(np.sort(obs_l), np.arange(1, len(obs_l)+1)/len(obs_l), color=c_raw, alpha=0.5, linewidth=1.5, where="post", label="Empirical ECDF (Observed)")
    ax2.plot(l_plot, cdf_fun(l_plot), color=c_corrected, linewidth=3.0, label="Inverted True CDF $F(l)$")
    
    # True analytical lognormal CDF conditioned on L >= c_truncation
    true_cdf_cond = (true_lognorm_dist.cdf(l_plot) - true_lognorm_dist.cdf(c_truncation)) / (1.0 - true_lognorm_dist.cdf(c_truncation))
    ax2.plot(l_plot, true_cdf_cond, color=c_true, linewidth=2.0, linestyle="--", label="Ground Truth Analytical CDF")
    
    if results["lifelines_verification"] is not None:
        ll_l, ll_s = results["lifelines_verification"]
        ax2.step(ll_l, 1.0 - ll_s, color="#e7298a", linestyle=":", linewidth=2.2, where="post", label="lifelines KMF Verification")
        
    ax2.set_title("Cumulative Distribution Function (CDF) Inversion", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Trace Length $l$ (m)", fontsize=11)
    ax2.set_ylabel("Cumulative Probability", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[*] Premium comparative distribution figure saved to: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print(" PIPELINE INVERSION COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
