import os
import sys
import numpy as np
import h5py
import math

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)

from dfnrec.size_intensity.chord_likelihood import censored_chord_log_likelihood, chord_pdf_ideal
from dfnrec.size_intensity.p32_estimator import estimate_size_model

def run_step_a(radii_set, gt_kr):
    print("\n--- Test A: Real HDF5 Radii MLE ---")
    R_min = 1.0
    n = len(radii_set)
    k_r_hill = n / np.sum(np.log(radii_set / R_min))
    
    kr_grid = np.linspace(0.5, 5.0, 100)
    best_ll = -1e10
    best_kr = None
    for kr in kr_grid:
        ll = n * math.log(kr) - (kr + 1.0) * np.sum(np.log(radii_set))
        if ll > best_ll:
            best_ll = ll
            best_kr = kr
    print(f"GT k_r: {gt_kr:.3f}")
    print(f"Hill Estimator k_r: {k_r_hill:.3f}")
    print(f"Grid MLE k_r: {best_kr:.3f}")

def run_step_b(gt_kr, n_samples=5000):
    print("\n--- Test B: Synthetic Full Chords k_r Inversion ---")
    rng = np.random.default_rng(42)
    r_min = 1.0
    r_max = 50.0
    k_obs = gt_kr - 1.0
    
    u = rng.uniform(0, 1, n_samples)
    radii = (r_min**(-k_obs) - u * (r_min**(-k_obs) - r_max**(-k_obs)))**(-1.0 / k_obs)
    
    u_intercept = rng.uniform(0, 1, n_samples)
    chords = 2.0 * radii * np.sqrt(1.0 - u_intercept**2)
    chords = np.clip(chords, 0.1, 2.0 * r_max - 1e-4)
    contained = np.ones_like(chords, dtype=bool)
    
    alpha_grid = np.linspace(1.5, 6.0, 91)
    best_ll = -1e10
    best_alpha = None
    for alpha in alpha_grid:
        ll = censored_chord_log_likelihood(chords, contained, alpha, r_min, r_max, L_min=0.01)
        if ll > best_ll:
            best_ll = ll
            best_alpha = alpha
    print(f"GT k_r: {gt_kr:.3f} (alpha: {gt_kr+1.0:.3f})")
    print(f"Estimated alpha: {best_alpha:.3f} => k_r: {best_alpha - 1.0:.3f}")

def run_step_c(gt_kr, n_samples=5000):
    print("\n--- Test C: Synthetic Clipped Chords k_r Inversion ---")
    rng = np.random.default_rng(42)
    r_min = 1.0
    r_max = 50.0
    k_obs = gt_kr - 1.0
    
    u = rng.uniform(0, 1, n_samples)
    radii = (r_min**(-k_obs) - u * (r_min**(-k_obs) - r_max**(-k_obs)))**(-1.0 / k_obs)
    
    u_intercept = rng.uniform(0, 1, n_samples)
    full_chords = 2.0 * radii * np.sqrt(1.0 - u_intercept**2)
    
    # Clip chords by observation window of typical size W
    W = 4.0
    chords = np.minimum(full_chords, W)
    contained = (full_chords <= W)
    
    # Filter very small chords
    mask = (chords >= 0.1)
    chords = chords[mask]
    contained = contained[mask]
    
    alpha_grid = np.linspace(1.5, 6.0, 91)
    best_ll = -1e10
    best_alpha = None
    for alpha in alpha_grid:
        ll = censored_chord_log_likelihood(chords, contained, alpha, r_min, r_max, L_min=0.1)
        if ll > best_ll:
            best_ll = ll
            best_alpha = alpha
    print(f"GT k_r: {gt_kr:.3f} (alpha: {gt_kr+1.0:.3f})")
    print(f"Estimated alpha: {best_alpha:.3f} => k_r: {best_alpha - 1.0:.3f}")

def run_step_d():
    import pandas as pd
    csv_path = "storage/output/ground_truth_traces_with_normals.csv"
    if not os.path.exists(csv_path):
        print("CSV not found")
        return
    df = pd.read_csv(csv_path)
    df_sub = df[df["set_id"] == 1]
    print("\n--- Test D: Real Traces Info (S1) ---")
    print("Columns:", df.columns.tolist())
    print("Count:", len(df_sub))
    
    dx = df_sub["p0_x"] - df_sub["p1_x"]
    dy = df_sub["p0_y"] - df_sub["p1_y"]
    dz = df_sub["p0_z"] - df_sub["p1_z"]
    chord_lengths = np.sqrt(dx**2 + dy**2 + dz**2).values
    is_contained = np.ones_like(chord_lengths, dtype=bool)
    
    alpha_grid = np.linspace(1.5, 6.0, 10)
    r_min_grid = np.linspace(0.1, 1.5, 8)
    r_max = 30.0
    
    print("\n2D LL surface (alpha vs r_min):")
    header = "r_min: " + " ".join(f"{r:.2f}     " for r in r_min_grid)
    print(header)
    for alpha in alpha_grid:
        row = f"a={alpha:.2f}: "
        for r_min in r_min_grid:
            ll = censored_chord_log_likelihood(chord_lengths, is_contained, alpha, r_min, r_max, L_min=0.1)
            row += f"{ll:8.2f} "
        print(row)

def main():
    h5_path = os.path.join(_root, "storage", "data", "dfn_export_for_python.h5")
    with h5py.File(h5_path, "r") as f:
        radii = f["/fractures/radii"][:]
        set_ids = f["/fractures/set_id"][:].flatten()
        
    mask = (set_ids == 1)
    s1_radii = radii[mask]
    run_step_a(s1_radii, 2.85)
    
    run_step_b(2.85)
    run_step_c(2.85)
    run_step_d()

if __name__ == "__main__":
    main()
