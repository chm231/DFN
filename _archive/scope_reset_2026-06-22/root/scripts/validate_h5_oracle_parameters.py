import os
import sys
import numpy as np
import h5py
import math

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)

from dfnrec.geometry.vector import trend_plunge_from_normal, axial_angle, normal_from_trend_plunge

def main():
    h5_path = os.path.join(_root, "storage", "data", "dfn_export_for_python.h5")
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found")
        sys.exit(1)
        
    print(f"Reading HDF5 DFN from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        normals = f["/fractures/normals"][:]
        radii = f["/fractures/radii"][:].flatten()
        set_ids = f["/fractures/set_id"][:].flatten()
        
    gt_orientations = {
        1: (338.1, 4.5),
        2: (100.4, 0.2),
        3: (212.9, 0.9),
        4: (3.3, 62.1),
        5: (243.0, 24.4),
    }
    
    gt_kr = {
        1: 2.85,
        2: 3.04,
        3: 3.01,
        5: 3.602,
    }
    
    print("\n" + "="*95)
    print(f"{'Set':<5} | {'Min R':>8} | {'Med R':>8} | {'Max R':>8} | {'Watson Mean Trend/Plunge':<25} | {'GT Orie':<10} | {'AngErr':>7} | {'k_r MLE':>8} | {'GT k_r':>8}")
    print("-" * 95)
    
    for s_idx in [1, 2, 3, 4, 5]:
        mask = (set_ids == s_idx)
        s_radii = radii[mask]
        s_normals = normals[mask]
        
        if len(s_radii) == 0:
            print(f"S{s_idx:<4} | No data")
            continue
            
        # Radii stats
        r_min = float(np.min(s_radii))
        r_med = float(np.median(s_radii))
        r_max = float(np.max(s_radii))
        
        # Watson Mean Pole
        T = np.zeros((3, 3))
        for n_vec in s_normals:
            n_norm = n_vec / np.linalg.norm(n_vec)
            T += np.outer(n_norm, n_norm)
        T /= len(s_normals)
        evals, evecs = np.linalg.eigh(T)
        mean_dir = evecs[:, 2]
        
        trend, plunge = trend_plunge_from_normal(mean_dir)
        
        # Angular error against GT
        gt_t, gt_p = gt_orientations[s_idx]
        n_gt = normal_from_trend_plunge(gt_t, gt_p)
        ang_err = math.degrees(axial_angle(mean_dir, n_gt))
        
        # k_r MLE
        if s_idx == 4:
            # Exponential Set S4: f(R) = lambda * exp(-lambda*(R - R_min))
            mean_r = float(np.mean(s_radii))
            kr_str = f"mean={mean_r:.2f}"
            gt_kr_str = "mean=4.000"
        else:
            # Power-law k_r MLE: k_r_hat = n / sum(log(R_i / R_min))
            R_min_support = 1.0
            valid_radii = s_radii[s_radii >= R_min_support]
            if len(valid_radii) > 0:
                kr_hat = len(valid_radii) / np.sum(np.log(valid_radii / R_min_support))
                kr_str = f"{kr_hat:.3f}"
            else:
                kr_str = "N/A"
            gt_kr_str = f"{gt_kr[s_idx]:.3f}"
            
        print(f"S{s_idx:<4} | {r_min:8.3f} | {r_med:8.3f} | {r_max:8.3f} | {trend:6.2f} / {plunge:5.2f}            | {gt_t:5.1f}/{gt_p:4.1f} | {ang_err:6.3f}° | {kr_str:>8} | {gt_kr_str:>8}")
        
    print("="*95)

if __name__ == "__main__":
    main()
