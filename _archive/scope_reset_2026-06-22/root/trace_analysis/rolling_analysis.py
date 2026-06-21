import os
import sys
import h5py
import numpy as np
from scipy.stats import lognorm

_here = r"c:\Users\user\OneDrive\2026-1\3D DFN modeling\trace_analysis"
_parent = r"c:\Users\user\OneDrive\2026-1\3D DFN modeling"
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_reconstruction_unified import ExcavationFace, classify_censoring, ParametricMLEEstimator
from run_real_hekmatnejad_faces import extract_real_traces_with_truth

def main():
    hdf5_path = os.path.join(_parent, "storage", "data", "dfn_export_for_python.h5")
    dat_path = os.path.join(_parent, "storage", "data", "단면_폴리곤.dat")
    
    # 1. Load tunnel polygon
    poly_y, poly_z = load_tunnel_polygon_from_dat(dat_path)
    poly_yz = np.column_stack([poly_y, poly_z])
    
    # 2. Load DFN data
    with h5py.File(hdf5_path, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        gt_radii = f['/fractures/radii'][:].ravel()
        gt_set_id = (f['/fractures/set_id'][:].ravel() if '/fractures/set_id' in f 
                     else np.ones(len(gt_radii), dtype=np.uint16))
        
        gt_centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        gt_normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        
    print(f"[*] Loaded 3D DFN database: {len(gt_radii):,} fractures.")
    
    # Define rolling windows (stride = 3m, 4 faces each)
    windows = [
        [0.0, 3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0, 12.0],
        [6.0, 9.0, 12.0, 15.0],
        [9.0, 12.0, 15.0, 18.0],
        [12.0, 15.0, 18.0, 21.0],
        [15.0, 18.0, 21.0, 24.0],
        [18.0, 21.0, 24.0, 27.0],
        [21.0, 24.0, 27.0, 30.0]
    ]
    
    results = []
    
    for idx, x_pos in enumerate(windows):
        print(f"\n[+] Running rolling window {idx+1}/8: x = {x_pos} m...")
        
        faces = []
        for i, xp in enumerate(x_pos):
            faces.append(ExcavationFace(
                face_id=i + 1,
                x_face=float(xp),
                tunnel_polygon_yz=poly_yz,
                advance_step=3.0 if i > 0 else 0.0
            ))
            
        # Extract traces
        obs_traces, true_unclipped_lengths = extract_real_traces_with_truth(
            gt_centers, gt_normals, gt_radii, gt_set_id, faces
        )
        
        # Censoring classification
        for face in faces:
            classify_censoring(obs_traces, face, tolerance=0.10)
            
        obs_lengths = np.array([t.length for t in obs_traces])
        censoring_types = np.array([t.censoring_class for t in obs_traces])
        true_lengths = np.array([true_unclipped_lengths[t.trace_id] for t in obs_traces])
        
        c_truncation = 0.15
        
        # Fit with self_calibrate=True
        estimator = ParametricMLEEstimator(
            min_truncation=c_truncation,
            correct_size_bias=False,
            window_diameter=10.0,
            self_calibrate=True
        )
        
        res = estimator.fit(obs_lengths, censoring_types)
        cdf_fun = res["cdf_function"]
        
        # Validation ECDF
        valid_mask = true_lengths >= c_truncation
        valid_true_lengths = true_lengths[valid_mask]
        sorted_true = np.sort(valid_true_lengths)
        ecdf_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
        
        rmse = np.sqrt(np.mean((cdf_fun(sorted_true) - ecdf_true)**2))
        
        # Calculate Observed Proportions for the table
        n_total = len(obs_lengths[obs_lengths >= c_truncation])
        t0 = np.sum(censoring_types[obs_lengths >= c_truncation] == 0)
        t1 = np.sum(censoring_types[obs_lengths >= c_truncation] == 1)
        t2 = np.sum(censoring_types[obs_lengths >= c_truncation] == 2)
        
        results.append({
            "window": f"{x_pos[0]:.0f} ~ {x_pos[-1]:.0f}m",
            "N": n_total,
            "t0_pct": f"{t0/n_total*100:.1f}%",
            "t1_pct": f"{t1/n_total*100:.1f}%",
            "t2_pct": f"{t2/n_total*100:.1f}%",
            "d1": f"+{estimator.d1:.1f}m",
            "d2": f"+{estimator.d2:.1f}m",
            "RMSE": f"{rmse:.5f}"
        })
        
    print("\n\n" + "="*80)
    print(" BATCH ROLLING WINDOW INVERSION VALIDATION RESULT TABLE")
    print("="*80)
    print("| 분석 터널 구간 | 총 흔적 개수 (N) | Type 0 비율 | Type 1 비율 | Type 2 비율 | Type 1 보정치 (d1) | Type 2 보정치 (d2) | 참값 대비 오차 (RMSE) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['window']} | {r['N']} | {r['t0_pct']} | {r['t1_pct']} | {r['t2_pct']} | {r['d1']} | {r['d2']} | **{r['RMSE']}** |")
    print("="*80)

if __name__ == '__main__':
    main()
