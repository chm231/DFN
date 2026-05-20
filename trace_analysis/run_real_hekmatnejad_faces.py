"""
Rigorous Inversion Analysis on 4 Consecutive Real Tunnel Faces using Hekmatnejad et al. (2018).
Loads storage DFN/tunnel datasets, extracts real physical traces, performs non-parametric inversion,
and renders individual 2D trace maps for all 4 excavation faces.
"""
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Tuple

# Insert parent directories
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_reconstruction.trace_types import ExcavationFace, FaceTrace
from trace_reconstruction.trace_preprocessor import classify_censoring
from trace_reconstruction.hekmatnejad_estimation import HekmatnejadEstimator
from trace_reconstruction.mle_estimation import ParametricMLEEstimator


def calculate_kappa_tensor_aligned(normals):
    """
    Estimates the Fisher concentration parameter kappa for a set of normals
    using dynamic orientation tensor alignment (PCA axis) on the hemisphere.
    Returns: (kappa, R_mag)
    """
    N = normals.shape[0]
    if N <= 1:
        return 0.0, 0.0
    
    # 1. Force unit vector normalization with numerical safety
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    n = normals / norms_safe
    zero_mask = (norms.ravel() <= 1e-12)
    if np.any(zero_mask):
        n[zero_mask] = np.array([0.0, 0.0, 1.0])
    
    # 2. Compute orientation tensor and principal axis (eigenvector of T)
    tensor = np.einsum('ij,ik->ijk', n, n)
    T = np.sum(tensor, axis=0) / N
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 3. Dynamic hemisphere alignment based on principal axis projection
    dots = np.dot(n, principal_axis)
    flip_mask = dots < 0
    n = n.copy()
    n[flip_mask] *= -1
    
    # 4. Resultant vector summation & Fisher Kappa computation
    R_vector = np.sum(n, axis=0)
    R_mag = np.linalg.norm(R_vector)
    
    denominator = N - R_mag
    kappa = 1e6 if denominator < 1e-6 else (N - 1) / denominator
        
    return kappa, R_mag


def extract_real_traces_with_truth(
    centers: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    set_ids: np.ndarray,
    faces: list
) -> Tuple[list, dict]:
    """
    Extracts observed traces on consecutive excavation faces from the real 3D DFN database,
    storing the actual unclipped length (L_full = 2*sqrt(R^2 - d^2)) as the ground truth.
    """
    from trace_reconstruction.forward_simulator import clip_line_segment_to_polygon
    
    obs_traces = []
    true_unclipped_lengths = {} # trace_id -> unclipped length L_full
    tid = 1
    
    for face in faces:
        x_f = face.x_face
        poly = face.tunnel_polygon_yz
        
        for i in range(len(radii)):
            cx, cy, cz = centers[i]
            nx, ny, nz = normals[i]
            radius = radii[i]
            set_id = int(set_ids[i])
            
            ny_z_sq = ny**2 + nz**2
            if ny_z_sq < 1e-12:
                continue # Fracture plane parallel to face
                
            C_rhs = nx * (cx - x_f) + ny * cy + nz * cz
            dist_to_line = abs(x_f - cx) / np.sqrt(ny_z_sq)
            
            if dist_to_line >= radius:
                continue # No physical intersection
                
            # Midpoint and ends of full unclipped chord
            factor = (ny * cy + nz * cz - C_rhs) / ny_z_sq
            y_mid = cy - ny * factor
            z_mid = cz - nz * factor
            mid_pt = np.array([y_mid, z_mid])
            
            chord_half_len = np.sqrt(radius**2 - dist_to_line**2)
            d_line = np.array([-nz, ny]) / np.sqrt(ny_z_sq)
            
            p0 = mid_pt - chord_half_len * d_line
            p1 = mid_pt + chord_half_len * d_line
            
            # Clip segment to tunnel polygon
            clipped = clip_line_segment_to_polygon(p0, p1, poly)
            
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
                    parent_fracture_id=i
                )
                t.set_id = set_id
                obs_traces.append(t)
                
                # Geometrical true unclipped 2D trace length
                true_unclipped_lengths[tid] = 2.0 * chord_half_len
                tid += 1
                
    return obs_traces, true_unclipped_lengths


def plot_single_face_trace_map(
    face: ExcavationFace,
    traces: list,
    save_path: str
):
    """
    Plots a publication-grade, beautiful 2D trace map for a single excavation face.
    Distinctly categorizes and styles traces based on their Censoring Class (0, 1, 2).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")
    
    # 1. Plot Tunnel Polygon
    poly = face.tunnel_polygon_yz
    poly_closed = np.vstack([poly, poly[0]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="#333333", linewidth=2.5, label="Tunnel Section Boundary")
    ax.fill(poly_closed[:, 0], poly_closed[:, 1], color="#f0f0f0", alpha=0.3)
    
    # Elegant custom HSL tailored colors for Censoring types
    c_type0 = "#1b9e77" # Deep premium teal (Contained / Uncensored)
    c_type1 = "#377eb8" # Ocean blue (One-end clipped)
    c_type2 = "#e41a1c" # Crimson red (Both-end clipped)
    
    type0_drawn = False
    type1_drawn = False
    type2_drawn = False
    
    # 2. Draw individual trace segments
    for t in traces:
        if t.face_id != face.face_id:
            continue
            
        y = [t.p0_y, t.p1_y]
        z = [t.p0_z, t.p1_z]
        
        cc = t.censoring_class
        if cc == 0:
            color = c_type0
            ls = "-"
            lw = 2.2
            label = "Type 0: Contained" if not type0_drawn else ""
            type0_drawn = True
        elif cc == 1:
            color = c_type1
            ls = "--"
            lw = 2.0
            label = "Type 1: One-end Clipped" if not type1_drawn else ""
            type1_drawn = True
        else:
            color = c_type2
            ls = ":"
            lw = 2.5
            label = "Type 2: Both-end Clipped" if not type2_drawn else ""
            type2_drawn = True
            
        ax.plot(y, z, color=color, linestyle=ls, linewidth=lw, label=label)

    # 3. Compute trace intensity metrics (P21) on this specific face
    tunnel_area = 0.0
    # Area of polygon using Shoelace formula
    n_pts = len(poly)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        tunnel_area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    tunnel_area = 0.5 * abs(tunnel_area)
    
    face_traces = [t for t in traces if t.face_id == face.face_id]
    tot_len = sum(t.length for t in face_traces)
    p21 = tot_len / tunnel_area if tunnel_area > 0 else 0.0
    
    # 4. Styling and annotations
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(f"2D Trace Map at Face {face.face_id} (x = {face.x_face:.1f}m)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Tunnel Y Coordinate (m)", fontsize=11)
    ax.set_ylabel("Tunnel Z Coordinate (m)", fontsize=11)
    
    # Professional legend and information box
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")
    
    info_text = (
        f"Face ID: {face.face_id}\n"
        f"X Position: {face.x_face:.1f} m\n"
        f"Total Traces: {len(face_traces)} ea\n"
        f"Trace Intensity P21: {p21:.4f} m/m²"
    )
    ax.text(
        0.05, 0.05, info_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
        verticalalignment="bottom"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Trace map figure successfully saved to: {save_path}")


def main():
    print("=" * 80)
    print(" Hekmatnejad et al. (2018) Inversion Analysis on 4 Real Excavation Faces")
    print("=" * 80)
    
    # Define file paths
    hdf5_path = os.path.join(_parent, "storage", "data", "dfn_export_for_python.h5")
    dat_path = os.path.join(_parent, "storage", "data", "단면_폴리곤.dat")
    output_dir = os.path.join(_here, "storage", "output", "hekmatnejad_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load tunnel section boundary
    print(f"[*] Parsing tunnel polygon .dat boundary: {dat_path}")
    poly_y, poly_z = load_tunnel_polygon_from_dat(dat_path)
    poly_yz = np.column_stack([poly_y, poly_z])
    print(f"    -> Parsed {len(poly_yz)} coordinate nodes.")
    
    # 2. Load 3D DFN database
    print(f"[*] Loading actual 3D DFN database: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        gt_radii = f['/fractures/radii'][:].ravel()
        gt_set_id = (f['/fractures/set_id'][:].ravel() if '/fractures/set_id' in f 
                     else np.ones(len(gt_radii), dtype=np.uint16))
        
        gt_centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        gt_normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        
    print(f"    -> Real 3D DFN contains {len(gt_radii):,} fractures.")
    
    # 3. Define 4 consecutive excavation faces
    x_positions = [6.0, 9.0, 12.0, 15.0]
    faces = []
    for i, x_pos in enumerate(x_positions):
        faces.append(ExcavationFace(
            face_id=i + 1,
            x_face=float(x_pos),
            tunnel_polygon_yz=poly_yz,
            advance_step=3.0 if i > 0 else 0.0
        ))
    print(f"    -> 4 Analysis faces established: x = {x_positions} m")
    
    # 4. Geometrical Trace extraction and true unclipped lengths mapping
    print("\n[*] Extruding discs & intersecting with 4 face planes...")
    obs_traces, true_unclipped_lengths = extract_real_traces_with_truth(
        gt_centers, gt_normals, gt_radii, gt_set_id, faces
    )
    print(f"    -> Successfully extracted {len(obs_traces)} real intersection traces.")
    
    # 5. Classify censoring boundaries
    print("[*] Performing boundary censoring distance classification (Type 0 / 1 / 2)...")
    for face in faces:
        classify_censoring(obs_traces, face, tolerance=0.10)
        
    # Print real statistics per face
    for face in faces:
        ft = [t for t in obs_traces if t.face_id == face.face_id]
        t0 = sum(1 for t in ft if t.censoring_class == 0)
        t1 = sum(1 for t in ft if t.censoring_class == 1)
        t2 = sum(1 for t in ft if t.censoring_class == 2)
        print(f"    - Face {face.face_id} (x={face.x_face}m): total={len(ft)}, Type0={t0}, Type1={t1}, Type2={t2}")
        
    # 6. Render individual 2D Trace maps for all 4 faces
    print("\n[*] Rendering high-fidelity 2D Trace maps...")
    for idx, face in enumerate(faces):
        plot_path = os.path.join(output_dir, f"trace_map_face_{face.face_id}.png")
        plot_single_face_trace_map(face, obs_traces, plot_path)
        
    # 7. Collect statistical parameters and perform Set-by-Set independent MLE and Terzaghi conversion
    unique_set_ids = sorted(list(set(t.set_id for t in obs_traces)))
    print(f"\n[*] Found {len(unique_set_ids)} unique fracture sets in the traces: {unique_set_ids}")

    # Calculate face polygon area using Shoelace formula
    n_pts = len(poly_yz)
    tunnel_area = 0.0
    for i in range(n_pts):
        j = (i + 1) % n_pts
        tunnel_area += poly_yz[i, 0] * poly_yz[j, 1] - poly_yz[j, 0] * poly_yz[i, 1]
    tunnel_area = 0.5 * abs(tunnel_area)
    total_sampling_area = len(faces) * tunnel_area

    # Calculate tunnel slab volume for true P32 verification
    x_positions_float = [f.x_face for f in faces]
    x_min, x_max = min(x_positions_float), max(x_positions_float)
    slab_span = x_max - x_min
    slab_volume = tunnel_area * slab_span

    # Localized 50m Cube Crop Box where the tunnel excavation faces and traces reside
    crop_limit = 25.0
    db_volume = (2.0 * crop_limit) ** 3
    print(f"[*] Set localized 3D DFN verification volume (50m Cube): {db_volume:.3f} m3")


    # Grid search / Truncation threshold limit
    c_truncation = 0.1

    # Storage for per-set results
    set_results = {}

    for sid in unique_set_ids:
        print(f"\n==================================================")
        print(f" PROCESSING FRACTURE SET {sid}")
        print(f"==================================================")

        # Slice observed data for this specific set
        set_traces = [t for t in obs_traces if t.set_id == sid]
        n_traces = len(set_traces)
        print(f"  * Observed trace sample size: {n_traces} traces")

        if n_traces < 5:
            print(f"  [Warning] Set {sid} has too few traces ({n_traces}) for robust MLE. Skipping.")
            continue

        obs_lengths_set = np.array([t.length for t in set_traces])
        censoring_types_set = np.array([t.censoring_class for t in set_traces])
        true_lengths_set = np.array([true_unclipped_lengths[t.trace_id] for t in set_traces])

        # 1. Direct MLE Estimator (Censoring & Truncation correction, but no Size-bias shift)
        estimator_direct = ParametricMLEEstimator(
            min_truncation=c_truncation,
            correct_size_bias=False,
            window_diameter=10.0,
            self_calibrate=True
        )
        print(f"  [*] Executing Direct Parametric MLE Inversion (Self-Calibrating)...")
        res_direct = estimator_direct.fit(obs_lengths_set, censoring_types_set)
        
        # 2. Unbiased MLE Estimator (Censoring & Truncation & Size-bias shift corrected)
        estimator_unbiased = ParametricMLEEstimator(
            min_truncation=c_truncation,
            correct_size_bias=True,
            window_diameter=10.0,
            self_calibrate=True
        )
        print(f"  [*] Executing Unbiased Parametric MLE Inversion (Self-Calibrating)...")
        res_unbiased = estimator_unbiased.fit(obs_lengths_set, censoring_types_set)

        # Inversion Quality Metrics against ground-truth unclipped trace lengths of intersected population
        valid_mask = true_lengths_set >= c_truncation
        valid_true_lengths = true_lengths_set[valid_mask]
        sorted_true = np.sort(valid_true_lengths)
        ecdf_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true)

        cdf_fun_direct = res_direct["cdf_function"]
        pdf_fun_direct = res_direct["pdf_function"]
        cdf_fun_unbiased = res_unbiased["cdf_function"]
        pdf_fun_unbiased = res_unbiased["pdf_function"]

        rmse_direct = np.sqrt(np.mean((cdf_fun_direct(sorted_true) - ecdf_true)**2)) if len(sorted_true) > 0 else 0.0
        rmse_unbiased = np.sqrt(np.mean((cdf_fun_unbiased(sorted_true) - ecdf_true)**2)) if len(sorted_true) > 0 else 0.0

        # Terzaghi Orientation Bias calculations
        sin_thetas = []
        for t in set_traces:
            parent_id = t.parent_fracture_id
            ny, nz = gt_normals[parent_id, 1], gt_normals[parent_id, 2]
            sin_theta = np.sqrt(ny**2 + nz**2)
            sin_thetas.append(sin_theta)
        mean_sin_theta = np.mean(sin_thetas) if sin_thetas else 1.0

        # P21 Linear density on 2D tunnel face
        tot_len_obs = np.sum(obs_lengths_set)
        p21_est = tot_len_obs / total_sampling_area

        # P32 Volumetric density estimation via baseline Terzaghi correction (Unconstrained)
        p32_est = p21_est / mean_sin_theta

        # --- NEW: Prior-Constrained Geostatistical Scale Correction (C_scale) ---
        rmin_3d = 1.0  # Physical minimum radius of the 3D database
        c_val = c_truncation
        r_cutoff_2d = c_val / 2.0  # Theoretical minimum radius that can spawn a trace of length c
        
        c_scale = 1.0
        best_name = res_unbiased["dist_name"]
        params = res_unbiased["params"]
        
        if best_name == "Lognormal":
            # Recover unbiased 3D lognormal parameters
            mu_b, sigma_b = params[0], params[1]
            sigma_R = sigma_b
            mu_R = mu_b - sigma_b**2
            
            from scipy.stats import norm
            # Compute truncated second moments analytically
            term_num = (-np.log(rmin_3d) + mu_R + 2.0 * (sigma_R**2)) / sigma_R
            term_den = (-np.log(r_cutoff_2d) + mu_R + 2.0 * (sigma_R**2)) / sigma_R
            c_scale = norm.cdf(term_num) / norm.cdf(term_den) if norm.cdf(term_den) > 0 else 1.0
            
        elif best_name == "Exponential":
            lam = params[0]
            # Analytical truncated second moment ratio for Expon: e^{-lambda x} * (x^2 + 2x/lambda + 2/lambda^2)
            num_ex = np.exp(-lam * rmin_3d) * (rmin_3d**2 + 2.0*rmin_3d/lam + 2.0/(lam**2))
            den_ex = np.exp(-lam * r_cutoff_2d) * (r_cutoff_2d**2 + 2.0*r_cutoff_2d/lam + 2.0/(lam**2))
            c_scale = num_ex / den_ex if den_ex > 0 else 1.0
            
        elif best_name == "Pareto":
            alpha_b = params[0]
            alpha_R = alpha_b + 1.0  # size-bias recovery
            if alpha_R > 2.0:
                c_scale = (r_cutoff_2d / rmin_3d) ** (alpha_R - 2.0)
            else:
                c_scale = 1.0
                
        p32_est_constrained = p32_est * c_scale

        # True 3D P32 from 3D DFN database localized within the active 50m Crop Box
        mask_set = (gt_set_id == sid)
        mask_crop = (np.abs(gt_centers[:, 0]) <= crop_limit) & \
                    (np.abs(gt_centers[:, 1]) <= crop_limit) & \
                    (np.abs(gt_centers[:, 2]) <= crop_limit)
        mask_3d_local = mask_set & mask_crop
        total_area_3d_local = np.sum(np.pi * (gt_radii[mask_3d_local] ** 2))
        p32_true = total_area_3d_local / db_volume

        # --- NEW: Fisher Concentration Parameter (kappa) Inversion ---
        # 1. Collect parent normal vectors
        set_normals_list = []
        for t in set_traces:
            parent_id = t.parent_fracture_id
            set_normals_list.append(gt_normals[parent_id])
        set_normals_arr = np.array(set_normals_list)
        
        # 2. Compute Fisher concentration parameter kappa and R magnitude using orientation tensor alignment
        kappa_est, R_len = calculate_kappa_tensor_aligned(set_normals_arr)

        error_pct = abs(p32_est - p32_true) / p32_true * 100 if p32_true > 0 else 0.0
        error_pct_constrained = abs(p32_est_constrained - p32_true) / p32_true * 100 if p32_true > 0 else 0.0

        print(f"  [*] Set {sid} Results:")
        print(f"    - Optimized Offsets: d1 = {estimator_unbiased.d1:.3f}m, d2 = {estimator_unbiased.d2:.3f}m")
        print(f"    - Fit RMSE (Direct): {rmse_direct:.5f}")
        print(f"    - Fit RMSE (Unbiased): {rmse_unbiased:.5f}")
        print(f"    - Mean sin(theta)  : {mean_sin_theta:.4f}")
        print(f"    - Est P21 Density  : {p21_est:.4f} m/m2")
        print(f"    - Scale Fact C_scale: {c_scale:.5f}")
        print(f"    - Baseline P32     : {p32_est:.4f} m2/m3  (Error: {error_pct:.2f} %)")
        print(f"    - Constrained P32  : {p32_est_constrained:.4f} m2/m3  (Error: {error_pct_constrained:.2f} %)")
        print(f"    - True P32 Density : {p32_true:.4f} m2/m3")
        print(f"    - Resultant Vector |R|: {R_len:.4f}")
        print(f"    - Est Fisher Kappa : {kappa_est:.4f}")

        set_results[sid] = {
            'n_traces': n_traces,
            'obs_lengths': obs_lengths_set,
            'censoring_types': censoring_types_set,
            'valid_true_lengths': valid_true_lengths,
            'sorted_true': sorted_true,
            'ecdf_true': ecdf_true,
            'dist_name': res_direct['dist_name'],
            'params': res_unbiased['params'],
            'pdf_direct': pdf_fun_direct,
            'cdf_direct': cdf_fun_direct,
            'pdf_unbiased': pdf_fun_unbiased,
            'cdf_unbiased': cdf_fun_unbiased,
            'rmse_direct': rmse_direct,
            'rmse_unbiased': rmse_unbiased,
            'd1': estimator_unbiased.d1,
            'd2': estimator_unbiased.d2,
            'mean_sin_theta': mean_sin_theta,
            'p21': p21_est,
            'p32_est': p32_est,
            'p32_est_constrained': p32_est_constrained,
            'p32_true': p32_true,
            'error_pct': error_pct,
            'error_pct_constrained': error_pct_constrained,
            'R_len': R_len,
            'kappa_est': kappa_est
        }

    # Print final geostatistical summary table
    print("\n" + "=" * 115)
    print("                 GEOSTATISTICAL MULTI-SET INVERSION & P32 SUMMARY REPORT (PRIOR-CONSTRAINED)")
    print("=" * 115)
    print(f" { 'SET' : <5} | { 'TRACES' : <6} | { 'MODEL' : <10} | { 'd1 / d2' : <9} | { 'RMSE(Unb)' : <9} | { 'P21' : <6} | { 'P32(Base)' : <9} | { 'P32(Const)' : <10} | { 'P32(True)' : <9} | { 'ERR_BASE(%)' : <11} | { 'ERR_CONST(%)' : <11}")
    print("-" * 115)
    for sid, r in set_results.items():
        offset_str = f"{r['d1']:.1f}/{r['d2']:.1f}"
        print(f" Set{sid:<2} | {r['n_traces']:<6} | {r['dist_name']:<10} | {offset_str:<9} | {r['rmse_unbiased']:<9.5f} | {r['p21']:<6.4f} | {r['p32_est']:<9.4f} | {r['p32_est_constrained']:<10.4f} | {r['p32_true']:<9.4f} | {r['error_pct']:<11.2f} | {r['error_pct_constrained']:<11.2f}")
    print("=" * 115)

    # Print final orientation and Fisher concentration parameter table
    print("\n" + "=" * 80)
    print("                 ORIENTATION & FISHER CONCENTRATION (KAPPA) REPORT")
    print("=" * 80)
    print(f" { 'SET' : <5} | { 'TRACES (N)' : <10} | { 'R_MAG (|R|)' : <12} | { 'KAPPA (Est)' : <12} | { 'KAPPA (True)' : <12} | { 'KAPPA (Design)' : <14}")
    print("-" * 80)
    true_kappas = {1: 13.13, 2: 19.90, 3: 10.37, 4: 10.37, 5: 23.76}
    design_kappas = {1: 13.06, 2: 19.62, 3: 10.46, 4: 10.13, 5: 23.52}
    for sid, r in set_results.items():
        tk = true_kappas.get(sid, 0.0)
        dk = design_kappas.get(sid, 0.0)
        print(f" Set{sid:<2} | {r['n_traces']:<10} | {r['R_len']:<12.4f} | {r['kappa_est']:<12.4f} | {tk:<12.4f} | {dk:<14.4f}")
    print("=" * 80)

    # 8. Render Multi-Set Inversion curves into a single high-quality grid figure
    val_plot_path = os.path.join(output_dir, "real_hekmatnejad_inversion_decoupled.png")
    
    # Try setting Malgun Gothic font for Korean support on Windows
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    n_active_sets = len(set_results)
    fig, axes = plt.subplots(n_active_sets, 4, figsize=(32, 5.5 * n_active_sets))
    fig.patch.set_facecolor("#fafafa")

    c_raw = "#d95f02"       # terracotta (Raw observed)
    c_direct = "#1b9e77"    # premium teal (Direct MLE)
    c_unbiased = "#7570b3"  # indigo (Unbiased MLE)
    c_true = "#252525"      # black (Ground Truth)

    for idx, (sid, r) in enumerate(set_results.items()):
        # Handle 1D or 2D axes array structure depending on number of sets
        if n_active_sets == 1:
            ax_pdf, ax_cdf, ax_rad, ax_rad_cdf = axes[0], axes[1], axes[2], axes[3]
        else:
            ax_pdf, ax_cdf, ax_rad, ax_rad_cdf = axes[idx, 0], axes[idx, 1], axes[idx, 2], axes[idx, 3]

        # PDF Subplot
        ax_pdf.set_facecolor("#ffffff")
        ax_pdf.hist(r['obs_lengths'], bins=20, density=True, alpha=0.15, color=c_raw, edgecolor=c_raw, 
                    label=f"관측 흔적 길이 [Observed]")
        
        l_plot = np.linspace(c_truncation, 15.0, 500)
        ax_pdf.plot(l_plot, r['pdf_direct'](l_plot), color=c_direct, linewidth=3.2, 
                    label=f"직접 MLE ({r['dist_name']}, 잘림보정) [Direct]")
        ax_pdf.plot(l_plot, r['pdf_unbiased'](l_plot), color=c_unbiased, linewidth=2.5, linestyle=":", 
                    label=f"무편향 MLE (크기 보정 포함) [Unbiased]")
        
        ax_pdf.hist(r['valid_true_lengths'], bins=20, density=True, histtype="step", color=c_true, linewidth=2.2, linestyle="--", 
                    label="실제 원래 참 분포 [True Unclipped]")
        
        ax_pdf.set_title(f"균열 세트 {sid} - 확률밀도함수 (PDF) 비교", fontsize=12, fontweight="bold", pad=12)
        ax_pdf.set_xlabel("흔적 길이 l (m)", fontsize=10)
        ax_pdf.set_ylabel("확률 밀도 (Probability Density)", fontsize=10)
        ax_pdf.set_xlim(0.0, 15.0)
        ax_pdf.grid(True, linestyle="--", alpha=0.4)
        ax_pdf.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=8.5)

        # CDF Subplot
        ax_cdf.set_facecolor("#ffffff")
        ax_cdf.step(np.sort(r['obs_lengths']), np.arange(1, len(r['obs_lengths'])+1)/len(r['obs_lengths']), color=c_raw, alpha=0.5, linewidth=2.0, where="post", 
                    label="관측 ECDF")
        ax_cdf.plot(l_plot, r['cdf_direct'](l_plot), color=c_direct, linewidth=3.2, 
                    label="직접 MLE CDF [Direct]")
        ax_cdf.plot(l_plot, r['cdf_unbiased'](l_plot), color=c_unbiased, linewidth=2.5, linestyle=":", 
                    label="무편향 MLE CDF [Unbiased]")
        ax_cdf.step(r['sorted_true'], r['ecdf_true'], color=c_true, linewidth=2.2, linestyle="--", 
                    label="실제 참 ECDF")

        ax_cdf.set_title(f"균열 세트 {sid} - 누적분포함수 (CDF) 비교", fontsize=12, fontweight="bold", pad=12)
        ax_cdf.set_xlabel("흔적 길이 l (m)", fontsize=10)
        ax_cdf.set_ylabel("누적 확률 (Cumulative Probability)", fontsize=10)
        ax_cdf.set_xlim(0.0, 15.0)
        ax_cdf.grid(True, linestyle="--", alpha=0.4)
        ax_cdf.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=8.5)

        # Add comprehensive stats box inside CDF
        stats_text = (
            f"■ 세트 {sid} 지반통계 지표\n"
            f"  - 모델: {r['dist_name']}\n"
            f"  - 자가 보정 d1/d2: {r['d1']:.2f}/{r['d2']:.2f} m\n"
            f"  - CDF RMSE (Unbiased): {r['rmse_unbiased']:.4f}\n"
            f"  - 추정 P21: {r['p21']:.4f} m/m2\n"
            f"  - 추정 P32: {r['p32_est']:.4f} m2/m3\n"
            f"  - 실제 P32: {r['p32_true']:.4f} m2/m3\n"
            f"  - P32 오차율: {r['error_pct']:.2f} %"
        )
        ax_cdf.text(
            0.48, 0.04, stats_text, transform=ax_cdf.transAxes, fontsize=9.5,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#fefefe", edgecolor="#cccccc", alpha=0.9),
            verticalalignment="bottom"
        )

        # 3rd Column: 3D True Radius Distribution vs Inverted 3D Radius PDF
        ax_rad.set_facecolor("#ffffff")
        # Filter local 3D radii inside 50m Crop Box
        mask_set_3d = (gt_set_id == sid)
        mask_crop_3d = (np.abs(gt_centers[:, 0]) <= crop_limit) & \
                        (np.abs(gt_centers[:, 1]) <= crop_limit) & \
                        (np.abs(gt_centers[:, 2]) <= crop_limit)
        valid_3d_radii = gt_radii[mask_set_3d & mask_crop_3d]

        if len(valid_3d_radii) > 0:
            ax_rad.hist(valid_3d_radii, bins=25, density=True, histtype="step", color=c_true, linewidth=2.2, linestyle="--",
                        label="실제 3D 참 반경 [True 3D Radius]")

        # Theoretical reconstructed 3D Pareto PDF
        rmin_3d = 1.0
        alpha_R = r['params'][0] + 1.0  # Size-bias corrected exponent
        r_plot = np.linspace(rmin_3d, 15.0, 500)
        pdf_r = alpha_R * (rmin_3d**alpha_R) / (r_plot**(alpha_R + 1))

        ax_rad.plot(r_plot, pdf_r, color=c_unbiased, linewidth=3.2,
                    label=f"역산된 3D 반경 PDF (Pareto, α_R={alpha_R:.2f}) [Inverted]")

        ax_rad.set_title(f"균열 세트 {sid} - 3D 참 반경 분포 역산", fontsize=12, fontweight="bold", pad=12)
        ax_rad.set_xlabel("균열 반경 R (m)", fontsize=10)
        ax_rad.set_ylabel("확률 밀도 (Probability Density)", fontsize=10)
        ax_rad.set_xlim(0.0, 15.0)
        ax_rad.grid(True, linestyle="--", alpha=0.4)
        ax_rad.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=8.5)

        # 4th Column: 3D True Radius CDF vs Inverted 3D Radius CDF
        ax_rad_cdf.set_facecolor("#ffffff")
        if len(valid_3d_radii) > 0:
            ax_rad_cdf.step(np.sort(valid_3d_radii), np.arange(1, len(valid_3d_radii)+1)/len(valid_3d_radii), 
                            color=c_true, linewidth=2.2, linestyle="--", label="실제 3D 참 반경 CDF [True 3D]")
        
        cdf_r = 1.0 - (rmin_3d / r_plot)**alpha_R
        ax_rad_cdf.plot(r_plot, cdf_r, color=c_unbiased, linewidth=3.2,
                        label=f"역산된 3D 반경 CDF (Pareto, α_R={alpha_R:.2f})")
        
        ax_rad_cdf.set_title(f"균열 세트 {sid} - 3D 참 반경 누적분포 (CDF) 역산", fontsize=12, fontweight="bold", pad=12)
        ax_rad_cdf.set_xlabel("균열 반경 R (m)", fontsize=10)
        ax_rad_cdf.set_ylabel("누적 확률 (Cumulative Probability)", fontsize=10)
        ax_rad_cdf.set_xlim(0.0, 15.0)
        ax_rad_cdf.grid(True, linestyle="--", alpha=0.4)
        ax_rad_cdf.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=8.5)

    plt.tight_layout()
    # Save PNG copy
    plt.savefig(val_plot_path, dpi=300, bbox_inches="tight")
    # Save PDF copy
    val_plot_pdf_path = val_plot_path.replace(".png", ".pdf")
    plt.savefig(val_plot_pdf_path, bbox_inches="tight")
    print(f"\n[*] Premium unified multi-set inversion figure saved to: {val_plot_path} and {val_plot_pdf_path}")
    plt.close()
    
    # Save copies to the validation path too to ensure validation files are updated
    val_only_path = os.path.join(output_dir, "real_hekmatnejad_inversion_validation.png")
    val_only_pdf_path = os.path.join(output_dir, "real_hekmatnejad_inversion_validation.pdf")
    import shutil
    shutil.copyfile(val_plot_path, val_only_path)
    shutil.copyfile(val_plot_pdf_path, val_only_pdf_path)
    print(f"[*] Copy saved to: {val_only_path} and {val_only_pdf_path}")


    # Generate individual premium figures for each set
    for sid, r in set_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor("#fafafa")
        
        ax1, ax2 = axes[0], axes[1]
        
        # PDF
        ax1.set_facecolor("#ffffff")
        ax1.hist(r['obs_lengths'], bins=20, density=True, alpha=0.15, color=c_raw, edgecolor=c_raw, label="관측 흔적 길이")
        ax1.plot(l_plot, r['pdf_direct'](l_plot), color=c_direct, linewidth=3.5, label=f"직접 MLE PDF ({r['dist_name']})")
        ax1.plot(l_plot, r['pdf_unbiased'](l_plot), color=c_unbiased, linewidth=2.5, linestyle=":", label="무편향 MLE PDF")
        ax1.hist(r['valid_true_lengths'], bins=20, density=True, histtype="step", color=c_true, linewidth=2.5, linestyle="--", label="실제 원래 참 분포")
        ax1.set_title(f"균열 세트 {sid} - 확률밀도함수 (PDF) 분석", fontsize=13, fontweight="bold", pad=15)
        ax1.set_xlabel("흔적 길이 l (m)", fontsize=11)
        ax1.set_ylabel("확률 밀도", fontsize=11)
        ax1.set_xlim(0.0, 15.0)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

        # CDF
        ax2.set_facecolor("#ffffff")
        ax2.step(np.sort(r['obs_lengths']), np.arange(1, len(r['obs_lengths'])+1)/len(r['obs_lengths']), color=c_raw, alpha=0.5, linewidth=2.0, where="post", label="관측 ECDF")
        ax2.plot(l_plot, r['cdf_direct'](l_plot), color=c_direct, linewidth=3.5, label="직접 MLE CDF")
        ax2.plot(l_plot, r['cdf_unbiased'](l_plot), color=c_unbiased, linewidth=2.5, linestyle=":", label="무편향 MLE CDF")
        ax2.step(r['sorted_true'], r['ecdf_true'], color=c_true, linewidth=2.5, linestyle="--", label="실제 참 ECDF")
        ax2.set_title(f"균열 세트 {sid} - 누적분포함수 (CDF) 분석", fontsize=13, fontweight="bold", pad=15)
        ax2.set_xlabel("흔적 길이 l (m)", fontsize=11)
        ax2.set_ylabel("누적 확률", fontsize=11)
        ax2.set_xlim(0.0, 15.0)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

        stats_text = (
            f"■ 세트 {sid} 지반통계 지표\n"
            f"  - 모델: {r['dist_name']}\n"
            f"  - 자가 보정 d1/d2: {r['d1']:.2f}/{r['d2']:.2f} m\n"
            f"  - CDF RMSE (Unbiased): {r['rmse_unbiased']:.4f}\n"
            f"  - 추정 P21: {r['p21']:.4f} m/m2\n"
            f"  - 추정 P32: {r['p32_est']:.4f} m2/m3\n"
            f"  - 실제 P32: {r['p32_true']:.4f} m2/m3\n"
            f"  - P32 오차율: {r['error_pct']:.2f} %"
        )
        ax2.text(
            0.48, 0.04, stats_text, transform=ax2.transAxes, fontsize=10,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#fefefe", edgecolor="#cccccc", alpha=0.9),
            verticalalignment="bottom"
        )

        set_plot_path = os.path.join(output_dir, f"real_hekmatnejad_inversion_set_{sid}.png")
        plt.tight_layout()
        plt.savefig(set_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Set {sid} individual premium figure saved to: {set_plot_path}")
    
    print("\n" + "=" * 80)
    print(" REAL DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
