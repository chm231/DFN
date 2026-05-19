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
        
    # 7. Collect statistical parameters for Hekmatnejad et al. (2018) pipeline
    obs_lengths = np.array([t.length for t in obs_traces])
    censoring_types = np.array([t.censoring_class for t in obs_traces])
    
    # Exact dip/intersection angle calculation from corresponding 3D normal vector
    dips_deg = []
    for t in obs_traces:
        # parent fracture index
        parent_id = t.parent_fracture_id
        # normal vector of parent
        ny, nz = gt_normals[parent_id, 1], gt_normals[parent_id, 2]
        sin_theta = np.sqrt(ny**2 + nz**2)
        theta_rad = np.arcsin(np.clip(sin_theta, 1e-6, 1.0))
        dips_deg.append(np.rad2deg(theta_rad))
    dips_deg = np.array(dips_deg)
    
    # Extract unclipped true lengths for validation
    true_lengths = np.array([true_unclipped_lengths[t.trace_id] for t in obs_traces])
    
    # Truncation threshold limit
    c_truncation = 0.15
    
    # 1. Direct MLE Estimator (Censoring & Truncation correction, but no Size-bias shift)
    # This represents the size-biased true length of the intersected population
    estimator_direct = ParametricMLEEstimator(
        min_truncation=c_truncation,
        correct_size_bias=False,
        window_diameter=10.0,
        self_calibrate=True
    )
    print("\n[*] Executing Direct Parametric MLE Inversion (Unsupervised Blind Self-Calibration)...")
    res_direct = estimator_direct.fit(obs_lengths, censoring_types)
    cdf_fun_direct = res_direct["cdf_function"]
    pdf_fun_direct = res_direct["pdf_function"]
    
    # 2. Unbiased MLE Estimator (Censoring & Truncation & Size-bias shift corrected)
    # This represents the true unbiased trace length of the entire 3D rock mass
    estimator_unbiased = ParametricMLEEstimator(
        min_truncation=c_truncation,
        correct_size_bias=True,
        window_diameter=10.0,
        self_calibrate=True
    )
    print("\n[*] Executing Unbiased Parametric MLE Inversion (Unsupervised Blind Self-Calibration)...")
    res_unbiased = estimator_unbiased.fit(obs_lengths, censoring_types)
    cdf_fun_unbiased = res_unbiased["cdf_function"]
    pdf_fun_unbiased = res_unbiased["pdf_function"]
    
    # Inversion Quality Metrics
    valid_mask = true_lengths >= c_truncation
    valid_true_lengths = true_lengths[valid_mask]
    sorted_true = np.sort(valid_true_lengths)
    ecdf_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
    
    # Compute RMSE against validation intersected true unclipped lengths
    rmse_direct = np.sqrt(np.mean((cdf_fun_direct(sorted_true) - ecdf_true)**2))
    rmse_unbiased = np.sqrt(np.mean((cdf_fun_unbiased(sorted_true) - ecdf_true)**2))
    
    l_grid = np.linspace(c_truncation, np.max(obs_lengths), 1000)
    dx = l_grid[1] - l_grid[0]
    pdf_integral_direct = np.sum(pdf_fun_direct(l_grid)) * dx
    pdf_integral_unbiased = np.sum(pdf_fun_unbiased(l_grid)) * dx
    
    print(f"\n" + "-" * 50)
    print(" REAL DATA PARAMETRIC INVERSION ACCURACY REPORT")
    print("-" * 50)
    print(f"  * Total Unified Trace Sample Size   : {len(obs_lengths)} traces")
    print(f"  * Truncation Limit (c)              : {c_truncation} m")
    print(f"  * Best Model Selected               : {res_direct['dist_name']}")
    print(f"  * Direct MLE (Censoring-Only) RMSE  : {rmse_direct:.5f}")
    print(f"  * Unbiased MLE (Censoring+Size) RMSE: {rmse_unbiased:.5f}")
    print(f"  * Direct PDF Area Under Curve       : {pdf_integral_direct:.4f}")
    print(f"  * Unbiased PDF Area Under Curve     : {pdf_integral_unbiased:.4f}")
    print("-" * 50)
    
    # 8. Plot Overall Inversion Validation Curves
    val_plot_path = os.path.join(output_dir, "real_hekmatnejad_inversion_decoupled.png")
    
    # Try setting Malgun Gothic font for Korean support on Windows
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#fafafa")
    
    c_raw = "#d95f02"       # terracotta (Raw biased data)
    c_direct = "#1b9e77"    # premium teal (Direct MLE: Censoring-only)
    c_unbiased = "#7570b3"  # indigo (Unbiased MLE: Censoring + Size-bias corrected)
    c_true = "#252525"      # black (Ground Truth unclipped intersected)
    
    # PANEL 1: PDF Curves
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    ax1.hist(obs_lengths, bins=20, density=True, alpha=0.15, color=c_raw, edgecolor=c_raw, 
             label="관측된 데이터 (왜곡됨/잘림) [Observed]")
    
    l_plot = np.linspace(c_truncation, 15.0, 500)
    ax1.plot(l_plot, pdf_fun_direct(l_plot), color=c_direct, linewidth=3.5, 
             label=f"직접 MLE PDF ({res_direct['dist_name']}, 잘림만 보정) [Direct MLE]")
    ax1.plot(l_plot, pdf_fun_unbiased(l_plot), color=c_unbiased, linewidth=2.5, linestyle=":", 
             label=f"무편향 MLE PDF (크기 보정 포함) [Unbiased MLE]")
    
    # Density plot of true unclipped lengths
    ax1.hist(valid_true_lengths, bins=20, density=True, histtype="step", color=c_true, linewidth=2.5, linestyle="--", 
             label="실제 원래 분포 (교차군 참값) [True Unclipped]")
    
    ax1.set_title("확률 밀도 함수 (PDF) 모수적 MLE 비교 (15m 스케일)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("균열 흔적 길이 $l$ (m)", fontsize=11)
    ax1.set_ylabel("확률 밀도 (Probability Density)", fontsize=11)
    ax1.set_xlim(0.0, 15.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=9)
    
    # PANEL 2: CDF Curves
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    ax2.step(np.sort(obs_lengths), np.arange(1, len(obs_lengths)+1)/len(obs_lengths), color=c_raw, alpha=0.5, linewidth=2.0, where="post", 
             label="관측 ECDF (왜곡됨) [Observed ECDF]")
    
    ax2.plot(l_plot, cdf_fun_direct(l_plot), color=c_direct, linewidth=3.5, 
             label="직접 MLE CDF (관측군 잘림 보정) [Direct MLE]")
    ax2.plot(l_plot, cdf_fun_unbiased(l_plot), color=c_unbiased, linewidth=2.5, linestyle=":", 
             label="무편향 MLE CDF (3D 암반 실제 크기 분포) [Unbiased MLE]")
    
    # True unclipped empirical ECDF
    ax2.step(sorted_true, ecdf_true, color=c_true, linewidth=2.5, linestyle="--", 
             label="실제 참 ECDF (교차군 참값) [True ECDF]")
        
    ax2.set_title("누적 분포 함수 (CDF) 모수적 MLE 비교 (15m 스케일)", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xlabel("균열 흔적 길이 $l$ (m)", fontsize=11)
    ax2.set_ylabel("누적 확률 (Cumulative Probability)", fontsize=11)
    ax2.set_xlim(0.0, 15.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=9)
    
    # Embed high-diagnostic performance text box
    stats_text = (
        f"■ 무이중페널티 모수적 MLE 지표\n"
        f"  - 최적 선택 분포        : {res_direct['dist_name']}\n"
        f"  - 직접 CDF 오차 (RMSE)  : {rmse_direct:.5f}\n"
        f"  - 무편향 CDF 오차 (RMSE): {rmse_unbiased:.5f}\n"
        f"  - 관측 샘플 개수        : {len(obs_lengths)} ea"
    )
    ax2.text(
        0.50, 0.05, stats_text, transform=ax2.transAxes, fontsize=10,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fefefe", edgecolor="#cccccc", alpha=0.9),
        verticalalignment="bottom"
    )
    
    plt.tight_layout()
    plt.savefig(val_plot_path, dpi=300, bbox_inches="tight")
    print(f"[*] Premium real data inversion figure saved to: {val_plot_path}")
    plt.close()
    
    # Save a copy to the validation path too to make sure both files get updated
    val_only_path = os.path.join(output_dir, "real_hekmatnejad_inversion_validation.png")
    import shutil
    shutil.copyfile(val_plot_path, val_only_path)
    print(f"[*] Copy saved to: {val_only_path}")
    
    print("\n" + "=" * 80)
    print(" REAL DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
