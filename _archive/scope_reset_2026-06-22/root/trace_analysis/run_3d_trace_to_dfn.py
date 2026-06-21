"""
Bayesian 3D DFN Inverse Reconstruction Pipeline using 3D Slab Traces.
Projects 3D traces from slabs onto slab-center planes and executes the Bayesian reconstruction.
"""
import os
import sys
import argparse
import time
import numpy as np
import h5py
import subprocess
from typing import List, Dict, Tuple

# Set local imports
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_reconstruction_unified import (
    ExcavationFace, FaceTrace, ReconstructedPlane, StochasticFracture,
    classify_censoring, cluster_axial_traces_doubled_gmm,
    cluster_reconstructed_normals_3d, match_faces_hungarian, apply_absence_penalization,
    fit_constrained_map_plane, sample_single_face_posterior_candidates,
    compute_residual_statistics_and_priors, run_manifold_glide_sa, evaluate_dfn_loss,
    export_dfn_to_hdf5
)
from slab_reconstruction.slab_types import Slab
from slab_reconstruction.slab_utils import extract_slab_segments_from_truth
from rough_face.generator import RoughFace
from rough_face.intersection import extract_rough_traces

def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a 3D plane to a set of 3D points using Singular Value Decomposition.
    """
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    _, _, vh = np.linalg.svd(shifted)
    normal = vh[-1]  # Unit normal
    if normal[0] < 0:
        normal = -normal
    dists = np.abs(np.dot(shifted, normal))
    residual = float(np.sqrt(np.mean(dists**2)))
    return normal, centroid, residual

def main():
    default_input = os.path.join(_parent, "storage", "data", "dfn_export_for_python.h5")
    default_tunnel = os.path.join(_parent, "storage", "data", "단면_폴리곤.dat")
    
    parser = argparse.ArgumentParser(description="3D Trace-based Bayesian DFN Inverse Reconstruction Pipeline")
    parser.add_argument("--input", default=default_input, help="Path to ground truth HDF5 DFN file")
    parser.add_argument("--tunnel-dat", default=default_tunnel, help="Path to tunnel polygon .dat file")
    parser.add_argument("--x-start", type=float, default=-25.0, help="Tunnel analysis start coordinate (m)")
    parser.add_argument("--x-end", type=float, default=25.0, help="Tunnel analysis end coordinate (m)")
    parser.add_argument("--spacing", type=float, default=3.0, help="Slab spacing / face advance step (m)")
    parser.add_argument("--thickness", type=float, default=0.2, help="Slab thickness (m) for trace collection")
    parser.add_argument("--subslices", type=int, default=10, help="Slab 당 보조 슬라이스 수")
    parser.add_argument("--sa-iterations", type=int, default=150, help="Number of Manifold Glide SA iterations")
    parser.add_argument("--output-dir", default="storage/output/reconstruction_results_3d_traces", help="Output directory")
    parser.add_argument("--run-block-detector", action="store_true", help="Automatically run downstream 3D GPU block detector")
    parser.add_argument("--domain-buffer", type=float, default=3.0, help="Domain box buffer along X-axis (m)")
    parser.add_argument("--rough", action="store_true", help="Use rough face simulation instead of flat planes")
    parser.add_argument("--dx", type=float, default=0.3, help="Rough face maximum amplitude (m)")
    parser.add_argument("--lc", type=float, default=1.0, help="Rough face correlation length (m)")
    parser.add_argument("--res", type=float, default=0.1, help="Rough face resolution (m)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    print_section("1. Loading Data & Environment Config")
    
    # 1. Load tunnel boundary polygon
    print(f"[*] 터널 단면 .dat 파싱 중: {args.tunnel_dat}")
    poly_y, poly_z = load_tunnel_polygon_from_dat(args.tunnel_dat)
    poly_yz = np.column_stack([poly_y, poly_z])
    print(f"  -> 로드된 터널 바운더리 폴리곤: {len(poly_yz)} 개 노드")
    
    # 2. Load ground-truth DFN
    print(f"[*] Ground-Truth DFN HDF5 로드 중: {args.input}")
    with h5py.File(args.input, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        gt_radii = f['/fractures/radii'][:].ravel()
        gt_set_id = (f['/fractures/set_id'][:].ravel() if '/fractures/set_id' in f 
                     else np.ones(len(gt_radii), dtype=np.uint16))
        
        gt_centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        gt_normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        
    print(f"  -> GT DFN 균열 개수: {len(gt_radii):,} 개")
    
    # Define slabs and faces along X-axis
    x_positions = np.arange(args.x_start, args.x_end + 1e-5, args.spacing)
    slabs = []
    faces = []
    for i, x_pos in enumerate(x_positions):
        slabs.append(Slab(i, x_pos, x_pos - args.thickness/2, x_pos + args.thickness/2, args.thickness))
        faces.append(ExcavationFace(
            face_id=i + 1,
            x_face=float(x_pos),
            tunnel_polygon_yz=poly_yz,
            advance_step=args.spacing if i > 0 else 0.0
        ))
    print(f"  -> 생성된 Slab & 막장 단면 개수: {len(faces)} 개 (x = {list(x_positions)} m)")
    
    print_section("2. 3D Slab Trace Extraction & 2D Projection")
    
    obs_traces = []
    trace_id_global = 1
    original_3d_points = {}  # trace_id_global -> (p0_3d, p1_3d)
    
    if args.rough:
        print(f"[*] 3D Rough Face 생성 및 교차 절리선 추출 중... (amplitude={args.dx}m, lc={args.lc}m, res={args.res}m)")
        fracture_data = {
            'centers': gt_centers,
            'normals': gt_normals,
            'radii': gt_radii
        }
        y_min, y_max = np.min(poly_yz[:, 0]), np.max(poly_yz[:, 0])
        z_min, z_max = np.min(poly_yz[:, 1]), np.max(poly_yz[:, 1])
        pad = 1.0
        y_range = (y_min - pad, y_max + pad)
        z_range = (z_min - pad, z_max + pad)
        
        for i, face in enumerate(faces):
            # Rough Face 생성
            rough_face = RoughFace(
                base_x=face.x_face,
                y_range=y_range,
                z_range=z_range,
                resolution=args.res,
                amplitude=args.dx,
                correlation_length=args.lc,
                seed=42 + i
            )
            
            # 교차선 추출
            rough_traces = extract_rough_traces(fracture_data, rough_face, poly_yz)
            for rt in rough_traces:
                # 3D 끝 두 점 추출
                poly = rt['points']
                p0 = poly[0]
                p1 = poly[-1]
                
                original_3d_points[trace_id_global] = (p0, p1)
                
                # FaceTrace 생성 (3D 좌표 정보 보존을 위해 x_face는 중점의 x를 사용)
                x_avg = float((p0[0] + p1[0]) / 2.0)
                ft = FaceTrace(
                    face_id=face.face_id,
                    trace_id=trace_id_global,
                    x_face=x_avg,
                    p0_y=float(p0[1]),
                    p0_z=float(p0[2]),
                    p1_y=float(p1[1]),
                    p1_z=float(p1[2]),
                    parent_fracture_id=rt['fracture_id']
                )
                obs_traces.append(ft)
                trace_id_global += 1
    else:
        print("[*] 3D Flat Slab 내부 균열 선분 추출 및 2D 투영 진행 중...")
        for slab, face in zip(slabs, faces):
            slab_traces = extract_slab_segments_from_truth(
                gt_centers, gt_normals, gt_radii, slab, poly_yz, sub_slice_count=args.subslices
            )
            for st in slab_traces:
                original_3d_points[trace_id_global] = (st.p0, st.p1)
                
                ft = FaceTrace(
                    face_id=face.face_id,
                    trace_id=trace_id_global,
                    x_face=face.x_face,
                    p0_y=float(st.p0[1]),
                    p0_z=float(st.p0[2]),
                    p1_y=float(st.p1[1]),
                    p1_z=float(st.p1[2]),
                    parent_fracture_id=st.parent_id
                )
                obs_traces.append(ft)
                trace_id_global += 1
            
    print(f"  -> 투영된 2D 트레이스 추출 성공: 총 {len(obs_traces)} 개 세그먼트 검측됨.")
    
    # Classify censoring for each face
    for face in faces:
        classify_censoring(obs_traces, face, tolerance=0.10)
        
    type_counts = {0: 0, 1: 0, 2: 0}
    for t in obs_traces:
        type_counts[t.censoring_class] += 1
    print(f"  -> Censoring Class 분류 완료:")
    print(f"     - Type 0 (Contained): {type_counts[0]} 개")
    print(f"     - Type 1 (One-end Clipped): {type_counts[1]} 개")
    print(f"     - Type 2 (Both-end Clipped): {type_counts[2]} 개")
    
    # 2D Axial GMM Clustering
    print("\n[*] 2D Axial Doubled-Angle GMM 클러스터링 실행 중...")
    optimal_k = cluster_axial_traces_doubled_gmm(obs_traces, max_k=4)
    print(f"  -> BIC 분석 결과, 최적 세트 개수 선정 완료: K = {optimal_k} 세트")
    for k in range(1, optimal_k + 1):
        cnt = sum(1 for t in obs_traces if t.set_id == k)
        print(f"     - Set {k}: {cnt} 개 트레이스")
        
    print_section("3. Bayes Factor Face Association")
    
    # Match traces across consecutive faces
    matched_pairs = []
    grouped_traces = {}
    for t in obs_traces:
        grouped_traces.setdefault(t.face_id, []).append(t)
        
    for f_idx in range(len(faces) - 1):
        f0 = faces[f_idx]
        f1 = faces[f_idx + 1]
        
        traces_f0 = grouped_traces.get(f0.face_id, [])
        traces_f1 = grouped_traces.get(f1.face_id, [])
        
        print(f"[*] Face {f0.face_id} (x={f0.x_face}m) ↔ Face {f1.face_id} (x={f1.x_face}m) 헝가리안 매칭 중...")
        matches = match_faces_hungarian(traces_f0, traces_f1, set_stats=None)
        
        if f_idx < len(faces) - 2:
            f2 = faces[f_idx + 2]
            traces_f2 = grouped_traces.get(f2.face_id, [])
            matches = apply_absence_penalization(
                matches, traces_f0, traces_f1, f2, traces_f2, set_stats=None
            )
            
        matched_pairs.extend(matches)
        
    accepted_matches = [m for m in matched_pairs if m.accepted]
    print(f"  -> 글로벌 매칭 검증 완료: 총 {len(accepted_matches)} 개 트레이스 트랙 확보.")
    for m in accepted_matches[:10]:
         print(f"     - [Track] Face {m.face_id_prev}(T{m.trace_id_prev}) ↔ Face {m.face_id_curr}(T{m.trace_id_curr}) (ln BF = {m.log_bayes_factor:.2f})")
    if len(accepted_matches) > 10:
         print(f"     - ...외 {len(accepted_matches) - 10}개 트랙 생략")
         
    print_section("4. Censoring-Aware Constrained MAP Fitting")
    
    # Reconstruct 3D deterministic planes from tracks
    det_planes = []
    track_id = 1
    
    matched_trace_ids = set()
    for m in accepted_matches:
        matched_trace_ids.add(m.trace_id_prev)
        matched_trace_ids.add(m.trace_id_curr)
        
    from collections import defaultdict
    adj_graph = defaultdict(list)
    for m in accepted_matches:
        adj_graph[m.trace_id_prev].append(m.trace_id_curr)
        adj_graph[m.trace_id_curr].append(m.trace_id_prev)
        
    visited_nodes = set()
    tracks = []
    
    for m in accepted_matches:
        for tid in [m.trace_id_prev, m.trace_id_curr]:
            if tid not in visited_nodes:
                comp = []
                queue = [tid]
                visited_nodes.add(tid)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj_graph[curr]:
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            queue.append(neighbor)
                
                comp_traces = [next(t for t in obs_traces if t.trace_id == cid) for cid in comp]
                comp_traces.sort(key=lambda x: x.face_id)
                tracks.append(comp_traces)
                
    default_mu = float(np.log(2.0))
    default_sigma = 0.35
    for track_traces in tracks:
        set_id = track_traces[0].set_id
        
        # Collect all original 3D endpoints for traces in this track
        track_pts_list = []
        for t in track_traces:
            p0_3d, p1_3d = original_3d_points[t.trace_id]
            track_pts_list.append(p0_3d)
            track_pts_list.append(p1_3d)
            
        track_pts = np.array(track_pts_list)
        
        # SVD plane fitting using exact 3D coordinates
        normal, centroid, residual = fit_plane_svd(track_pts)
        
        # Estimate radius from 3D points extent
        diffs = track_pts[:, None, :] - track_pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        extent = np.max(dists)
        radius = max(0.5, float(extent / 2.0))
        
        dp = ReconstructedPlane(
            plane_id=track_id,
            point_x=float(centroid[0]),
            point_y=float(centroid[1]),
            point_z=float(centroid[2]),
            normal_x=float(normal[0]),
            normal_y=float(normal[1]),
            normal_z=float(normal[2]),
            radius=radius,
            source_trace_ids=[t.trace_id for t in track_traces],
            set_id=set_id
        )
        det_planes.append(dp)
        track_id += 1
        
    print(f"  -> Deterministic Multi-face Planes 역산 완료: 총 {len(det_planes)} 개 트랙 (병합 전 매칭 수: {len(accepted_matches)})")
    
    if len(det_planes) > 0:
        print("[*] 3D Normal 벡터 구면 가우스 혼합 분포 재추정 (Two-stage clustering)...")
        optimal_k_3d, set_stats = cluster_reconstructed_normals_3d(det_planes, max_k=optimal_k)
        print(f"  -> 3D Set Statistics 정밀 산정 완료 (K={optimal_k_3d}):")
        for sid, (mean_normal, kappa) in set_stats.items():
            print(f"     - Set {sid}: Mean Normal = {mean_normal}, VMF kappa = {kappa:.2f}")
    else:
        optimal_k_3d = optimal_k
        set_stats = {k: (np.array([1.0, 0.0, 0.0]), 10.0) for k in range(1, optimal_k + 1)}
        print("[!] 경고: 매칭된 트랙이 존재하지 않습니다. 임의의 3D Set 정보를 구성합니다.")
        
    # Generate Probabilistic Candidates for Single-Face Traces
    print("\n[*] Single-face traces 불확실성 전개 및 후보군 다각화 (Candidate Posterior Sampling)...")
    single_face_candidates = []
    for t in obs_traces:
        if t.trace_id not in matched_trace_ids:
            set_id = t.set_id or 1
            mean_n, kappa = set_stats.get(set_id, (np.array([1.0, 0.0, 0.0]), 10.0))
            
            candidates = sample_single_face_posterior_candidates(
                track_id, t, mean_n, kappa, mu_s=default_mu, sigma_s=default_sigma, set_id=set_id, n_samples=5
            )
            single_face_candidates.extend(candidates)
            track_id += 1
            
    print(f"  -> Probabilistic Single-face 후보군 생성 완료: {len(single_face_candidates)} 개 생성됨.")
    
    print_section("5. Residual DFN Analysis & Moment Matching")
    
    residual_priors = compute_residual_statistics_and_priors(
        obs_traces, det_planes, faces, set_stats
    )
    for sid, priors in residual_priors.items():
        r_avg = np.exp(priors['mu_s'] + 0.5 * priors['sigma_s']**2)
        print(f"     - Set {sid}: 잔류 P32={priors['P32']:.4f}, P30={priors['P30']:.6f}")
        print(f"                 Moment-Matched Size: mu_s={priors['mu_s']:.3f}, sigma_s={priors['sigma_s']:.3f} (Avg Radius={r_avg:.2f}m)")
        
    print_section("6. Manifold Glide SA Optimization")
    
    domain = {
        'xmin': args.x_start - args.domain_buffer,
        'xmax': args.x_end + args.domain_buffer,
        'ymin': -10.0,
        'ymax': 10.0,
        'zmin': -10.0,
        'zmax': 10.0
    }
    domain_box = np.array([domain['xmin'], domain['xmax'], domain['ymin'], domain['ymax'], domain['zmin'], domain['zmax']])
    
    print(f"[*] Manifold Glide SA 탐색 알고리즘 개시 (Iter={args.sa_iterations})...")
    sa_fixed_planes = det_planes + [cp for cp in single_face_candidates if cp.plane_id % 1000 == 0]
    optimized_priors, stochastic_dfn, final_sim_traces = run_manifold_glide_sa(
        obs_traces, sa_fixed_planes, faces, set_stats, residual_priors, domain, sa_iterations=args.sa_iterations
    )
    
    final_loss_dict = evaluate_dfn_loss(obs_traces, final_sim_traces)
    
    print("[*] 측면 분석용 2D 트레이스 정밀 맵핑 결과 시각화 중...")
    from plot_trace_comparison import plot_side_by_side_trace_comparison, plot_overlay_trace_comparison
    comparison_plot_path = os.path.join(args.output_dir, "trace_side_by_side_comparison.png")
    plot_side_by_side_trace_comparison(obs_traces, final_sim_traces, faces, comparison_plot_path)
    
    overlay_plot_path = os.path.join(args.output_dir, "trace_overlay_comparison.png")
    plot_overlay_trace_comparison(obs_traces, final_sim_traces, faces, overlay_plot_path)
    
    print_section("7. Exporting 3D DFN to HDF5")
    out_hdf5_file = os.path.join(args.output_dir, "reconstructed_dfn.h5")
    export_dfn_to_hdf5(
        out_hdf5_file, det_planes, single_face_candidates, stochastic_dfn, poly_yz, domain_box,
        x_start=args.x_start, x_end=args.x_end
    )
    
    block_results_path = os.path.join(args.output_dir, "block_results")
    n_detected_blocks = 0
    max_block_vol = 0.0
    median_block_vol = 0.0
    block_detection_time = 0.0
    
    if args.run_block_detector:
        print_section("8. Invoking Downstream 3D Block Detector")
        detector_script = os.path.join(_parent, "dfn_analysis", "run_dfn_pipeline.py")
        
        cmd = [
            sys.executable,
            detector_script,
            "--input", out_hdf5_file,
            "--voxel_size", "0.5",
            "--tol_factor", "0.6",
            "--min_voxels", "8",
            "--outdir", block_results_path,
            "--no_gpu"
        ]
        
        print(f"[*] Command: {' '.join(cmd)}")
        block_detector_start = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        block_detection_time = time.time() - block_detector_start
        
        print(res.stdout)
        if res.stderr:
             print("\n[!] Warnings/Errors during block detection:")
             print(res.stderr)
             
        summary_json = os.path.join(block_results_path, "block_summary.json")
        if os.path.exists(summary_json):
            import json
            with open(summary_json, 'r', encoding='utf-8') as f_sum:
                block_sum = json.load(f_sum)
                n_detected_blocks = block_sum.get("n_blocks", 0)
                blocks_info = block_sum.get("blocks", [])
                if blocks_info:
                    vols = [b.get("volume_m3", 0.0) for b in blocks_info]
                    max_block_vol = max(vols)
                    median_block_vol = float(np.median(vols))
                    
    total_elapsed = time.time() - start_time
    
    print_section("FINAL INVERSE RECONSTRUCTION REPORT")
    
    print("\n[1] TRACE RECONSTRUCTION METRICS")
    print("-" * 50)
    print(f"  * Observed Trace Count       : {final_loss_dict['obs_count']} 개")
    n_total_obs = len(obs_traces)
    n_matched = len(matched_trace_ids)
    n_single = n_total_obs - n_matched
    pct_matched = (n_matched / n_total_obs * 100) if n_total_obs > 0 else 0.0
    pct_single = (n_single / n_total_obs * 100) if n_total_obs > 0 else 0.0
    print(f"    - Matched Traces (Multi-face): {n_matched} 개 ({pct_matched:.2f} %)")
    print(f"    - Unmatched Traces (Single-face): {n_single} 개 ({pct_single:.2f} %)")
    print(f"  * Reconstructed Trace Count  : {final_loss_dict['sim_count']} 개")
    print(f"  * Trace Count Relative Error : {final_loss_dict['count_error'] * 100:.2f} %")
    print(f"  * P21 Total Intensity Error  : {final_loss_dict['p21_error'] * 100:.2f} %")
    print(f"  * Mean Length Observed       : {final_loss_dict['mean_L_obs']:.3f} m")
    print(f"  * Mean Length Reconstructed  : {final_loss_dict['mean_L_sim']:.3f} m")
    print(f"  * Mean Length Error          : {final_loss_dict['length_error'] * 100:.2f} %")
    print(f"  * Censoring Class Error      : {final_loss_dict['censoring_error'] * 100:.2f} %")
    print(f"  * Optimal 3D Sets Found      : {optimal_k_3d} Sets")
    
    print("\n[2] BLOCK FORMATION METRICS")
    print("-" * 50)
    if args.run_block_detector:
        print(f"  * Reconstructed Block Count  : {n_detected_blocks} 개")
        print(f"  * Maximum Block Volume       : {max_block_vol:.3f} m³")
        print(f"  * Median Block Volume        : {median_block_vol:.3f} m³")
        print(f"  * Block Detection Time       : {block_detection_time:.2f} 초")
    else:
        print("  * Block detector was not executed. Enable --run-block-detector to check block potential.")
        
    print(f"\n  * Total Elapsed Pipeline Time: {total_elapsed:.2f} 초")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
