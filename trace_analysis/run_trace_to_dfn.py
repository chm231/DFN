"""
Rigorous Bayesian 3D DFN Inverse Reconstruction Pipeline — CLI Orchestrator.
Integrates all advanced mathematical modules into a single, high-fidelity pipeline.
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
sys.path.insert(0, _here)
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


def extract_observed_traces_from_truth(
    centers: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    set_ids: np.ndarray,
    faces: List[ExcavationFace]
) -> List[FaceTrace]:
    """
    Extracts observed 2D traces on tunnel excavation faces from a 3D ground-truth DFN.
    Utilizes our high-fidelity analytical disc-to-face intersection model.
    """
    from trace_reconstruction_unified import intersect_disc_with_face
    
    obs_traces = []
    tid = 1
    
    for face in faces:
        for i in range(len(radii)):
            ft = intersect_disc_with_face(
                centers[i, 0], centers[i, 1], centers[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                radii[i], face, start_trace_id=tid, set_id=int(set_ids[i]),
                parent_fracture_id=i
            )
            obs_traces.extend(ft)
            tid += len(ft)
            
    return obs_traces


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Bayesian 3D DFN Inverse Reconstruction Pipeline")
    parser.add_argument("--input", required=True, help="Path to ground truth HDF5 DFN file")
    parser.add_argument("--tunnel-dat", required=True, help="Path to tunnel polygon .dat file")
    parser.add_argument("--x-start", type=float, default=0.0, help="Tunnel analysis start coordinate (m)")
    parser.add_argument("--x-end", type=float, default=6.0, help="Tunnel analysis end coordinate (m)")
    parser.add_argument("--advance-step", type=float, default=3.0, help="Distance between tunnel faces (m)")
    parser.add_argument("--sa-iterations", type=int, default=150, help="Number of Manifold Glide SA iterations")
    parser.add_argument("--output-dir", default="storage/output/reconstruction_results", help="Output directory")
    parser.add_argument("--run-block-detector", action="store_true", help="Automatically run downstream 3D GPU block detector")
    parser.add_argument("--domain-buffer", type=float, default=3.0, help="Domain box buffer along X-axis (m) (increase to avoid boundary-touch block filtering)")
    
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
    
    # Define analysis faces along X-axis
    x_positions = np.arange(args.x_start, args.x_end + 1e-5, args.advance_step)
    faces = []
    for i, x_pos in enumerate(x_positions):
        faces.append(ExcavationFace(
            face_id=i + 1,
            x_face=float(x_pos),
            tunnel_polygon_yz=poly_yz,
            advance_step=args.advance_step if i > 0 else 0.0
        ))
    print(f"  -> 생성된 막장 단면 개수: {len(faces)} 개 (x = {list(x_positions)} m)")
    
    print_section("2. 2D Trace Extraction & Preprocessing")
    
    # Extract 2D traces on excavation faces from GT DFN
    print("[*] 3D 균열 원판 - 2D 막면 폴리곤 정밀 교차 검정 실행 중...")
    obs_traces = extract_observed_traces_from_truth(gt_centers, gt_normals, gt_radii, gt_set_id, faces)
    print(f"  -> 교차 트레이스 추출 성공: 총 {len(obs_traces)} 개 세그먼트 검측됨.")
    
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
        
    # Match between consecutive face IDs: e.g. Face 1 to Face 2, Face 2 to Face 3
    for f_idx in range(len(faces) - 1):
        f0 = faces[f_idx]
        f1 = faces[f_idx + 1]
        
        traces_f0 = grouped_traces.get(f0.face_id, [])
        traces_f1 = grouped_traces.get(f1.face_id, [])
        
        print(f"[*] Face {f0.face_id} (x={f0.x_face}m) ↔ Face {f1.face_id} (x={f1.x_face}m) 헝가리안 매칭 중...")
        # Resolve global matching via Hungarian algorithm based on Bayes Factor
        matches = match_faces_hungarian(traces_f0, traces_f1, set_stats=None)
        
        # Apply 3-face absence penalty if Face 2 (the third face) exists in sequence
        if f_idx < len(faces) - 2:
            f2 = faces[f_idx + 2]
            traces_f2 = grouped_traces.get(f2.face_id, [])
            matches = apply_absence_penalization(
                matches, traces_f0, traces_f1, f2, traces_f2, set_stats=None
            )
            
        matched_pairs.extend(matches)
        
    accepted_matches = [m for m in matched_pairs if m.accepted]
    print(f"  -> 글로벌 매칭 검증 완료: 총 {len(accepted_matches)} 개 트레이스 트랙 확보.")
    for m in accepted_matches:
         print(f"     - [Track] Face {m.face_id_prev}(T{m.trace_id_prev}) ↔ Face {m.face_id_curr}(T{m.trace_id_curr}) (ln BF = {m.log_bayes_factor:.2f})")
         
    print_section("4. Censoring-Aware Constrained MAP Fitting")
    
    # Reconstruct 3D deterministic planes from tracks
    det_planes = []
    track_id = 1
    
    # Single face traces (to be represented as candidate samples later)
    matched_trace_ids = set()
    for m in accepted_matches:
        matched_trace_ids.add(m.trace_id_prev)
        matched_trace_ids.add(m.trace_id_curr)
        
    # Merge pairwise matches into complete multi-face tracks using BFS/DFS connected components
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
                
                # Fetch original FaceTrace objects and sort them by Face ID to maintain sequential order
                comp_traces = [next(t for t in obs_traces if t.trace_id == cid) for cid in comp]
                comp_traces.sort(key=lambda x: x.face_id)
                tracks.append(comp_traces)
                
    # Fit 3D planes for each merged track (can contain 2 or 3 traces)
    default_mu = float(np.log(2.0))
    default_sigma = 0.35
    for track_traces in tracks:
        set_id = track_traces[0].set_id
        dp = fit_constrained_map_plane(
            track_id, track_traces, mu_s=default_mu, sigma_s=default_sigma, set_id=set_id
        )
        det_planes.append(dp)
        track_id += 1
        
    print(f"  -> Deterministic Multi-face Planes 역산 완료: 총 {len(det_planes)} 개 트랙 (병합 전 매칭 수: {len(accepted_matches)})")
    
    # Spherical GMM re-estimation (Two-stage clustering) to obtain exact 3D set statistics
    print("[*] 3D Normal 벡터 구면 가우스 혼합 분포 재추정 (Two-stage clustering)...")
    optimal_k_3d, set_stats = cluster_reconstructed_normals_3d(det_planes, max_k=optimal_k)
    print(f"  -> 3D Set Statistics 정밀 산정 완료 (K={optimal_k_3d}):")
    for sid, (mean_normal, kappa) in set_stats.items():
        print(f"     - Set {sid}: Mean Normal = {mean_normal}, VMF kappa = {kappa:.2f}")
        
    # Generate Probabilistic Candidates for Single-Face Traces
    print("\n[*] Single-face traces 불확실성 전개 및 후보군 다각화 (Candidate Posterior Sampling)...")
    single_face_candidates = []
    for t in obs_traces:
        if t.trace_id not in matched_trace_ids:
            set_id = t.set_id or 1
            mean_n, kappa = set_stats.get(set_id, (np.array([1.0, 0.0, 0.0]), 10.0))
            
            # Sample B candidates from posterior
            candidates = sample_single_face_posterior_candidates(
                track_id, t, mean_n, kappa, mu_s=default_mu, sigma_s=default_sigma, set_id=set_id, n_samples=5
            )
            single_face_candidates.extend(candidates)
            track_id += 1
            
    print(f"  -> Probabilistic Single-face 후보군 생성 완료: {len(single_face_candidates)} 개 생성됨.")
    
    print_section("5. Residual DFN Analysis & Moment Matching")
    
    # Resolve size priors using joint moment-matching of residual lengths
    print("[*] Deterministic 차감형 잔류 트레이스 강도-개수 적률 정밀 해석 실행 중...")
    residual_priors = compute_residual_statistics_and_priors(
        obs_traces, det_planes, faces, set_stats
    )
    for sid, priors in residual_priors.items():
        r_avg = np.exp(priors['mu_s'] + 0.5 * priors['sigma_s']**2)
        print(f"     - Set {sid}: 잔류 P32={priors['P32']:.4f}, P30={priors['P30']:.6f}")
        print(f"                 Moment-Matched Size: mu_s={priors['mu_s']:.3f}, sigma_s={priors['sigma_s']:.3f} (Avg Radius={r_avg:.2f}m)")
        
    print_section("6. Manifold Glide SA Optimization")
    
    # Domain definition
    domain = {
        'xmin': args.x_start - args.domain_buffer,
        'xmax': args.x_end + args.domain_buffer,
        'ymin': -10.0,
        'ymax': 10.0,
        'zmin': -10.0,
        'zmax': 10.0
    }
    domain_box = np.array([domain['xmin'], domain['xmax'], domain['ymin'], domain['ymax'], domain['zmin'], domain['zmax']])
    
    # Run coupled SA to match intensity constraint ridge
    print(f"[*] Manifold Glide SA 탐색 알고리즘 개시 (Iter={args.sa_iterations})...")
    # Select only the first candidate per single-face trace to avoid over-counting during SA trace matching
    sa_fixed_planes = det_planes + [cp for cp in single_face_candidates if cp.plane_id % 1000 == 0]
    optimized_priors, stochastic_dfn, final_sim_traces = run_manifold_glide_sa(
        obs_traces, sa_fixed_planes, faces, set_stats, residual_priors, domain, sa_iterations=args.sa_iterations
    )
    
    # Compare observed and simulated traces
    final_loss_dict = evaluate_dfn_loss(obs_traces, final_sim_traces)
    
    # Save premium side-by-side comparison figure
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
    
    # Run block detector if requested
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
            "--no_gpu" # Safe default fallback, can be toggled
        ]
        
        print(f"[*] Command: {' '.join(cmd)}")
        block_detector_start = time.time()
        # Execute the process synchronously
        res = subprocess.run(cmd, capture_output=True, text=True)
        block_detection_time = time.time() - block_detector_start
        
        print(res.stdout)
        if res.stderr:
             print("\n[!] Warnings/Errors during block detection:")
             print(res.stderr)
             
        # Read block detector outputs
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
    
    # Segment 1: Trace Reconstruction Metrics
    print("\n[1] TRACE RECONSTRUCTION METRICS")
    print("-" * 50)
    print(f"  * Observed Trace Count       : {final_loss_dict['obs_count']} 개")
    print(f"  * Reconstructed Trace Count  : {final_loss_dict['sim_count']} 개")
    print(f"  * Trace Count Relative Error : {final_loss_dict['count_error'] * 100:.2f} %")
    print(f"  * P21 Total Intensity Error  : {final_loss_dict['p21_error'] * 100:.2f} %")
    print(f"  * Mean Length Observed       : {final_loss_dict['mean_L_obs']:.3f} m")
    print(f"  * Mean Length Reconstructed  : {final_loss_dict['mean_L_sim']:.3f} m")
    print(f"  * Mean Length Error          : {final_loss_dict['length_error'] * 100:.2f} %")
    print(f"  * Censoring Class Error      : {final_loss_dict['censoring_error'] * 100:.2f} %")
    print(f"  * Optimal 3D Sets Found      : {optimal_k_3d} Sets")
    
    # Segment 2: Block Formation Metrics
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
