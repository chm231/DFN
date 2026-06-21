"""
run_slab_pipeline.py
=====================
Slab 기반 균열 원판 복원 & DFN 파라미터 추출 통합 파이프라인.

9단계 파이프라인:
  1/9: Slab 생성
  2/9: 3D 선분 추출 & 클러스터링
  3/9: Slab 간 링킹 (Hungarian)
  4/9: 글로벌 평면 복원
  5/9: 반경 추정 (radius_estimator)
  6/9: 세트 자동 분류 (set_classifier)
  7/9: DFN 파라미터 추출 (dfn_parameter_extractor)
  8/9: Ground Truth 대비 평가
  9/9: 시각화 + 결과 저장
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import h5py

# Core package import (dfn_analysis)
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(os.path.dirname(_here))
_dfn_path = os.path.join(_parent, "dfn_analysis")
if _dfn_path not in sys.path:
    sys.path.insert(0, _dfn_path)

from run_dfn_pipeline import load_hdf5
from .slab_types import Slab, LocalCandidate, EvaluationResult
from .slab_utils import extract_slab_segments_from_truth
from .clustering import get_major_truth_id
from .slab_trace_bridge import cluster_3d_segments

from .plane_fitting import fit_plane_svd, get_points_extent
from .candidate_linking import link_adjacent_slabs
from .global_reconstruction import merge_links_into_chains, reconstruct_global_planes
from .evaluator import evaluate_reconstruction_performance, evaluate_per_set
from .visualizer import plot_reconstruction_3d_pyvista, plot_evaluation_metrics

# 신규 모듈
from .radius_estimator import estimate_radii_for_planes
from .set_classifier import classify_sets
from .dfn_parameter_extractor import (
    extract_dfn_parameters, 
    format_dfn_summary_table, 
    export_dfn_parameters_json
)

from tqdm import tqdm

def main():
    # 기본 경로를 프로젝트 루트(_parent) 기준으로 절대 경로화
    default_input = os.path.join(_parent, "storage", "data", "dfn_export_for_python.h5")
    default_outdir = os.path.join(_parent, "storage", "output", "slab_reconstruction")

    parser = argparse.ArgumentParser(description="Slab 기반 균열 원판 복원 & DFN 파라미터 추출 파이프라인")
    parser.add_argument('--input', default=default_input, help="HDF5 DFN 파일")
    parser.add_argument('--spacing', type=float, default=3.0, help="Slab 간격 (m)")
    parser.add_argument('--thickness', type=float, default=0.2, help="Slab 두께 (m)")
    parser.add_argument('--subslices', type=int, default=10, help="Slab 당 보조 슬라이스 수")
    parser.add_argument('--outdir', default=default_outdir, help="결과 저장 폴더")
    
    # Clustering & Matching Thresholds
    parser.add_argument('--eps', type=float, default=1.0, help="Clustering 거리 임계값 (m)")
    parser.add_argument('--min_points', type=int, default=5, help="최소 클러스터 포인트 수")
    parser.add_argument('--max_link_score', type=float, default=3.0, help="최대 매칭 점수 임계값")
    parser.add_argument('--max_sets', type=int, default=6, help="탐색할 최대 세트 수")
    parser.add_argument('--export_cad', action='store_true', help="Slab 및 Trace를 AutoCAD SCR로 내보냄")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    
    print("=" * 70)
    print(" [Slab-based Reconstruction & DFN Parameter Extraction Pipeline]")
    print("=" * 70)

    # =========================================================================
    # [1/9] 데이터 로드 & Slab 생성
    # =========================================================================
    h5_abs = os.path.abspath(os.path.join(_here, args.input))
    data = load_hdf5(h5_abs)
    
    centers = data['centers'].astype(np.float32)
    normals = data['normals'].astype(np.float32)
    radii = data['radii'].astype(np.float32)
    poly_yz = data.get('poly_YZ', None)
    crop_box = data['crop_box']
    
    # Ground Truth set_id 로드 시도
    gt_set_ids = None
    try:
        with h5py.File(h5_abs, 'r') as f:
            if '/fractures/set_id' in f:
                gt_set_ids = f['/fractures/set_id'][:].ravel().astype(int)
    except Exception:
        pass
    
    start_x, end_x = float(crop_box[0]), float(crop_box[1])
    
    print(f"\n[1/9] Generating Slabs along X-axis: {start_x:.1f} to {end_x:.1f}")
    print(f"  -> Loaded {len(radii):,} ground truth fractures.")
    
    x_centers = np.arange(start_x + args.spacing/2, end_x, args.spacing)
    slabs = []
    for i, xc in enumerate(x_centers):
        slabs.append(Slab(i, xc, xc - args.thickness/2, xc + args.thickness/2, args.thickness))
    print(f"  -> Created {len(slabs)} slabs (spacing={args.spacing}m, thickness={args.thickness}m).")

    # =========================================================================
    # [2/9] 3D 선분 추출 & 클러스터링
    # =========================================================================
    print(f"\n[2/9] Extracting 3D segments and clustering within each Slab...")
    all_candidates = {}
    all_slab_segments = []
    
    for slab in tqdm(slabs, desc="Processing Slabs"):
        # 3D 선분(Traces) 직접 추출
        slab_traces = extract_slab_segments_from_truth(
            centers, normals, radii, slab, poly_yz, sub_slice_count=args.subslices
        )
        
        # CAD 시각화용 선분 리스트
        segs = [np.array([tr.p0, tr.p1]) for tr in slab_traces]
        all_slab_segments.append(segs)
        
        if len(slab_traces) == 0:
            all_candidates[slab.index] = []
            continue
            
        # 3D 선분 기반 클러스터링 수행
        clusters_idx_list = cluster_3d_segments(
            slab_traces, 
            dist_threshold=args.eps,
            angle_penalty_threshold=0.8,
            min_samples=args.min_points
        )
        
        slab_cands = []
        for j, c_idx in enumerate(clusters_idx_list):
            # 그룹화된 선분들의 양 끝점들을 수집하여 피팅용 점군 형성
            cluster_pts_list = []
            cluster_truth_ids_list = []
            for t_idx in c_idx:
                tr = slab_traces[t_idx]
                cluster_pts_list.append(tr.p0)
                cluster_pts_list.append(tr.p1)
                cluster_truth_ids_list.append(tr.parent_id)
                
            cluster_pts = np.array(cluster_pts_list)
            cluster_truth_ids = np.array(cluster_truth_ids_list)
            
            # 피팅
            n, c, res = fit_plane_svd(cluster_pts)
            extent = get_points_extent(cluster_pts)
            major_id = get_major_truth_id(cluster_truth_ids)
            
            slab_cands.append(LocalCandidate(
                slab_index=slab.index,
                candidate_id=j,
                points=cluster_pts,
                normal=n,
                centroid=c,
                residual=res,
                extent=extent,
                truth_fracture_ids=cluster_truth_ids.tolist(),
                major_truth_id=major_id
            ))
            
        all_candidates[slab.index] = slab_cands
        
    n_total_cands = sum(len(v) for v in all_candidates.values())
    print(f"  -> Extracted {n_total_cands} local candidates across all slabs.")

    # CAD 내보내기
    if args.export_cad:
        from .cad_exporter import export_slabs_and_traces_to_cad
        scr_path = os.path.join(args.outdir, 'slab_data_export.scr')
        export_slabs_and_traces_to_cad(slabs, all_slab_segments, poly_yz, scr_path)

    # =========================================================================
    # [3/9] Slab 간 링킹 (Hungarian Matching)
    # =========================================================================
    print(f"\n[3/9] Linking adjacent slabs using Hungarian Method...")
    all_links = []
    for i in range(len(slabs) - 1):
        cands_A = all_candidates[slabs[i].index]
        cands_B = all_candidates[slabs[i+1].index]
        
        links = link_adjacent_slabs(cands_A, cands_B, max_score_threshold=args.max_link_score)
        all_links.extend(links)
        
    print(f"  -> Established {len(all_links)} links.")

    # =========================================================================
    # [4/9] 글로벌 평면 복원
    # =========================================================================
    print(f"\n[4/9] Merging links into chains and fitting global planes...")
    chains = merge_links_into_chains(all_links)
    reconstructed = reconstruct_global_planes(all_candidates, chains)
    print(f"  -> Reconstructed {len(reconstructed)} global planes.")

    # =========================================================================
    # [5/9] 반경 추정
    # =========================================================================
    print(f"\n[5/9] Estimating fracture disc radii...")
    reconstructed = estimate_radii_for_planes(reconstructed, slab_spacing=args.spacing)
    
    est_radii = [p.estimated_radius for p in reconstructed]
    if est_radii:
        print(f"  -> Radius Statistics:")
        print(f"     - Mean:   {np.mean(est_radii):.2f} m")
        print(f"     - Median: {np.median(est_radii):.2f} m")
        print(f"     - Min:    {np.min(est_radii):.2f} m")
        print(f"     - Max:    {np.max(est_radii):.2f} m")

    # =========================================================================
    # [6/9] 세트 자동 분류
    # =========================================================================
    print(f"\n[6/9] Classifying fracture sets via Spherical K-means...")
    optimal_k, set_stats = classify_sets(reconstructed, max_k=args.max_sets)
    
    print(f"  -> Optimal K = {optimal_k} sets identified.")
    for sid, (mean_n, kappa) in set_stats.items():
        n_in_set = sum(1 for p in reconstructed if p.set_id == sid)
        print(f"     - Set {sid}: {n_in_set} planes, kappa = {kappa:.2f}, "
              f"Mean Normal = [{mean_n[0]:.3f}, {mean_n[1]:.3f}, {mean_n[2]:.3f}]")

    # =========================================================================
    # [7/9] DFN 파라미터 추출
    # =========================================================================
    print(f"\n[7/9] Extracting DFN statistical parameters...")
    
    # 터널 단면적 계산
    tunnel_area = None
    if poly_yz is not None and len(poly_yz) >= 3:
        n_pts = len(poly_yz)
        area = 0.0
        for i in range(n_pts):
            j = (i + 1) % n_pts
            area += poly_yz[i, 0] * poly_yz[j, 1]
            area -= poly_yz[j, 0] * poly_yz[i, 1]
        tunnel_area = 0.5 * abs(area)
    
    dfn_result = extract_dfn_parameters(
        reconstructed,
        set_stats,
        slab_x_range=(start_x, end_x),
        tunnel_poly_yz=poly_yz,
        tunnel_area=tunnel_area
    )
    
    # 요약 테이블 출력
    summary_table = format_dfn_summary_table(
        dfn_result,
        gt_centers=centers,
        gt_normals=normals,
        gt_radii=radii,
        gt_set_ids=gt_set_ids
    )
    print(f"\n{summary_table}")
    
    # DFN 파라미터 JSON 내보내기
    dfn_json_path = os.path.join(args.outdir, 'dfn_parameters.json')
    export_dfn_parameters_json(dfn_result, dfn_json_path)
    print(f"\n  -> DFN parameters exported to: {dfn_json_path}")

    # =========================================================================
    # [8/9] Ground Truth 대비 평가
    # =========================================================================
    print(f"\n[8/9] Evaluating reconstruction against Ground Truth...")
    eval_res = evaluate_reconstruction_performance(
        reconstructed, centers, normals, radii
    )
    
    print("\n" + "=" * 50)
    print(f" [Final Reconstruction Result Summary]")
    print(f" - Truth Planes:     {eval_res.total_truth:,}")
    print(f" - Reconstructed:    {eval_res.total_reconstructed:,}")
    print(f" - Matched Success:  {eval_res.matched_count:,}")
    print(f" - Success Rate:     {eval_res.success_rate:.1f} %")
    print(f" - Avg Angle Error:  {eval_res.avg_angle_error:.2f} deg")
    print(f" - Avg Dist Error:   {eval_res.avg_dist_error:.2f} m")
    print(f" - Avg Radius Error: {eval_res.avg_radius_error:.2f} m")
    print("=" * 50)
    
    # 세트별 분리 평가
    if gt_set_ids is not None:
        per_set_eval = evaluate_per_set(
            reconstructed, centers, normals, radii, gt_set_ids
        )
        print("\n  [Per-Set Evaluation]")
        for sid, e_res in sorted(per_set_eval.items()):
            print(f"    Set {sid}: Matched={e_res.matched_count}, "
                  f"AngErr={e_res.avg_angle_error:.2f}°, "
                  f"DistErr={e_res.avg_dist_error:.2f}m, "
                  f"RadErr={e_res.avg_radius_error:.2f}m")

    # Export 결과를 JSON으로 저장
    metrics_path = os.path.join(args.outdir, 'reconstruction_metrics.json')
    metrics_dict = {
        'total_truth': eval_res.total_truth,
        'total_reconstructed': eval_res.total_reconstructed,
        'matched_count': eval_res.matched_count,
        'avg_angle_error': eval_res.avg_angle_error,
        'avg_dist_error': eval_res.avg_dist_error,
        'avg_radius_error': eval_res.avg_radius_error,
        'success_rate': eval_res.success_rate
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    # =========================================================================
    # [9/9] 시각화 + 결과 저장
    # =========================================================================
    print(f"\n[9/9] Generating visualization plots...")
    viz_plot_path = os.path.join(args.outdir, 'comparison_summary.png')
    plot_evaluation_metrics(eval_res, viz_plot_path)
    
    # 3D PyVista (Interactive)
    plot_reconstruction_3d_pyvista(
        reconstructed,
        truth_centers=centers,
        truth_normals=normals,
        truth_radii=radii,
        tunnel_poly_yz=poly_yz,
        x_range=(start_x, end_x),
        save_path=os.path.join(args.outdir, 'reconstruction_3d_view.png')
    )
    
    elapsed = time.time() - t0
    print(f"\n[Done] Pipeline completed in {elapsed:.1f}s.")
    print(f"  -> Output saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
