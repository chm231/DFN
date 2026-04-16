import os
import sys
import argparse
import time
import json
import numpy as np
import h5py
import pandas as pd

# Core package import (dfn_analysis)
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(os.path.dirname(_here))
_dfn_path = os.path.join(_parent, "dfn_analysis")
if _dfn_path not in sys.path:
    sys.path.insert(0, _dfn_path)

from run_dfn_pipeline import load_hdf5
from .slab_types import Slab, LocalCandidate, EvaluationResult
from .slab_utils import extract_slab_points_from_truth
from .clustering import cluster_slab_points, get_major_truth_id
from .plane_fitting import fit_plane_svd, get_points_extent
from .candidate_linking import link_adjacent_slabs
from .global_reconstruction import merge_links_into_chains, reconstruct_global_planes
from .evaluator import evaluate_reconstruction_performance
from .visualizer import plot_reconstruction_3d_pyvista, plot_evaluation_metrics
from tqdm import tqdm

def main():
    # 기본 경로를 프로젝트 루트(_parent) 기준으로 절대 경로화
    default_input = os.path.join(_parent, "storage", "data", "dfn_export_for_python.h5")
    default_outdir = os.path.join(_parent, "storage", "output", "slab_reconstruction")

    parser = argparse.ArgumentParser(description="Slab 기반 균열 평면 복원 파이프라인 (Direction B)")
    parser.add_argument('--input', default=default_input, help="HDF5 DFN 파일")
    parser.add_argument('--spacing', type=float, default=3.0, help="Slab 간격 (m)")
    parser.add_argument('--thickness', type=float, default=0.2, help="Slab 두께 (m)")
    parser.add_argument('--subslices', type=int, default=10, help="Slab 당 보조 슬라이스 수")
    parser.add_argument('--outdir', default=default_outdir, help="결과 저장 폴더")
    
    # Clustering & Matching Thresholds
    parser.add_argument('--eps', type=float, default=1.0, help="DBSCAN eps (m)")
    parser.add_argument('--min_points', type=int, default=5, help="최소 클러스터 포인트 수")
    parser.add_argument('--max_link_score', type=float, default=3.0, help="최대 매칭 점수 임계값")
    parser.add_argument('--export_cad', action='store_true', help="Slab 및 Trace를 AutoCAD SCR로 내보냄")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    
    print("="*60)
    print(" [Slab-based Reconstruction Pipeline] Start Verification")
    print("="*60)

    # 1. 데이터 로드 (Ground Truth)
    h5_abs = os.path.abspath(os.path.join(_here, args.input))
    data = load_hdf5(h5_abs)
    
    centers = data['centers'].astype(np.float32)
    normals = data['normals'].astype(np.float32)
    radii = data['radii'].astype(np.float32)
    poly_yz = data.get('poly_YZ', None)
    crop_box = data['crop_box']
    
    start_x, end_x = float(crop_box[0]), float(crop_box[1])
    
    # 2. Slab 생성
    print(f"\n[1/6] Generating Slabs along X-axis: {start_x:.1f} to {end_x:.1f}")
    x_centers = np.arange(start_x + args.spacing/2, end_x, args.spacing)
    slabs = []
    for i, xc in enumerate(x_centers):
        slabs.append(Slab(i, xc, xc - args.thickness/2, xc + args.thickness/2, args.thickness))
    print(f" -> Created {len(slabs)} slabs.")

    # 3. Slab 데이터 수집 및 로컬 후보 추출
    print(f"\n[2/6] Extracting clusters and local planes within each Slab...")
    all_candidates = {}
    all_slab_segments = [] # CAD 내보내기용
    
    for slab in tqdm(slabs, desc="Processing Slabs"):
        pts, ids, segs = extract_slab_points_from_truth(centers, normals, radii, slab, poly_yz, sub_slice_count=args.subslices)
        all_slab_segments.append(segs)
        
        if len(pts) == 0:
            all_candidates[slab.index] = []
            continue
            
        # 클러스터링 (인덱스 반환)
        clusters_idx_list = cluster_slab_points(pts, eps=args.eps, min_samples=args.min_points)
        
        slab_cands = []
        for j, c_idx in enumerate(clusters_idx_list):
            cluster_pts = pts[c_idx]
            cluster_truth_ids = ids[c_idx]
            
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
    print(f" -> Extracted {n_total_cands} local candidates across all slabs.")

    # CAD 내보내기
    if args.export_cad:
        from .cad_exporter import export_slabs_and_traces_to_cad
        scr_path = os.path.join(args.outdir, 'slab_data_export.scr')
        export_slabs_and_traces_to_cad(slabs, all_slab_segments, poly_yz, scr_path)

    # 4. Slab 간 링킹 (Hungarian Matching)
    print(f"\n[3/6] Linking adjacent slabs using Hungarian Method...")
    all_links = []
    for i in range(len(slabs) - 1):
        cands_A = all_candidates[slabs[i].index]
        cands_B = all_candidates[slabs[i+1].index]
        
        links = link_adjacent_slabs(cands_A, cands_B, max_score_threshold=args.max_link_score)
        all_links.extend(links)
        
    print(f" -> Established {len(all_links)} links.")

    # 5. 글로벌 평면 복원
    print(f"\n[4/6] Merging links into chains and fitting global planes...")
    chains = merge_links_into_chains(all_links)
    reconstructed = reconstruct_global_planes(all_candidates, chains)
    print(f" -> Reconstructed {len(reconstructed)} global planes.")

    # 6. 평가 및 결과 저장
    print(f"\n[5/6] Evaluating reconstruction against Ground Truth...")
    eval_res = evaluate_reconstruction_performance(reconstructed, centers, normals)
    
    print("\n" + "="*40)
    print(f" [Final Result Summary]")
    print(f" - Truth Planes: {eval_res.total_truth:,}")
    print(f" - Reconstructed: {eval_res.total_reconstructed:,}")
    print(f" - Matched Succes: {eval_res.matched_count:,}")
    print(f" - Success Rate: {eval_res.success_rate:.1f} %")
    print(f" - Avg Angle Error: {eval_res.avg_angle_error:.2f} deg")
    print(f" - Avg Dist Error: {eval_res.avg_dist_error:.2f} m")
    print("="*40)

    # Export 결과를 JSON 및 CSV로 저장
    metrics_path = os.path.join(args.outdir, 'reconstruction_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_res.__dict__, f, indent=4)
        
    # 7. 시각화
    print(f"\n[6/6] Generating visualization plots...")
    viz_plot_path = os.path.join(args.outdir, 'comparison_summary.png')
    plot_evaluation_metrics(eval_res, viz_plot_path)
    
    # 3D PyVista (Interative)
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
    print(f" -> Output saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
