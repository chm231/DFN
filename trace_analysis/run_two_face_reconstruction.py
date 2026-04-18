"""
run_two_face_reconstruction.py
두 개의 막장면(Face) 데이터를 활용하여 3차원 균열을 복원하는 전용 파이프라인.
"""

import argparse
import os
import sys
import numpy as np
import h5py
import pyvista as pv

# 로컬 모듈 로드 설정
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
_dfn_path = os.path.join(_parent, "dfn_analysis")
sys.path.insert(0, _here)
sys.path.insert(0, _dfn_path)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_reconstruction.trace_types import ExcavationFace, FaceTrace
from trace_reconstruction.excavation_face_traces import extract_excavation_face_traces_from_truth
from trace_reconstruction.trace_matching import match_traces_between_faces, build_trace_tracks
from trace_reconstruction.plane_reconstruction import fit_plane_from_trace_track

def main():
    parser = argparse.ArgumentParser(description="Two-Face Inverse Fracture Reconstruction")
    parser.add_argument('--input', default='storage/data/dfn_export_for_python.h5', help="원본 H5 데이터 (Truth)")
    parser.add_argument('--tunnel_dat', default='storage/data/단면_폴리곤.dat', help="터널 단면 polygon 데이터")
    parser.add_argument('--x_curr', type=float, default=20.0, help="현재 발굴 막장 위치 (X)")
    parser.add_argument('--interval', type=float, default=3.0, help="연속된 막장 사이의 간격 (m)")
    parser.add_argument('--forward_dist', type=float, default=5.0, help="예측할 전방 도메인 길이 (m)")
    parser.add_argument('--halo', type=float, default=10.0, help="터널 주변 도메인 반경 (m)")
    parser.add_argument('--voxel_size', type=float, default=0.5, help="분석용 복셀 크기 (m)")
    parser.add_argument('--visualize', action='store_true', help="복원 결과 시각화 여부")
    args = parser.parse_args()

    print("="*60)
    print(f" [Inverse] 2-Face Based Fracture Reconstruction")
    print(f" -> Current X: {args.x_curr}m, Interval: {args.interval}m")
    print("="*60)

    # 1. 데이터 로드 (터널 및 DFN)
    poly_y, poly_z = load_tunnel_polygon_from_dat(args.tunnel_dat)
    poly_yz = np.column_stack([poly_y, poly_z])
    
    with h5py.File(args.input, 'r') as f:
        centers = f['/fractures/centers'][:]
        normals = f['/fractures/normals'][:]
        radii = f['/fractures/radii'][:].ravel()
        if centers.shape[0] == 3 and centers.shape[0] < centers.shape[1]: centers = centers.T
        if normals.shape[0] == 3 and normals.shape[0] < normals.shape[1]: normals = normals.T

    # 2. 2개 막장면 Trace 추출 (Simulation from truth)
    face_pos = [args.x_curr - args.interval, args.x_curr]
    grouped_traces = {}
    
    print(f"\n[1/3] Extracting traces at X = {face_pos[0]} and {face_pos[1]}...")
    for i, x in enumerate(face_pos):
        face = ExcavationFace(face_id=i+1, x_face=x, tunnel_polygon_yz=poly_yz, advance_step=args.interval)
        traces = extract_excavation_face_traces_from_truth(centers, normals, radii, face)
        grouped_traces[i+1] = traces
        print(f"  - Face {i+1} (X={x:.1f}): {len(traces)} traces found.")

    # 3. 매칭 및 3차원 복원
    print(f"\n[2/3] Matching traces and reconstructing 3D planes...")
    # min_faces=2로 설정하여 두 면만 있어도 트랙 생성 가능하게 함
    valid_tracks = build_trace_tracks(grouped_traces, min_faces=2)
    
    reconstructed_planes = []
    for i, track in enumerate(valid_tracks):
        plane = fit_plane_from_trace_track(track, i+1)
        if plane:
            reconstructed_planes.append(plane)
    
    print(f"  - Reconstructed {len(reconstructed_planes)} planes from matching pairs.")

    # 4. 도메인 필터링 (입력 막장 ~ 전방 관심 깊이 전체 구간으로 확장)
    # X: [x_curr - interval, x_curr + forward], Y: [-15.5, 15.5], Z: [-15.0, 15.0]
    d_xmin = args.x_curr - args.interval
    d_xmax = args.x_curr + args.forward_dist
    d_ymin, d_ymax = -15.5, 15.5
    d_zmin, d_zmax = -15.0, 15.0
    
    print(f"\n[3/3] Filtering and Clipping planes to expanded domain...")
    print(f"  - Domain Box: X[{d_xmin}, {d_xmax}], Y[{d_ymin}, {d_ymax}], Z[{d_zmin}, {d_zmax}]")
    
    # 도메인 박스 (클리핑용)
    domain_bounds = (d_xmin, d_xmax, d_ymin, d_ymax, d_zmin, d_zmax)
    domain_box_mesh = pv.Box(bounds=domain_bounds)

    # 5. 시각화
    if args.visualize and reconstructed_planes:
        print("\n[Viz] Launching 3D Viewer (Expanded Domain Mode)...")
        plotter = pv.Plotter()
        
        # 5.1 터널 형상 구현 (실제 굴착된 3m 구간만 생성)
        pts = np.column_stack([np.full(len(poly_yz), d_xmin), poly_yz])
        faces = np.array([len(poly_yz)] + list(range(len(poly_yz))))
        tunnel_cap = pv.PolyData(pts, faces=faces)
        # 압출 길이를 실제 굴착 간격(interval)으로 제한
        tunnel_3d = tunnel_cap.extrude((args.interval, 0, 0), capping=True)
        plotter.add_mesh(tunnel_3d, color='lightblue', opacity=0.4, show_edges=True, label="Excavated Tunnel (Known)")
        
        # 5.2 좌표계 격자
        plotter.show_grid(color='gray', font_size=10, location='outer')

        # 5.3 예측 도메인 가이드 박스 (전방 5m 포함 전체)
        plotter.add_mesh(domain_box_mesh, color='white', opacity=0.05, style='wireframe', label="Prediction Zone")
        
        # 5.4 복원된 평면들 (원판 Disc 형태로 시각화)
        for i, p in enumerate(reconstructed_planes):
            # 복원된 반경을 사용하는 원판 생성
            disc = pv.Disc(center=(p.point_x, p.point_y, p.point_z), 
                           normal=(p.normal_x, p.normal_y, p.normal_z), 
                           outer=p.radius, inner=0, c_res=40)
            
            # 원판 본체 (반투명 주황색)
            plotter.add_mesh(disc, color='orange', opacity=0.15, label="Reconstructed Fracture" if i==0 else None)
            
            # 검은색 윤곽선 (원판의 둘레 강조)
            edges = disc.extract_feature_edges(boundary_edges=True, feature_edges=False, non_manifold_edges=False, manifold_edges=False)
            plotter.add_mesh(edges, color='black', line_width=1)
            
        plotter.add_legend()
        plotter.add_axes()
        plotter.show()
    
    print("\n[Done] Reconstruction process completed.")

if __name__ == '__main__':
    main()
