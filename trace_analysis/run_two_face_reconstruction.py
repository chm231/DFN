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

    # 4. 도메인 필터링 (전방 5m, 주변 10m)
    # 도메인 정의: X=[x_curr, x_curr + forward], Y/Z = [tunnel_min-halo, tunnel_max+halo]
    y_min, y_max = np.min(poly_y), np.max(poly_y)
    z_min, z_max = np.min(poly_z), np.max(poly_z)
    
    d_xmin, d_xmax = args.x_curr, args.x_curr + args.forward_dist
    d_ymin, d_ymax = y_min - args.halo, y_max + args.halo
    d_zmin, d_zmax = z_min - args.halo, z_max + args.halo
    
    print(f"\n[3/3] Filtering planes to forward domain...")
    print(f"  - Filter Box: X[{d_xmin}, {d_xmax}], Y[{d_ymin:.1f}, {d_ymax:.1f}], Z[{d_zmin:.1f}, {d_zmax:.1f}]")
    
    final_planes = []
    for p in reconstructed_planes:
        # 평면 원판이 타겟 도메인 박스와 교차하는지 거칠게 필터링
        # (원판의 중심이 도메인 박스 근처에 있는지 확인)
        cx, cy, cz = p.point_x, p.point_y, p.point_z
        r = p.radius
        
        # AABB 교차 검사
        if (cx + r >= d_xmin and cx - r <= d_xmax and
            cy + r >= d_ymin and cy - r <= d_ymax and
            cz + r >= d_zmin and cz - r <= d_zmax):
            final_planes.append(p)
            
    print(f"  - Final planes in target domain: {len(final_planes)}")

    # 5. 시각화 (선택 사항)
    if args.visualize and final_planes:
        print("\n[Viz] Launching 3D Viewer...")
        plotter = pv.Plotter()
        
        # 5.1 터널 형상 구현 (Extrusion)
        # 시작 단면 생성 (X = d_xmin)
        pts = np.column_stack([np.full(len(poly_yz), d_xmin), poly_yz])
        # PyVista 폴리곤 데이터 생성 (닫힌 루프 가정)
        faces = np.array([len(poly_yz)] + list(range(len(poly_yz))))
        tunnel_cap = pv.PolyData(pts, faces=faces)
        # 전방 거리만큼 X축 방향으로 압출
        tunnel_3d = tunnel_cap.extrude((args.forward_dist, 0, 0), capping=True)
        plotter.add_mesh(tunnel_3d, color='lightblue', opacity=0.15, show_edges=True, label="Tunnel Domain")
        
        # 5.2 좌표계 격자(X, Y, Z Axes Grid) 표시
        plotter.show_grid(color='gray', font_size=10, location='outer')

        # 5.3 도메인 박스 표시 (외곽 가이드)
        domain_box = pv.Box(bounds=(d_xmin, d_xmax, d_ymin, d_ymax, d_zmin, d_zmax))
        plotter.add_mesh(domain_box, color='white', opacity=0.1, style='wireframe')
        
        # 5.3 복원된 평면들 (더 투명하게 설정)
        for p in final_planes:
            disc = pv.Disc(center=(p.point_x, p.point_y, p.point_z), 
                           normal=(p.normal_x, p.normal_y, p.normal_z), 
                           outer=p.radius, inner=0, c_res=40)
            plotter.add_mesh(disc, color='orange', opacity=0.3)  # 투명도 0.3으로 하향
            
        plotter.add_legend()
        plotter.add_axes()
        plotter.show()
    
    print("\n[Done] Reconstruction process completed.")

if __name__ == '__main__':
    main()
