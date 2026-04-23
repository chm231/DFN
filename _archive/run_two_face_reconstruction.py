import argparse
import h5py
import numpy as np
import pyvista as pv
from trace_reconstruction.trace_types import ExcavationFace, FaceTrace
from trace_reconstruction.excavation_face_traces import extract_excavation_face_traces_from_truth
from trace_reconstruction.trace_matching import build_trace_tracks
from trace_reconstruction.plane_reconstruction import fit_plane_from_trace_track

def load_tunnel_polygon_from_dat(filepath: str):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]

def main():
    parser = argparse.ArgumentParser(description="[Direction B] Inverse Fracture Reconstruction")
    parser.add_argument('--input', default='storage/data/dfn_export_for_python.h5', help="원본 H5 데이터 (Truth)")
    parser.add_argument('--tunnel_dat', default='storage/data/단면_폴리곤.dat', help="터널 단면 polygon 데이터")
    parser.add_argument('--x_curr', type=float, default=20.0, help="현재 발굴 막장 위치 (X)")
    parser.add_argument('--num_faces', type=int, default=2, help="분석에 사용할 막장면의 개수 (기본 2개)")
    parser.add_argument('--interval', type=float, default=3.0, help="막장 간격 (dx, m)")
    parser.add_argument('--forward_dist', type=float, default=5.0, help="현재 막장 뒤 암반 예측 깊이 (m)")
    parser.add_argument('--halo', type=float, default=10.0, help="터널 주변 도메인 반경 (m)")
    parser.add_argument('--voxel_size', type=float, default=0.5, help="분석용 복셀 크기 (m)")
    parser.add_argument('--visualize', action='store_true', help="복원 결과 시각화 여부")
    args = parser.parse_args()

    print("="*60)
    print(f" [Inverse] Multi-Face Fracture Reconstruction")
    print(f" -> Current X: {args.x_curr}m")
    print(f" -> Use Faces: {args.num_faces} faces (dx: {args.interval}m)")
    print(f" -> Forward Predict: {args.forward_dist}m")
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

    # 2. 다중 막장면 Trace 추출 (Simulation from truth)
    face_pos = [args.x_curr - (i * args.interval) for i in range(args.num_faces)]
    face_pos.reverse()
    
    grouped_traces = {}
    
    print(f"\n[1/3] Extracting traces for {args.num_faces} faces...")
    for i, x in enumerate(face_pos):
        face = ExcavationFace(face_id=i+1, x_face=x, tunnel_polygon_yz=poly_yz, advance_step=args.interval)
        traces = extract_excavation_face_traces_from_truth(centers, normals, radii, face)
        grouped_traces[i+1] = traces
        print(f"  - Face {i+1} (X={x:.1f}): {len(traces)} traces found.")

    # 3. 매칭 및 3차원 복원
    print(f"\n[2/3] Matching traces and reconstructing 3D planes...")
    valid_tracks = build_trace_tracks(grouped_traces, min_faces=args.num_faces)
    
    reconstructed_planes = []
    for i, track in enumerate(valid_tracks):
        plane = fit_plane_from_trace_track(track, i+1)
        if plane:
            reconstructed_planes.append(plane)
    
    print(f"  - Reconstructed {len(reconstructed_planes)} planes from matching triples/pairs.")

    # 4. 도메인 필터링
    d_xmin = args.x_curr
    d_xmax = args.x_curr + args.forward_dist
    d_ymin, d_ymax = -15.5, 15.5
    d_zmin, d_zmax = -15.0, 15.0
    
    print(f"\n[3/3] Filtering and Clipping planes to expanded domain...")
    print(f"  - Domain Box: X[{d_xmin}, {d_xmax}], Y[{d_ymin}, {d_ymax}], Z[{d_zmin}, {d_zmax}]")
    
    domain_bounds = (d_xmin, d_xmax, d_ymin, d_ymax, d_zmin, d_zmax)
    domain_box_mesh = pv.Box(bounds=domain_bounds)

    # 5. 시각화
    if args.visualize:
        print("\n[Viz] Launching 3D Viewer (Expanded Domain Mode)...")
        plotter = pv.Plotter()
        
        # 5.1 터널 형상 구현 (분석에 사용된 전체 구간 시각화)
        d_xmin_eval = face_pos[0]
        total_eval_dist = args.x_curr - d_xmin_eval
        
        pts = np.column_stack([np.full(len(poly_yz), d_xmin_eval), poly_yz])
        faces = np.array([len(poly_yz)] + list(range(len(poly_yz))))
        tunnel_cap = pv.PolyData(pts, faces=faces)
        tunnel_3d = tunnel_cap.extrude((total_eval_dist, 0, 0), capping=True)
        plotter.add_mesh(tunnel_3d, color='lightblue', opacity=0.4, show_edges=True, label="Analyzed Tunnel Segment")
        
        plotter.show_grid(color='gray', font_size=10, location='outer')
        plotter.add_mesh(domain_box_mesh, color='black', opacity=1.0, style='wireframe', line_width=3, label="Prediction Zone")
        
        print("  - Visualizing traces with censoring color-coding...")
        for fid, traces in grouped_traces.items():
            for t in traces:
                color = 'green' # VISIBLE
                if t.censoring.name == 'ONE_END_CLIPPED': color = 'yellow'
                elif t.censoring.name == 'BOTH_END_CLIPPED': color = 'red'
                
                line = pv.Line((t.x_face, t.p0_y, t.p0_z), (t.x_face, t.p1_y, t.p1_z))
                plotter.add_mesh(line, color=color, line_width=1, label="Trace" if fid==1 and t.trace_id==1 else None)

        if reconstructed_planes:
            for i, p in enumerate(reconstructed_planes):
                disc = pv.Disc(center=(p.point_x, p.point_y, p.point_z), 
                               normal=(p.normal_x, p.normal_y, p.normal_z), 
                               outer=p.radius, inner=0, c_res=40)
                
                r = 1.0 - p.confidence
                g = p.confidence
                rgb_color = [r, g, 0.2]
                
                plotter.add_mesh(disc, color=rgb_color, opacity=0.3, 
                                 label=f"Plane (Conf:{p.confidence:.2f})" if i < 3 else None)
                
                edges = disc.extract_feature_edges(boundary_edges=True)
                plotter.add_mesh(edges, color='black', line_width=1)
            
        plotter.add_legend()
        plotter.add_axes()
        plotter.show()
        
if __name__ == "__main__":
    main()
