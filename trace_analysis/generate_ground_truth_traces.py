import os
import sys
import argparse
import numpy as np
import h5py
import pandas as pd

# Set paths for local imports
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_reconstruction_unified import ExcavationFace, FaceTrace, intersect_disc_with_face
from rough_face.generator import RoughFace
from rough_face.intersection import extract_rough_traces

def main():
    parser = argparse.ArgumentParser(description="Generate Labeled Ground Truth Traces with True Set IDs from 3D DFN")
    parser.add_argument("--input", required=True, help="Path to ground truth HDF5 DFN file")
    parser.add_argument("--tunnel-dat", required=True, help="Path to tunnel polygon .dat file")
    parser.add_argument("--x-start", type=float, default=0.0, help="Tunnel analysis start coordinate (m)")
    parser.add_argument("--x-end", type=float, default=6.0, help="Tunnel analysis end coordinate (m)")
    parser.add_argument("--advance-step", type=float, default=3.0, help="Distance between tunnel faces (m)")
    parser.add_argument("--output-prefix", default="storage/output/ground_truth_traces", help="Output file prefix (saves .h5 and .csv)")
    
    # Rough Face Options
    parser.add_argument("--flat", action="store_true", help="Use flat planes instead of rough face simulation")
    parser.add_argument("--dx", type=float, default=0.3, help="Rough face maximum amplitude (m)")
    parser.add_argument("--lc", type=float, default=1.0, help="Rough face correlation length (m)")
    parser.add_argument("--res", type=float, default=0.1, help="Rough face resolution (m)")
    
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
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
    
    # Extract traces
    obs_traces = []
    tid = 1
    
    if not args.flat:
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
        
        for idx, face in enumerate(faces):
            rough_face = RoughFace(
                base_x=face.x_face,
                y_range=y_range,
                z_range=z_range,
                resolution=args.res,
                amplitude=args.dx,
                correlation_length=args.lc,
                seed=42 + idx
            )
            rough_traces = extract_rough_traces(fracture_data, rough_face, poly_yz)
            for rt in rough_traces:
                poly = rt['points']
                p0 = poly[0]
                p1 = poly[-1]
                
                # FaceTrace 생성
                x_avg = float((p0[0] + p1[0]) / 2.0)
                ft = FaceTrace(
                    face_id=face.face_id,
                    trace_id=tid,
                    x_face=x_avg,
                    p0_y=float(p0[1]),
                    p0_z=float(p0[2]),
                    p1_y=float(p1[1]),
                    p1_z=float(p1[2]),
                    parent_fracture_id=int(rt['fracture_id'])
                )
                ft.set_id = int(gt_set_id[rt['fracture_id']])
                ft.p0_x = float(p0[0])
                ft.p1_x = float(p1[0])
                
                obs_traces.append(ft)
                tid += 1
    else:
        print("[*] 3D 균열 원판 - 2D 평면 막면 교차 검정 실행 및 정답지 Trace 생성 중...")
        for face in faces:
            for i in range(len(gt_radii)):
                ft = intersect_disc_with_face(
                    gt_centers[i, 0], gt_centers[i, 1], gt_centers[i, 2],
                    gt_normals[i, 0], gt_normals[i, 1], gt_normals[i, 2],
                    gt_radii[i], face, start_trace_id=tid, set_id=int(gt_set_id[i]),
                    parent_fracture_id=i
                )
                for t in ft:
                    t.p0_x = face.x_face
                    t.p1_x = face.x_face
                obs_traces.extend(ft)
                tid += len(ft)
            
    print(f"  -> 교차 트레이스 정답지 추출 완료: 총 {len(obs_traces)} 개 세그먼트 생성됨.")

    # 3. Export to HDF5
    h5_path = args.output_prefix + ".h5"
    print(f"[*] HDF5 파일 출력 중: {h5_path}")
    
    trace_ids = np.array([t.trace_id for t in obs_traces], dtype=np.int32)
    face_ids = np.array([t.face_id for t in obs_traces], dtype=np.int32)
    x_faces = np.array([t.x_face for t in obs_traces], dtype=np.float32)
    p0_x = np.array([getattr(t, 'p0_x', t.x_face) for t in obs_traces], dtype=np.float32)
    p0_y = np.array([t.p0_y for t in obs_traces], dtype=np.float32)
    p0_z = np.array([t.p0_z for t in obs_traces], dtype=np.float32)
    p1_x = np.array([getattr(t, 'p1_x', t.x_face) for t in obs_traces], dtype=np.float32)
    p1_y = np.array([t.p1_y for t in obs_traces], dtype=np.float32)
    p1_z = np.array([t.p1_z for t in obs_traces], dtype=np.float32)
    set_ids = np.array([t.set_id for t in obs_traces], dtype=np.uint16)
    parent_ids = np.array([t.parent_fracture_id if t.parent_fracture_id is not None else -1 for t in obs_traces], dtype=np.int32)
    
    with h5py.File(h5_path, 'w') as f:
        grp = f.create_group('traces')
        grp.create_dataset('trace_id', data=trace_ids, compression='gzip')
        grp.create_dataset('face_id', data=face_ids, compression='gzip')
        grp.create_dataset('x_face', data=x_faces, compression='gzip')
        grp.create_dataset('p0_x', data=p0_x, compression='gzip')
        grp.create_dataset('p0_y', data=p0_y, compression='gzip')
        grp.create_dataset('p0_z', data=p0_z, compression='gzip')
        grp.create_dataset('p1_x', data=p1_x, compression='gzip')
        grp.create_dataset('p1_y', data=p1_y, compression='gzip')
        grp.create_dataset('p1_z', data=p1_z, compression='gzip')
        grp.create_dataset('set_id', data=set_ids, compression='gzip')
        grp.create_dataset('parent_fracture_id', data=parent_ids, compression='gzip')
        
        grp_tunnel = f.create_group('tunnel')
        grp_tunnel.create_dataset('poly_YZ', data=poly_yz.T, compression='gzip')
        
    print(f"  -> HDF5 정답지 파일 생성 성공.")
    
    # 4. Export to CSV
    csv_path = args.output_prefix + ".csv"
    print(f"[*] CSV 파일 출력 중: {csv_path}")
    
    df = pd.DataFrame({
        'trace_id': trace_ids,
        'face_id': face_ids,
        'x_face': x_faces,
        'p0_x': p0_x,
        'p0_y': p0_y,
        'p0_z': p0_z,
        'p1_x': p1_x,
        'p1_y': p1_y,
        'p1_z': p1_z,
        'set_id': set_ids,
        'parent_fracture_id': parent_ids
    })
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV 정답지 파일 생성 성공: {csv_path}")
    print("\n[Done] All ground truth trace datasets have been successfully created!")

if __name__ == "__main__":
    main()
