"""
[Direction B: Inverse Reconstruction]
B 방향 역산 파이프라인을 테스트하는 데모 실행 스크립트.
미리 만들어진 샘플 CSV나 임의의 Dummy Trace를 생성하여 파이프라인을 돌립니다.
"""
import os
import sys
import argparse
import numpy as np

# A 방향 패키지 임포트 (load_hdf5 활용)
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from detect_blocks_gpu import load_hdf5

from .trace_types import FaceTrace, ExcavationFace
from .face_trace_io import save_face_traces
from .excavation_face_traces import extract_excavation_face_traces_from_truth
from .reconstruction_pipeline import run_inverse_pipeline


def main():
    print("="*60)
    print(" [INFO] Running Inverse Reconstruction Pipeline (Direction B) ")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='B 방향 진단 (역산) 파이프라인 데모')
    parser.add_argument('--input', required=True, help='A 방향 HDF5 파일 경로')
    parser.add_argument('--dx', type=float, default=3.0, help='굴착 막장면 간격 (m)')
    parser.add_argument('--start_x', type=float, default=None, help='시작 x 좌표')
    parser.add_argument('--end_x', type=float, default=None, help='종료 x 좌표')
    parser.add_argument('--outdir', default='inverse_results', help='결과 저장 폴더')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. HDF5 데이터 로드 (Ground Truth DFN)
    print(f"\n[Info] Loading GT DFN data from: {args.input}")
    dfn_data = load_hdf5(args.input)
    
    domain_box = dfn_data['domain_box']
    start_x = args.start_x if args.start_x is not None else float(domain_box[0])
    end_x = args.end_x if args.end_x is not None else float(domain_box[1])
    
    tunnel_poly_yz = dfn_data.get('poly_YZ', None)
    if tunnel_poly_yz is None:
        print("[WARN] 터널 형상(poly_YZ)이 없습니다. 전체 단면으로 테스트합니다.")
        
    centers = dfn_data['centers']
    normals = dfn_data['normals']
    radii = dfn_data['radii']
    
    # 2. 막장면(Excavation Faces) 진행에 따른 Trace 추출
    print(f"\n[Info] 생성 구간: X={start_x:.1f}m ~ {end_x:.1f}m (dx={args.dx:.1f}m)")
    
    x_positions = np.arange(start_x, end_x + 1e-5, args.dx)
    all_traces = []
    
    for i, x_pos in enumerate(x_positions):
        face_id = i + 1
        face = ExcavationFace(face_id=face_id, x_face=x_pos, tunnel_polygon_yz=tunnel_poly_yz, advance_step=args.dx)
        
        # 교차선 수학 연산 + Shapely 폴리곤 클리핑
        curr_traces = extract_excavation_face_traces_from_truth(centers, normals, radii, face)
        all_traces.extend(curr_traces)
        
    print(f" -> 성공: {len(x_positions)}개의 막장면으로부터 터널 규격에 맞는 Trace {len(all_traces)}개 추출.")
    
    # 3. CSV 저장
    traces_csv = os.path.join(args.outdir, "synthetic_face_traces.csv")
    save_face_traces(all_traces, traces_csv)
    print(f" -> [Export] {traces_csv}")
    
    # 4. 역산 파이프라인 가동
    planes, blocks = run_inverse_pipeline(traces_csv, tunnel_poly_yz)
    
    print("\n[Result] 복원된 3D 평면(Reconstructed Planes) 결과 요약:")
    if not planes:
        print("  - 복원된 평면이 없습니다.")
    else:
        for p in planes[:5]: # 너무 많으면 상위 5개만 출력
            print(f"  - Plane {p.plane_id:03d} | Center(X,Y,Z): ({p.point_x:.1f},{p.point_y:.1f},{p.point_z:.1f}) | Normal: ({p.normal_x:.2f},{p.normal_y:.2f},{p.normal_z:.2f}) | Source IDs: {p.source_trace_ids}")
        if len(planes) > 5:
            print(f"  ... (외 {len(planes) - 5}개)")
        
    print(f"\n[Result] 추후 연동될 기하학 블록(Block Candiates) 개수: {len(blocks)}")
    print("="*60 + "\n")
    

if __name__ == "__main__":
    main()
