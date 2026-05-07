import os
import sys
import time
import argparse
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """Numpy 배열 및 스칼라를 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from .generator import RoughFace
from .intersection import extract_rough_traces
from .visualizer_rough import plot_rough_reconstruction, plot_rough_comparison_2d

# 경로 설정
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
_dfn_analysis_path = os.path.join(_root, "dfn_analysis")

if _dfn_analysis_path not in sys.path:
    sys.path.insert(0, _dfn_analysis_path)

from run_dfn_pipeline import load_hdf5

def main():
    parser = argparse.ArgumentParser(description="Rough Face 기반 균열 관측 시뮬레이션")
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    
    default_input = os.path.join(_root, "storage", "data", "dfn_export_for_python.h5")
    default_outdir = os.path.join(_root, "storage", "output", "rough_face_simulation")
    
    parser.add_argument('--input', default=default_input, help="HDF5 DFN 파일")
    parser.add_argument('--outdir', default=default_outdir, help="결과 저장 폴더")
    parser.add_argument('--interval', type=float, default=3.0, help="굴착 간격 (m)")
    parser.add_argument('--dx', type=float, default=0.3, help="굴착면 최대 요철 진폭 (m)")
    parser.add_argument('--lc', type=float, default=1.0, help="상관 거리 (Correlation Length, m)")
    parser.add_argument('--res', type=float, default=0.1, help="그리드 해상도 (m)")
    parser.add_argument('--num_faces', type=int, default=0, help="시뮬레이션할 막장 수 (0이면 전체 구간)")
    parser.add_argument('--visualize', action='store_true', help="동작 확인용 시각화 여부")
    parser.add_argument('--export_cad', action='store_true', help="결과를 AutoCAD SCR로 내보냄")
    
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    print("="*60)
    print(" [Rough Face Simulation Pipeline] Start")
    print("="*60)
    
    # 1. 데이터 로드
    data = load_hdf5(args.input)
    fracture_data = {
        'centers': data['centers'].astype(np.float32),
        'normals': data['normals'].astype(np.float32),
        'radii': data['radii'].astype(np.float32)
    }
    poly_yz = data.get('poly_YZ', None)
    crop_box = data['crop_box']
    
    # 터널 Y-Z 범위 계산 (정사각형 도메인)
    y_min, y_max = np.min(poly_yz[:, 0]), np.max(poly_yz[:, 0])
    z_min, z_max = np.min(poly_yz[:, 1]), np.max(poly_yz[:, 1])
    # 여유 공간 추가
    pad = 1.0
    y_range = (y_min - pad, y_max + pad)
    z_range = (z_min - pad, z_max + pad)
    
    # 2. 막장별 시뮬레이션
    start_x = float(crop_box[0])
    end_x = float(crop_box[1])
    
    x_positions = np.arange(start_x, end_x + 0.1, args.interval)
    if args.num_faces > 0:
        x_positions = x_positions[:args.num_faces]
    
    n_total_faces = len(x_positions)
    faces_results = []
    
    all_rough_traces_data = [] # CAD용
    all_ideal_traces_data = [] # CAD용
    
    for i, curr_x in enumerate(x_positions):
        print(f"\n[Face {i+1}/{n_total_faces}] x = {curr_x:.1f} m")
        
        # A. Rough Face 생성
        rough_face = RoughFace(
            base_x=curr_x,
            y_range=y_range,
            z_range=z_range,
            resolution=args.res,
            amplitude=args.dx,
            correlation_length=args.lc,
            seed=42 + i
        )
        
        # B. Ideal Face 생성 (비교용)
        ideal_face = RoughFace(
            base_x=curr_x,
            y_range=y_range,
            z_range=z_range,
            resolution=args.res,
            amplitude=0.0
        )
        
        # C. Trace 추출
        rough_traces = extract_rough_traces(fracture_data, rough_face, poly_yz)
        ideal_traces = extract_rough_traces(fracture_data, ideal_face, poly_yz)
        
        all_rough_traces_data.append(rough_traces)
        all_ideal_traces_data.append(ideal_traces)
        
        print(f" -> Found {len(rough_traces)} traces (Rough) vs {len(ideal_traces)} (Ideal)")
        
        # D. 저장용 데이터 구성
        faces_results.append({
            'face_id': i,
            'center_x': curr_x,
            'rough_traces_count': len(rough_traces),
            'ideal_traces_count': len(ideal_traces)
        })
        
        # E. 시각화 (각 Face별)
        if args.visualize:
            viz_name_3d = os.path.join(args.outdir, f"face_{i}_3d_view.png")
            viz_name_2d = os.path.join(args.outdir, f"face_{i}_comparison_2d.png")
            plot_rough_comparison_2d(rough_traces, ideal_traces, viz_name_2d)
            plot_rough_reconstruction(rough_face, rough_traces, ideal_traces, viz_name_3d)

    # 3. 통합 결과 저장
    # A. JSON 요약 및 상세 데이터
    summary_path = os.path.join(args.outdir, "simulation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(faces_results, f, indent=4)
        
    detail_path = os.path.join(args.outdir, "rough_traces_data.json")
    with open(detail_path, 'w') as f:
        # 모든 face의 trace 데이터를 구조화하여 저장
        detailed_data = []
        for i, face_traces in enumerate(all_rough_traces_data):
            detailed_data.append({
                'face_id': i,
                'center_x': x_positions[i],
                'traces': face_traces
            })
        json.dump(detailed_data, f, indent=4, cls=NumpyEncoder)
        
    # B. CAD 익스포트
    if args.export_cad:
        from .cad_exporter import export_rough_traces_to_cad
        scr_path = os.path.join(args.outdir, "rough_traces_export.scr")
        export_rough_traces_to_cad(all_rough_traces_data, x_positions, poly_yz, scr_path)
        
    print("\n" + "="*60)
    print(f" [Simulation Done] Results saved to: {os.path.abspath(args.outdir)}")
    print("="*60)

if __name__ == "__main__":
    main()
