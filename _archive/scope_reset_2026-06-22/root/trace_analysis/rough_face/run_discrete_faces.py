import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Tuple

# 경로 설정
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
_root = os.path.dirname(_parent)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "dfn_analysis"))

from trace_analysis.rough_face.generator import RoughFace
from trace_analysis.rough_face.intersection import extract_rough_traces
from trace_analysis.rough_face.visualizer_rough import plot_rough_reconstruction, plot_rough_comparison_2d
from trace_analysis.load_tunnel_dat import load_tunnel_polygon_from_dat
from run_dfn_pipeline import load_hdf5

def main():
    print("=" * 80)
    print(" Rough Face Trace Simulation at Discrete Faces [0m, 3m, 6m, 9m]")
    print("=" * 80)
    
    # 1. 파일 경로 지정
    hdf5_path = os.path.join(_root, "storage", "data", "dfn_export_for_python.h5")
    dat_path = os.path.join(_root, "storage", "data", "단면_폴리곤.dat")
    outdir = os.path.join(_root, "storage", "output", "discrete_faces_simulation")
    os.makedirs(outdir, exist_ok=True)
    
    # 2. 마제형 터널 설계 단면 폴리곤 로드
    print(f"[*] Loading actual horseshoe tunnel polygon from: {dat_path}")
    if os.path.exists(dat_path):
        poly_y, poly_z = load_tunnel_polygon_from_dat(dat_path)
        poly_yz = np.column_stack([poly_y, poly_z])
        print(f"    -> Successfully parsed {len(poly_yz)} boundary nodes.")
    else:
        print("[Error] 설계 폴리곤 파일(.dat)이 없습니다.")
        return
        
    # 3. 3D DFN 로드
    print(f"[*] Loading 3D DFN database: {hdf5_path}")
    data = load_hdf5(hdf5_path)
    fracture_data = {
        'centers': data['centers'].astype(np.float32),
        'normals': data['normals'].astype(np.float32),
        'radii': data['radii'].astype(np.float32)
    }
    print(f"    -> Ground Truth DFN contains {len(fracture_data['radii']):,} fractures.")
    
    # 터널 Y-Z 도메인 범위 계산
    y_min, y_max = np.min(poly_yz[:, 0]), np.max(poly_yz[:, 0])
    z_min, z_max = np.min(poly_yz[:, 1]), np.max(poly_yz[:, 1])
    pad = 1.0
    y_range = (y_min - pad, y_max + pad)
    z_range = (z_min - pad, z_max + pad)
    
    # 4. 지정된 위치 [0m, 3m, 6m, 9m]에서의 막장면 시뮬레이션 및 이미지 저장
    x_positions = [0.0, 3.0, 6.0, 9.0]
    
    for idx, curr_x in enumerate(x_positions):
        print(f"\n[Face {idx+1}/4] Excavation Face at x = {curr_x:.1f} m")
        
        # A. 30cm 진폭 요철 막장면 생성
        rough_face = RoughFace(
            base_x=curr_x,
            y_range=y_range,
            z_range=z_range,
            resolution=0.1,
            amplitude=0.3,
            correlation_length=1.0,
            seed=100 + int(curr_x)
        )
        
        # B. 비교용 이상적인 평평한 막장면 생성
        ideal_face = RoughFace(
            base_x=curr_x,
            y_range=y_range,
            z_range=z_range,
            resolution=0.1,
            amplitude=0.0
        )
        
        # C. 3D 굴곡 교차선 추출
        rough_traces = extract_rough_traces(fracture_data, rough_face, poly_yz)
        ideal_traces = extract_rough_traces(fracture_data, ideal_face, poly_yz)
        
        print(f"    -> Extracted Traces: {len(rough_traces)} (Rough) vs {len(ideal_traces)} (Ideal)")
        
        # D. 고해상도 시각화 차트 저장 (오프스크린 강제 설정 적용됨)
        viz_name_2d = os.path.join(outdir, f"face_x_{int(curr_x)}m_comparison_2d.png")
        viz_name_3d = os.path.join(outdir, f"face_x_{int(curr_x)}m_view_3d.png")
        
        plot_rough_comparison_2d(rough_traces, ideal_traces, viz_name_2d)
        plot_rough_reconstruction(rough_face, rough_traces, ideal_traces, viz_name_3d)
        
        print(f"    -> Saved 2D Plot: {viz_name_2d}")
        print(f"    -> Saved 3D Plot: {viz_name_3d}")
        
    print("\n" + "=" * 80)
    print(f" All 4 discrete faces generated successfully! Outputs saved in: {outdir}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
