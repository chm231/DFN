r"""
run_trace_pipeline.py  –  2D Trace 기반 역산 및 블록 탐지 파이프라인

사용법:
    & "C:\Users\user\miniconda3\python.exe" "c:\Users\user\OneDrive\2026-1\3D DFN modeling\trace_analysis\run_trace_pipeline.py"

결과물:
    - storage/output/actual_dfn_reconstruction_results/ 내부에 CSV 및 시각화 결과 저장
"""
import os
import sys
import numpy as np
import h5py
from typing import List

# Import local modules
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
_core_path = os.path.join(_parent, "dfn_analysis")

sys.path.insert(0, _here)
sys.path.insert(0, _core_path) # Core engine (block_detector, etc.)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_reconstruction.trace_types import ExcavationFace, FaceTrace
from trace_reconstruction.excavation_face_traces import extract_excavation_face_traces_from_truth
from trace_reconstruction.face_trace_io import save_face_traces
from trace_reconstruction.reconstruction_pipeline import run_inverse_pipeline
from visualize_blocks import plot_block_3d_pyvista_interactive

def main():
    # 1. Config
    dfn_path = r"c:\Users\user\OneDrive\2026-1\3D DFN modeling\storage\data\dfn_export_for_python.h5"
    dat_path = r"c:\Users\user\OneDrive\2026-1\3D DFN modeling\storage\data\단면_폴리곤.dat"
    out_dir = r"c:\Users\user\OneDrive\2026-1\3D DFN modeling\storage\output\actual_dfn_reconstruction_results"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*60)
    print(" [Inverse Reconstruction] Actual DFN -> Traces -> Blocks")
    print("="*60)
    
    # 2. Load Tunnel Polygon
    print(f"\n[1/5] Loading tunnel polygon from {dat_path}...")
    poly_y, poly_z = load_tunnel_polygon_from_dat(dat_path)
    poly_yz = np.column_stack([poly_y, poly_z])
    print(f" -> Loaded {len(poly_yz)} points. Units converted to meters.")
    print(f" -> Polygon Y range: {np.min(poly_y):.1f} to {np.max(poly_y):.1f}")
    print(f" -> Polygon Z range: {np.min(poly_z):.1f} to {np.max(poly_z):.1f}")
    
    # 3. Load Actual DFN
    print(f"\n[2/5] Loading actual DFN from {dfn_path}...")
    with h5py.File(dfn_path, 'r') as f:
        centers = f['/fractures/centers'][:]
        normals = f['/fractures/normals'][:]
        radii = f['/fractures/radii'][:].ravel()
        crop_box = f['/meta/crop_box'][:].ravel()
        
        # Transpose correction (MATLAB vs Python)
        if centers.shape[0] == 3 and centers.shape[0] < centers.shape[1]:
            centers = centers.T
        if normals.shape[0] == 3 and normals.shape[0] < normals.shape[1]:
            normals = normals.T
            
    start_x, end_x = float(crop_box[0]), float(crop_box[1])
    print(f" -> DFN loaded: {len(radii)} fractures. X range: {start_x:.1f} to {end_x:.1f}")
    
    # 4. Generate 2D Traces from Tunnel Path
    print(f"\n[3/5] Extracting 2D traces along the tunnel path (X-axis)...")
    dx = 3.0 # Tunnel advance step
    x_positions = np.arange(start_x, end_x, dx)
    all_traces = []
    
    for i, x_pos in enumerate(x_positions):
        face = ExcavationFace(face_id=i+1, x_face=x_pos, tunnel_polygon_yz=poly_yz, advance_step=dx)
        curr_traces = extract_excavation_face_traces_from_truth(centers, normals, radii, face)
        all_traces.extend(curr_traces)
        
    traces_csv = os.path.join(out_dir, "extracted_actual_traces.csv")
    save_face_traces(all_traces, traces_csv)
    print(f" -> Extracted {len(all_traces)} traces across {len(x_positions)} faces.")
    print(f" -> Saved to: {traces_csv}")
    
    # 5. Run Inverse Pipeline
    print(f"\n[4/5] Running Inverse Pipeline (Track matching -> Plane reconstruction -> Block detection)...")
    run_params = {
        'block_kwargs': {
            'voxel_size': 0.5,
            'tol_factor': 0.6,
            'min_voxels': 8,
            'connectivity': 26
        }
    }
    planes, blocks, labels, grid_info = run_inverse_pipeline(traces_csv, poly_yz, start_x, end_x, params=run_params)
    
    print(f"\n[Result Summary]")
    print(f" -> Reconstructed Planes: {len(planes)}")
    print(f" -> Detected Blocks: {len(blocks)}")
    
    # 6. Visualization
    if blocks:
        print(f"\n[5/5] Launching 3D Interactive Viewer...")
        vis_save_path = os.path.join(out_dir, "reconstructed_block_viz.png")
        # dummy state as it's not strictly used in mesh gen but required in signature
        dummy_state = np.zeros(labels.shape, dtype=np.uint8) 
        plot_block_3d_pyvista_interactive(
            labels, dummy_state, grid_info, blocks, 
            tunnel_poly_YZ=poly_yz,
            save_path=vis_save_path
        )
    else:
        print("\n[!] No blocks detected from the reconstructed planes.")
        if planes:
             print(" -> Planes were reconstructed but no stable blocks formed with the tunnel.")
        else:
             print(" -> No planes were reconstructed. Check matching parameters.")

if __name__ == "__main__":
    main()
