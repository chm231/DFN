import os
import numpy as np
import pyvista as pv
from typing import List, Dict, Any, Optional
from .generator import RoughFace

def plot_rough_reconstruction(
    rough_face: RoughFace,
    traces: List[Dict[str, Any]],
    ideal_traces: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None
):
    """
    Rough Face와 추출된 Polyline Trace들을 3D로 시각화합니다.
    """
    p = pv.Plotter()
    p.set_background("white")
    
    # 1. Rough Face 시각화 (StructuredGrid)
    # X, Y, Z shape: (NZ, NY)
    grid = pv.StructuredGrid(rough_face.X, rough_face.Y, rough_face.Z)
    p.add_mesh(grid, color="wheat", opacity=0.8, show_edges=False, label="Rough Excavation Face")
    
    # 2. Rough Traces 시각화 (Polylines)
    if traces:
        for t in traces:
            pts = t['points']
            if len(pts) < 2: continue
            # Polyline 생성
            poly = pv.PolyData(pts)
            lines = np.arange(len(pts))
            poly.lines = np.hstack(([len(pts)], lines))
            p.add_mesh(poly, color="red", line_width=4, label="Rough Trace" if t == traces[0] else None)
            
    # 3. Ideal Traces 시각화 (비교용 - 점선 또는 다른 색상)
    if ideal_traces:
        for t in ideal_traces:
            pts = t['points']
            if len(pts) < 2: continue
            poly = pv.PolyData(pts)
            lines = np.arange(len(pts))
            poly.lines = np.hstack(([len(pts)], lines))
            p.add_mesh(poly, color="blue", line_width=2, opacity=0.5, label="Ideal Trace (Flat)" if t == ideal_traces[0] else None)

    p.add_legend()
    p.add_axes()
    
    if save_path:
        # 스크린샷 저장
        p.show(screenshot=save_path, interactive=True)
    else:
        p.show()

def plot_rough_comparison_2d(
    traces: List[Dict[str, Any]],
    ideal_traces: List[Dict[str, Any]],
    save_path: str
):
    """
    Y-Z 평면에 투영된 이상적 Trace와 Rough Trace의 형상 차이를 비교합니다.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # Ideal (Blue)
    for t in ideal_traces:
        pts = t['points']
        plt.plot(pts[:, 1], pts[:, 2], 'b--', alpha=0.5, label='Ideal' if t == ideal_traces[0] else "")
        
    # Rough (Red)
    for t in traces:
        pts = t['points']
        plt.plot(pts[:, 1], pts[:, 2], 'r-', linewidth=2, label='Rough' if t == traces[0] else "")
        
    plt.title("Trace Comparison: Ideal (Flat) vs Rough Face")
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
