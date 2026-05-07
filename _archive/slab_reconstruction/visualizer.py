import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from .slab_types import Slab, LocalCandidate, ReconstructedPlane, EvaluationResult

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

def plot_reconstruction_3d_pyvista(
    reconstructed_list: List[ReconstructedPlane],
    truth_centers: np.ndarray = None,
    truth_normals: np.ndarray = None,
    truth_radii: np.ndarray = None,
    tunnel_poly_yz: np.ndarray = None,
    x_range: Tuple[float, float] = (-25, 25),
    save_path: str = None
):
    """
    원본 DFN과 복원된 평면을 PyVista로 3D 중첩 시각화
    """
    if not HAS_PYVISTA:
        print("[Visualizer] PyVista가 없어 3D 시각화를 건너뜁니다.")
        return
        
    if not reconstructed_list:
        print("[Visualizer] 복원된 평면이 없어 시각화를 건너뜁니다.")
        return
        
    p = pv.Plotter()
    p.set_background("white")
    
    # 1. 터널 튜브 시각화
    if tunnel_poly_yz is not None:
        xmin, xmax = x_range
        n_pts = len(tunnel_poly_yz)
        pts = []
        for i in range(n_pts):
            y, z = tunnel_poly_yz[i]
            pts.append([xmin, y, z]); pts.append([xmax, y, z])
        faces = []
        for i in range(n_pts - 1):
            p0 = 2*i; p1 = 2*i+1; p3 = 2*(i+1)+1; p2 = 2*(i+1)
            faces.extend([3, p0, p1, p3]); faces.extend([3, p0, p3, p2])
        tunnel_mesh = pv.PolyData(np.array(pts), np.array(faces))
        p.add_mesh(tunnel_mesh, color='lightgray', opacity=0.15, label="Tunnel")

    # 2. 원본 DFN 시각화 (사용자 요청에 따라 속도 향상을 위해 비활성화)
    # 480만 개의 데이터를 렌더링할 경우 병목이 발생하므로 복원된 평면만 표시합니다.
    pass

    # 3. 복원된 평면 시각화 (불투명)
    recon_discs = []
    for recon in reconstructed_list:
        # 평면 크기는 임의로 설정 (포인트 익스텐트 활용 가능)
        disc = pv.Disc(center=recon.centroid, normal=recon.normal, inner=0.0, outer=10.0, c_res=24)
        recon_discs.append(disc)
        # 포인트 클라우드도 함께 표시
        p.add_mesh(pv.PolyData(recon.points), color='red', point_size=3.0)
        
    if recon_discs:
        recon_mesh = pv.MultiBlock(recon_discs)
        p.add_mesh(recon_mesh, color='red', opacity=0.7, label="Reconstructed Planes")

    p.add_legend()
    p.add_axes()
    
    if save_path:
        # 렌더링 후 스크린샷 저장
        p.show(screenshot=save_path, interactive=True)
    else:
        p.show()
    
    print(f"▶ [Viz] 3D 뷰어가 열렸습니다. ({len(reconstructed_list)} reconstructed)")

def plot_evaluation_metrics(eval_res: EvaluationResult, save_path: str):
    """지표 요약 그래프 생성"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    labels = ['Truth', 'Reconstructed', 'Matched']
    values = [eval_res.total_truth, eval_res.total_reconstructed, eval_res.matched_count]
    ax.bar(labels, values, color=['blue', 'red', 'green'])
    ax.set_title(f"Reconstruction Summary (Success: {eval_res.success_rate:.1f}%)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
