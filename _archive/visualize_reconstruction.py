"""
[Direction B: Inverse Reconstruction]
역산된 3D 평면(Planes)과 터널의 외곽 형상을 PyVista를 사용해 3차원 공간에서 렌더링하는 시각화 모듈입니다.
"""
import pyvista as pv
import numpy as np
from typing import List
from .trace_types import ReconstructedPlane

def plot_reconstructed_planes_interactive(
    planes: List[ReconstructedPlane], 
    tunnel_poly_yz: np.ndarray, 
    start_x: float, 
    end_x: float,
    plane_size: float = 10.0,
    max_planes: int = 1000
):
    """
    복원된 3D Plane 객체 리스트와 터널 폴리곤(XZ)을 받아 PyVista 시각화 창(Interactive)에 출력합니다.
    (렌더링 부하를 막기 위해 최대 렌더링 평면 수를 제한할 수 있습니다)
    """
    print(f"\n[Viz] PyVista Interactive 3D 렌더링 준비 중... (평면 {len(planes)}개)")
    
    plotter = pv.Plotter()
    
    # 1. 터널 지오메트리 구축 (start_x ~ end_x 로 돌출)
    if tunnel_poly_yz is not None and len(tunnel_poly_yz) >= 3:
        # 터널의 시작면 점군
        n_pts = len(tunnel_poly_yz)
        pts_start = np.zeros((n_pts, 3))
        pts_start[:, 0] = start_x
        pts_start[:, 1] = tunnel_poly_yz[:, 0]
        pts_start[:, 2] = tunnel_poly_yz[:, 1]
        
        # 터널의 끝면 점군
        pts_end = np.zeros((n_pts, 3))
        pts_end[:, 0] = end_x
        pts_end[:, 1] = tunnel_poly_yz[:, 0]
        pts_end[:, 2] = tunnel_poly_yz[:, 1]
        
        # 터널 벽면 폴리곤(Faces) 구성
        faces = []
        for i in range(n_pts - 1):
            faces.extend([4, i, i+1, i+1+n_pts, i+n_pts])
        # 폴리곤 닫기 (마지막 점과 첫 점 연결)
        faces.extend([4, n_pts-1, 0, n_pts, 2*n_pts-1])
        
        all_pts = np.vstack([pts_start, pts_end])
        tunnel_mesh = pv.PolyData(all_pts, np.array(faces))
        tunnel_mesh.compute_normals(inplace=True)
        
        plotter.add_mesh(tunnel_mesh, color='lightblue', opacity=0.3, show_edges=True, label='Tunnel Wall')
    
    # 2. 복원된 평면(Plane) 시각화
    display_planes = planes
    if len(planes) > max_planes:
        print(f"  [WARN] Plane이 너무 많습니다({len(planes)}개)! 부하 방지를 위해 {max_planes}개만 무작위 추출 시각화합니다.")
        np.random.seed(42)
        idx = np.random.choice(len(planes), max_planes, replace=False)
        display_planes = [planes[i] for i in idx]
        
    for p in display_planes:
        center = (p.point_x, p.point_y, p.point_z)
        direction = (p.normal_x, p.normal_y, p.normal_z)
        
        # pyvista.Disc를 사용하여 실제 절리 원판의 형태로 시각화합니다 (inner=0, outer=추정반경)
        plane_mesh = pv.Disc(center=center, inner=0.0, outer=p.radius, normal=direction, r_res=1, c_res=36)
        
        # 반투명하게 여러 색상으로 표현
        plotter.add_mesh(plane_mesh, color='orange', opacity=0.6, show_edges=True)
        
    # 축과 그리드 설정
    plotter.show_grid()
    plotter.add_axes()
    plotter.camera_position = 'iso'
    
    # 창 띄우기 (Blocking)
    print("  [Viz] 3D 그래픽 창이 별도로 열립니다. 확인 후 창을 닫아주세요.")
    plotter.show(title="Direction B: Reconstructed Planes")
