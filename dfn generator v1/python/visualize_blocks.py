"""
visualize_blocks.py
블록 탐지 결과 시각화 (PyVista 기반)

 - 3D 부드러운 메쉬 (Marching Cubes 적용)
 - 터널 반투명 메쉬 표시
 - 마우스 회전/확대 가능 대화형 3D 뷰어
"""

from __future__ import annotations
import numpy as np

try:
    import pyvista as pv
    from skimage.measure import marching_cubes
except ImportError:
    pv = None

def plot_block_3d_pyvista(
    labels: np.ndarray,       # (Nx, Ny, Nz) int32
    block_info: list,
    grid_info: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
    save_path: str = "block_3d_pyvista.png",
):
    """
    labels 에서 값이 0보다 큰 영역(블록)에 대해 Marching Cubes로
    부드러운 메쉬를 생성하여 PyVista 대화형 창에 렌더링합니다.
    """
    if pv is None:
        print("  [Viz] PyVista 또는 scikit-image가 설치되지 않아 3D 시각화를 건너뜁니다.")
        return

    n_blocks = len(block_info)
    if n_blocks == 0:
        print("  [Viz] 탐지된 블록 없음.")
        return

    print("  [Viz] PyVista 3D 렌더링 준비 중 (Marching Cubes 적용)...")

    # 그리드 좌표 및 간격 정보
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    vs = float(grid_info['voxel_size'])
    spacing = (vs, vs, vs)
    
    # 마스킹 배열에 사용하는 인덱스가 (0,0,0)일 때 실제 공간 좌표 시작점
    origin = np.array([xs[0], ys[0], zs[0]])

    plotter = pv.Plotter()
    plotter.set_background('white')

    # 보통 tab20과 같은 matplotlib 호환 컬러맵 사용 가능
    try:
        colors = pv.colors.get_cmap("tab20")
    except AttributeError:
        import matplotlib.cm as cm
        colors = cm.get_cmap("tab20", max(20, n_blocks))

    # 1. 터널 시각화 (반투명 원통형 메쉬 생성)
    if tunnel_poly_YZ is not None:
        xmin, xmax = xs[0], xs[-1]
        n_pts = len(tunnel_poly_YZ)
        
        pts = []
        faces = []
        for i in range(n_pts):
            y, z = tunnel_poly_YZ[i]
            pts.append([xmin, y, z])
            pts.append([xmax, y, z])
            
        # 표면을 삼각 메쉬 형태로 구성
        for i in range(n_pts - 1):
            p0 = 2 * i
            p1 = 2 * i + 1
            p2 = 2 * (i + 1)
            p3 = 2 * (i + 1) + 1
            # 2개의 삼각형(Quad) 추가
            faces.extend([3, p0, p1, p3])
            faces.extend([3, p0, p3, p2])
            
        tunnel_mesh = pv.PolyData(np.array(pts), np.array(faces))
        # 반투명 멘더링
        plotter.add_mesh(tunnel_mesh, color='lightblue', opacity=0.3, style='surface', label='Tunnel')

    # 2. 블록들에 대해 Marching Cubes 적용 및 렌더링
    for i, b in enumerate(block_info):
        label_id = b['label']
        # 특정 블록만의 불리언 마스크
        mask = (labels == label_id)
        
        try:
            # 부드러운 표면(Isosurface) 추출 (0과 1 사이의 경계인 0.5 레벨)
            verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
            
            # 배열 인덱스 기반 좌표(verts)를 실제 공간 좌표로 보정
            # marching_cubes가 생성한 verts 위치는 mask 배열의 인덱스에 spacing을 곱한 값이므로
            # 여기에 원점(origin)만 더해주면 실제 위치와 일치하게 됩니다.
            verts += origin
            
            # PyVista의 faces 배열 구조: [다각형 정점 수, 정점인덱스1, 정점인덱스2, 정점인덱스3]
            pv_faces = np.pad(faces, ((0, 0), (1, 0)), constant_values=3).flatten()
            
            block_mesh = pv.PolyData(verts, pv_faces)
            
            cidx = i % 20
            color = colors(cidx) if callable(colors) else colors.colors[cidx]
            
            plotter.add_mesh(block_mesh, color=color[:3], smooth_shading=True, label=f'Block {label_id}')
            
        except ValueError:
            # 해상도 대비 너무 작은 블록(예: 2x2x2 미만)은 추출 실패할 수 있음
            pass

    print("  [Viz] 인터랙티브 뷰어 창을 엽니다. 마우스로 회전 및 확대/축소가 가능합니다.")
    print("        창을 닫으면 파이프라인 처리가 완전히 종료됩니다.")
    
    # 축 좌표 표시
    plotter.show_axes()
    
    # 스크린샷 자동 저장 설정 가능 여부 (선택)
    # plotter.show(screenshot=save_path) 
    
    # 대화형 그래픽 띄우기
    plotter.show()

