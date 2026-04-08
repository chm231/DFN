r"""
plot_3d_tunnel_fractures.py
3차원 터널 튜브와 교차하는 균열원판(Disc)의 시각화 파이프라인.

옵션:
& "C:\Users\user\miniconda3\python.exe" "c:\Users\user\OneDrive\2026-1\3D DFN modeling\dfn generator v1\python\plot_3d_tunnel_fractures.py" --input "c:\Users\user\OneDrive\2026-1\3D DFN modeling\dfn generator v1\src\main\dfn_output_cube250m\dfn_export_for_python.h5"
  --mode [all | intersect | inside]
    - all: 터널 내부 완전히 포함된 균열 + 경계와 교차하는 균열 모두 표시
    - intersect: 터널 경계면을 뚫고 지나가는(교차하는) 균열만 표시
    - inside: 터널 튜브 안에 완전히 포함된(독립된) 균열만 표시
"""

import argparse
import sys
import numpy as np
import h5py
from matplotlib.path import Path

try:
    import pyvista as pv
except ImportError:
    print("[오류] PyVista 모듈이 필요합니다. ('pip install pyvista')")
    sys.exit(1)

def load_data(h5_path):
    print(f"📂 HDF5 로드 중: {h5_path}")
    data = {}
    with h5py.File(h5_path, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        data['centers'] = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        data['normals'] = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        data['radii']   = f['/fractures/radii'][:].ravel()
        
        if '/tunnel/poly_YZ' in f:
            raw_p = f['/tunnel/poly_YZ'][:]
            data['poly_YZ'] = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p
        else:
            print("[오류] HDF5 내에 터널 폴리곤 데이터가 없습니다.")
            sys.exit(1)
            
        if '/meta/crop_box' in f:
            data['crop_box'] = f['/meta/crop_box'][:].ravel()
        elif '/meta/domain_box' in f:
            data['crop_box'] = f['/meta/domain_box'][:].ravel()
        else:
            data['crop_box'] = None
    return data

def distance_to_polygon(pts_y, pts_z, poly_y, poly_z):
    """(N,) 개의 점과 M 단면 폴리곤 사이의 최단 거리를 벡터화 연산하여 반환합니다."""
    N = len(pts_y)
    M = len(poly_y)
    min_dsq = np.full(N, np.inf)
    for i in range(M):
        y1, z1 = poly_y[i], poly_z[i]
        y2, z2 = poly_y[(i+1)%M], poly_z[(i+1)%M]
        
        dy = y2 - y1
        dz = z2 - z1
        L2 = dy**2 + dz**2
        if L2 == 0:
            d2 = (pts_y - y1)**2 + (pts_z - z1)**2
        else:
            t = ((pts_y - y1) * dy + (pts_z - z1) * dz) / L2
            # 폴리곤 선분(0~1) 내로 클리핑
            t = np.clip(t, 0.0, 1.0)
            py = y1 + t * dy
            pz = z1 + t * dz
            d2 = (pts_y - py)**2 + (pts_z - pz)**2
        min_dsq = np.minimum(min_dsq, d2)
    return np.sqrt(min_dsq)

def main():
    parser = argparse.ArgumentParser(description="3D 터널-균열원판 교차 시각화 도구")
    parser.add_argument('--input', required=True, help="dfn_export_for_python.h5 경로")
    parser.add_argument('--mode', choices=['all', 'intersect', 'inside'], default='all',
                        help="all: 교차+완전포함 / intersect: 터널경계 교차만 / inside: 터널내부 완전포함만")
    parser.add_argument('--max_plot', type=int, default=10000, 
                        help="최대 렌더링 균열 수 제한 (과부하 방지)")
    args = parser.parse_args()

    data = load_data(args.input)
    centers = data['centers']
    normals = data['normals']
    radii = data['radii']
    poly_YZ = data['poly_YZ']
    poly_Y = poly_YZ[:, 0]
    poly_Z = poly_YZ[:, 1]
    
    crop = data['crop_box']
    if crop is not None:
        xmin, xmax = crop[0], crop[1]
    else:
        xmin, xmax = np.min(centers[:, 0]), np.max(centers[:, 0])

    N_total = len(radii)
    print(f"▶ 총 균열 데이터 스캔 중: {N_total:,} 개")

    # 1. Broad-phase 필터링 (X축 도메인 내 존재하는 것만)
    in_bbox = (centers[:, 0] + radii >= xmin) & (centers[:, 0] - radii <= xmax)
    centers = centers[in_bbox]
    normals = normals[in_bbox]
    radii = radii[in_bbox]
    
    # 2. 터널 폴리곤 (YZ평면) 기반 거리계산 및 내부상태 판별 연산
    cy, cz = centers[:, 1], centers[:, 2]
    dist_yz = distance_to_polygon(cy, cz, poly_Y, poly_Z)
    
    # Matplotlib Path 기반 in/out 판별
    poly_verts = np.column_stack([poly_Y, poly_Z])
    path = Path(poly_verts)
    pts_yz = np.column_stack([cy, cz])
    is_inside_yz = path.contains_points(pts_yz)
    
    # X축 포함 여부
    is_inside_x = (centers[:, 0] - radii >= xmin) & (centers[:, 0] + radii <= xmax)
    is_outside_x = (centers[:, 0] + radii < xmin) | (centers[:, 0] - radii > xmax)

    # 3. 위상학적 집합(Topology Sets) 체계 분류
    # 3-1. Fully Inside (완전히 포함): 중심이 YZ 내부이고 벽면거리가 반경이상 + X가 완전히 내부
    fully_inside = is_inside_yz & (dist_yz >= radii) & is_inside_x
    
    # 3-2. Fully Outside (완전히 외부): 중심이 YZ 외부이고 벽면거리가 반경이상 + X가 완전히 외부인 경우 등
    fully_outside_yz = (~is_inside_yz) & (dist_yz >= radii)
    fully_outside = fully_outside_yz | is_outside_x
    
    # 3-3. Boundary Intersecting (경계 교차): 내/외부 아무것도 아닌 나머지 합집합
    intersecting = ~(fully_inside | fully_outside)

    # 4. 사용자 지정 모드 필터링 적용
    if args.mode == 'inside':
        mask = fully_inside
        mode_str = "완전히 내부에 포함된 원판 (Fully Inside)"
    elif args.mode == 'intersect':
        mask = intersecting
        mode_str = "터널 튜브 표면을 자르고 교차하는 원판 (Boundary Intersecting)"
    else:
        mask = fully_inside | intersecting
        mode_str = "터널 내 연결된 모든 원판 (Inside + Intersecting)"

    final_centers = centers[mask]
    final_normals = normals[mask]
    final_radii = radii[mask]
    final_count = len(final_radii)

    print(f"▶ 공간 필터링 완료:")
    print(f"   - 터널 내 완전 포함: {fully_inside.sum():,} 개")
    print(f"   - 터널 경계 관통: {intersecting.sum():,} 개")
    print(f"   - 터널 외 완전 독립: {fully_outside.sum():,} 개")
    print(f"▶ 선택 모드({args.mode}) 렌더링 대기열: {final_count:,} 개")

    if final_count > args.max_plot:
        print(f"  [경고] 최대 렌더링 한계치 초과 ({args.max_plot}개). 무작위로 샘플링하여 렌더링합니다.")
        rng = np.random.default_rng(42)
        idx = rng.choice(final_count, args.max_plot, replace=False)
        final_centers = final_centers[idx]
        final_normals = final_normals[idx]
        final_radii = final_radii[idx]
        final_count = len(final_radii)

    # 5. PyVista 시각화 구성
    print("▶ PyVista 3D 엔진 구성 시작...")
    plotter = pv.Plotter()
    plotter.set_background("white")

    # 5-1. 터널 튜브 메쉬(반투명 원통) 생성
    n_pts = len(poly_YZ)
    pts = []
    for i in range(n_pts):
        y, z = poly_YZ[i]
        pts.append([xmin, y, z])
        pts.append([xmax, y, z])
    
    faces = []
    for i in range(n_pts - 1):
        p0 = 2 * i; p1 = 2 * i + 1; p2 = 2 * (i + 1); p3 = 2 * (i + 1) + 1
        faces.extend([3, p0, p1, p3])
        faces.extend([3, p0, p3, p2])
        
    tunnel_mesh = pv.PolyData(np.array(pts), np.array(faces))
    plotter.add_mesh(tunnel_mesh, color='darkgray', opacity=0.35, show_edges=True, edge_color='black', label="Tunnel Tube")

    # 5-2. 균열 원판 생성
    # 다중 PolyData 병합 최적화를 통해 렌더링 속도 개선
    discs = []
    cmap = pv.colors.get_cmap("tab20")
    for i in range(final_count):
        # pyvista.Disc(center, direction, inner, outer)
        # c_res: 원판 테두리의 다각형 분할 개수 (계산 효율을 위해 24각형 정도로 설정)
        disc = pv.Disc(center=final_centers[i], direction=final_normals[i], inner=0.0, outer=final_radii[i], c_res=24)
        discs.append(disc)

    if discs:
        # 단일 MultiBlock으로 결합하여 Plotter 부하 경감
        blocks = pv.MultiBlock(discs)
        plotter.add_mesh(blocks, cmap=cmap, show_scalar_bar=False, scalars=np.arange(final_count) % 20, 
                         opacity=0.85, smooth_shading=True)

    plotter.add_text(f"Mode: {args.mode.upper()}  |  Visible Fractures: {final_count:,}", 
                     position='upper_left', color='black', font_size=12)
    plotter.show_axes()
    
    print("\n✅ 창을 뛰웠습니다! 회전, 확대, 축소를 마음껏 조작해보세요.")
    plotter.show()


if __name__ == "__main__":
    main()
