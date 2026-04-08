"""
visualize_blocks.py
블록 탐지 결과 시각화 모듈 (PyVista 및 Matplotlib 기반)

 - 3D 부드러운 메쉬 (Marching Cubes 적용, PyVista)
 - 3D 점묘법 스캐터 (Matplotlib 3D)
 - 2D 단면 및 통계 대시보드 (Matplotlib)
"""

from __future__ import annotations
import numpy as np

try:
    import pyvista as pv
    from skimage.measure import marching_cubes
    from scipy import ndimage
except ImportError:
    pv = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_block_3d_pyvista_interactive(
    labels: np.ndarray,       # (Nx, Ny, Nz) int32
    state: np.ndarray,
    grid_info: dict,
    block_info: list,
    tunnel_poly_YZ: np.ndarray | None = None,
    downsample_stride: int = 2,
    save_path: str | None = None,
):
    """
    labels 에서 값이 0보다 큰 영역(블록)에 대해 Marching Cubes로
    부드러운 메쉬를 생성하여 PyVista 대화형 창에 렌더링합니다. (Interactive)
    x, y, z 그리드도 함께 표시됩니다.
    """
    if pv is None:
        print("  [Viz] PyVista 또는 scikit-image가 설치되지 않아 3D 시각화를 건너뜁니다.")
        return

    n_blocks = len(block_info)
    if n_blocks == 0:
        print("  [Viz] 탐지된 블록 없음 - 배경 및 터널만 시각화합니다.")

    print("  [Viz] PyVista Interactive 3D 렌더링 준비 중 (Marching Cubes 적용)...")

    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    vs = float(grid_info['voxel_size'])
    spacing = (vs, vs, vs)
    origin = np.array([xs[0], ys[0], zs[0]])

    plotter = pv.Plotter()
    plotter.set_background('white')

    try:
        cmap = cm.get_cmap('tab20', 20)
    except AttributeError:
        cmap = matplotlib.colormaps['tab20']

    # 터널 반투명 메쉬 (폴리곤 압출)
    if tunnel_poly_YZ is not None:
        xmin, xmax = xs[0], xs[-1]
        n_pts = len(tunnel_poly_YZ)
        pts = []
        for i in range(n_pts):
            y, z = tunnel_poly_YZ[i]
            pts.append([xmin, y, z])
            pts.append([xmax, y, z])
        
        faces = []
        for i in range(n_pts - 1):
            p0 = 2 * i; p1 = 2 * i + 1; p2 = 2 * (i + 1); p3 = 2 * (i + 1) + 1
            faces.extend([3, p0, p1, p3])
            faces.extend([3, p0, p3, p2])
            
        tunnel_mesh = pv.PolyData(np.array(pts), np.array(faces))
        plotter.add_mesh(tunnel_mesh, color='lightblue', opacity=0.3, style='surface', label='Tunnel')

    # 1. 빠른 바운딩 박스 탐색 (scipy.ndimage.find_objects)
    print("  [Viz] 블록 경계 상자(Bounding Box) 빠른 스캔 중...")
    try:
        slices = ndimage.find_objects(labels)
    except Exception as e:
        print(f"  [Viz] Warning: find_objects 실패 ({e}), fallback 사용.")
        slices = None

    # 블록 메쉬 (Marching Cubes)
    for i, b in enumerate(block_info):
        label_id = int(b['label'])
        
        # 2. Bounding Box Crop (1 voxel 패딩 추가로 닫힌 메쉬 유지)
        if slices is not None and label_id <= len(slices) and slices[label_id-1] is not None:
            s = slices[label_id-1]
            bbox_slice = tuple(slice(max(0, sl.start - 1), min(labels.shape[dim], sl.stop + 1)) 
                               for dim, sl in enumerate(s))
        else:
            coords = np.where(labels == label_id)
            if len(coords[0]) == 0: continue
            bbox_slice = tuple(slice(max(0, np.min(c)-1), min(labels.shape[dim], np.max(c)+2)) 
                               for dim, c in enumerate(coords))
            
        region = labels[bbox_slice]
        mask = (region == label_id)
        
        # 3. Downsampling (선택적)
        if downsample_stride > 1 and mask.size > 1000:
            stride = downsample_stride
        else:
            stride = 1
            
        # VTK 기반 고성능 메쉬 추출 (ImageData.contour)
        try:
            # VTK 그리드 설정
            Nx, Ny, Nz = mask.shape
            off = [origin[0] + bbox_slice[0].start * spacing[0], 
                   origin[1] + bbox_slice[1].start * spacing[1], 
                   origin[2] + bbox_slice[2].start * spacing[2]]
            
            try:
                grid = pv.ImageData(dimensions=(Nx, Ny, Nz), spacing=spacing, origin=off)
            except AttributeError:
                grid = pv.UniformGrid(dimensions=(Nx, Ny, Nz), spacing=spacing, origin=off)
                
            grid.point_data["values"] = mask.flatten(order="F")
            block_mesh = grid.contour([0.5])
            
            cidx = i % 20
            color_val = cmap(cidx)[:3]  # RGB tuple
            
            plotter.add_mesh(block_mesh, color=color_val, smooth_shading=True, label=f'Block {label_id}')
        except Exception as e:
            print(f"  [Viz][WARN] Block {label_id} mesh creation failed: {e}")
            pass

    # x,y,z 그리드 표시
    plotter.show_grid(color='black', font_size=10)
    
    # 뷰어 카메라 위치 한 번 초기화 (isometric)
    plotter.view_isometric()
    
    if save_path:
        try:
            # 1. 먼저 이미지를 렌더링하여 저장 (창을 띄우지 않고 메모리에서 수행 가능하도록 유도)
            plotter.show(screenshot=save_path, auto_close=False, interactive=False)
            print(f"  [Viz] 저장: {save_path}")
        except Exception as e:
            print(f"  [Viz][WARN] PyVista screenshot 저장 실패: {e}")

    # 2. 진짜 인터랙티브 창을 띄움 (여기서 코드가 멈추고 사용자의 조작을 대기함)
    print("  [Viz] 인터랙티브 뷰어 창이 뜹니다. 마우스 회전/확대 가능. (창을 닫으면 프로그램이 계속됩니다.)")
    plotter.show(interactive=True)


def _extract_block_patches(
    lbl: int,
    labels: np.ndarray,
    voxel_class: np.ndarray,
    fracture_owner: np.ndarray,
    fracture_data: dict,
    grid_info: dict,
    shell_thickness: int = 2,
    min_contact_voxels: int = 10
) -> list:
    """
    특정 블록(lbl)의 경계를 이루는 균열 원판 조각(Patch) 메쉬 리스트를 추출합니다.
    """
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    vs = grid_info['voxel_size']
    
    # 1. Bounding Box Crop
    objs = ndimage.find_objects((labels == lbl).astype(np.int32))
    if not objs or objs[0] is None: return []
    bbox_slice = objs[0]
    
    # 여유분 추가 (쉘 확장용)
    b = shell_thickness
    bbox_slice = (
        slice(max(0, bbox_slice[0].start-b), min(labels.shape[0], bbox_slice[0].stop+b)),
        slice(max(0, bbox_slice[1].start-b), min(labels.shape[1], bbox_slice[1].stop+b)),
        slice(max(0, bbox_slice[2].start-b), min(labels.shape[2], bbox_slice[2].stop+b))
    )
    
    sub_labels = labels[bbox_slice]
    sub_class = voxel_class[bbox_slice]
    sub_owner = fracture_owner[bbox_slice]
    
    # 2. 쉘(Shell) 추출 및 접촉 균열 식별
    block_mask = (sub_labels == lbl)
    shell = ndimage.binary_dilation(block_mask, iterations=shell_thickness) & ~block_mask
    
    contact_owners = sub_owner[shell & (sub_class == 1)] # FRACTURE 클래스만
    unique_owners, counts = np.unique(contact_owners, return_counts=True)
    
    valid_fids = unique_owners[(unique_owners > 0) & (counts >= min_contact_voxels)]
    
    patches = []
    for fid in valid_fids:
        # 이 균열에 해당하는 복셀 좌표들
        f_mask = (sub_owner == fid) & shell
        coords = np.argwhere(f_mask)
        
        # 실제 좌표로 변환
        pts = coords.copy().astype(float)
        pts[:, 0] = xs[0] + (coords[:, 0] + bbox_slice[0].start) * vs
        pts[:, 1] = ys[0] + (coords[:, 1] + bbox_slice[1].start) * vs
        pts[:, 2] = zs[0] + (coords[:, 2] + bbox_slice[2].start) * vs
        
        # 균열 정보 (법선 등)
        f_idx = fid - 1
        if f_idx >= len(fracture_data['normals']): continue
        norm = fracture_data['normals'][f_idx]
        
        patch_mesh = _create_fracture_patch_mesh(pts, norm)
        if patch_mesh:
            patches.append(patch_mesh)
            
    return patches


def plot_all_blocks_with_fractures(
    labels: np.ndarray,
    voxel_class: np.ndarray,
    fracture_owner: np.ndarray,
    grid_info: dict,
    block_info: list,
    fracture_data: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
    shell_thickness: int = 2,
    min_contact_voxels: int = 15,
    save_path: str = None
):
    """
    [사진 2] 모든 블록과 그 경계 균열 패치를 하나의 화면에 통합 시각화합니다.
    """
    # GPU 데이터 변환
    if hasattr(labels, 'get'): labels = labels.get()
    if hasattr(voxel_class, 'get'): voxel_class = voxel_class.get()
    if hasattr(fracture_owner, 'get'): fracture_owner = fracture_owner.get()
    
    print(f"\n  [Viz] 전역 인터페이스 시각화(사진 2) 중... (블록: {len(block_info)}개)")
    
    plotter = pv.Plotter()
    plotter.set_background('white')
    cmap = plt.get_cmap('tab20')
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    vs = grid_info['voxel_size']

    for i, b in enumerate(block_info):
        lbl = b['label']
        color = cmap(i % 20)[:3]
        
        # 균열 패치들 (블록 본체 메쉬는 제외)
        patches = _extract_block_patches(lbl, labels, voxel_class, fracture_owner, fracture_data, grid_info, shell_thickness, min_contact_voxels)
        for p in patches:
            plotter.add_mesh(p, color=color, opacity=0.4, line_width=1)

    # 터널
    if tunnel_poly_YZ is not None:
        xmin, xmax = xs[0], xs[-1]
        pts, faces = [], []
        for pt in tunnel_poly_YZ:
            pts.append([xmin, pt[0], pt[1]]); pts.append([xmax, pt[0], pt[1]])
        for j in range(0, len(tunnel_poly_YZ)*2 -2, 2):
            faces.extend([4, j, j+1, j+3, j+2])
        plotter.add_mesh(pv.PolyData(np.array(pts), np.array(faces)), color='lightblue', opacity=0.1, style='wireframe')

    plotter.show_grid(color='black', font_size=10)
    plotter.view_isometric()
    
    print("  [Viz] 전역 인터페이스 뷰어 창을 엽니다. (사진 2 저장 포함)")
    plotter.show(screenshot=save_path)


def _create_fracture_patch_mesh(points: np.ndarray, normal: np.ndarray):
    """
    3D 점들을 평면에 투영하여 Delaunay 2D 메쉬를 생성한 뒤, 다시 3D로 복원합니다.
    """
    if len(points) < 3:
        return None
    
    normal = np.array(normal) / np.linalg.norm(normal)
    origin = points.mean(axis=0)
    
    # 1. 로컬 좌표계 (u, v, n) 생성
    if abs(normal[2]) < 0.9:
        axis = np.array([0, 0, 1])
    else:
        axis = np.array([0, 1, 0])
    u = np.cross(normal, axis); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # 2. 2D 투영 (u, v 평면)
    rel_pts = points - origin
    pts_2d = np.zeros((len(points), 2))
    pts_2d[:, 0] = rel_pts @ u
    pts_2d[:, 1] = rel_pts @ v
    
    # 3. Delaunay 2D (PyVista)
    poly_2d = pv.PolyData(np.column_stack([pts_2d, np.zeros(len(pts_2d))]))
    mesh_2d = poly_2d.delaunay_2d()
    
    # 4. 3D 복원 (Delaunay가 점 순서를 바꿀 수 있으므로 mesh_2d.points 사용)
    pts_projected_2d = mesh_2d.points[:, :2]
    pts_3d = (pts_projected_2d[:, 0:1] * u + 
              pts_projected_2d[:, 1:2] * v + origin)
    
    return pv.PolyData(pts_3d, mesh_2d.faces)


def plot_block_with_bounding_fractures(
    labels: np.ndarray,
    voxel_class: np.ndarray,
    fracture_owner: np.ndarray,
    target_label: int,
    fracture_data: dict,
    grid_info: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
    shell_thickness: int = 2,
    min_contact_voxels: int = 15,
    show_block_surface: bool = True,
    show_all_blocks: bool = False,
    show_fractures: bool = False,
    interactive: bool = True,
    save_path: str = None
):
    """
    특정 블록(target_label)과 그 블록의 경계를 형성하는 실제 균열(Fracture Discs)을 함께 시각화합니다.
    """
    if pv is None:
        print("  [Viz] PyVista가 필요합니다.")
        return

    # GPU(CuPy) 데이터일 경우 CPU(NumPy)로 전송
    if hasattr(labels, 'get'): labels = labels.get()
    if hasattr(voxel_class, 'get'): voxel_class = voxel_class.get()
    if hasattr(fracture_owner, 'get'): fracture_owner = fracture_owner.get()

    print(f"  [Viz] 블록 #{target_label} 및 경계 균열 추출 중...")
    
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    vs = float(grid_info['voxel_size'])
    origin = np.array([xs[0], ys[0], zs[0]])

    # 1. 블록 마스크 및 경계 쉘 추출
    block_mask = (labels == target_label)
    if not np.any(block_mask):
        print(f"  [ERROR] 블록 ID {target_label}을 찾을 수 없습니다.")
        return

    # 팽창(Dilation)을 통해 블록 주변 쉘 획득
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(block_mask, structure=struct, iterations=shell_thickness)
    shell = dilated & (~block_mask)
    
    # 2. 인접 균열 ID 추출 및 통계
    adj_owners = fracture_owner[shell]
    unique_ids, counts = np.unique(adj_owners, return_counts=True)
    
    # 유효한 균열(ID >= 0)만 필터링
    valid = (unique_ids >= 0)
    unique_ids = unique_ids[valid]
    counts = counts[valid]
    
    # 접촉 복셀 수 기준 내림차순 정렬
    sort_idx = np.argsort(-counts)
    unique_ids = unique_ids[sort_idx]
    counts = counts[sort_idx]

    # 최소 접촉 수 필터링
    meaningful = counts >= min_contact_voxels
    unique_ids = unique_ids[meaningful]
    counts = counts[meaningful]

    print(f"    - 감지된 인접 균열: {len(unique_ids)}개 (min_contact={min_contact_voxels})")

    # 3. 렌더링 준비
    plotter = pv.Plotter()
    plotter.set_background('white')

    # (B) 터널 지오메트리
    if tunnel_poly_YZ is not None:
        xmin, xmax = xs[0], xs[-1]
        pts = []; faces = []
        for i, (y, z) in enumerate(tunnel_poly_YZ):
            pts.append([xmin, y, z]); pts.append([xmax, y, z])
            if i < len(tunnel_poly_YZ)-1:
                p0=2*i; p1=2*i+1; p2=2*(i+1); p3=2*(i+1)+1
                faces.extend([3, p0, p1, p3]); faces.extend([3, p0, p3, p2])
        tunnel_mesh = pv.PolyData(np.array(pts), np.array(faces))
        plotter.add_mesh(tunnel_mesh, color='lightblue', opacity=0.15, style='wireframe', label='Tunnel')

    # (C) 블록 표면 (PyVista/VTK Optimized)
    if show_block_surface:
        # Bounding Box Crop for efficiency
        objs = ndimage.find_objects(block_mask.astype(np.int32))
        if not objs or objs[0] is None:
            print(f"    [WARN] Block {target_label} has no valid bounding box.")
            return

        obj = objs[0]
        bbox = tuple(slice(max(0, s.start-1), min(labels.shape[d], s.stop+1)) for d, s in enumerate(obj))
        local_mask = block_mask[bbox]
        
        try:
            verts, faces_mc, _, _ = marching_cubes(local_mask, level=0.5, spacing=(vs,vs,vs))
            verts += origin + np.array([bbox[0].start*vs, bbox[1].start*vs, bbox[2].start*vs])
            pv_faces = np.pad(faces_mc, ((0,0), (1,0)), constant_values=3).flatten()
            block_mesh = pv.PolyData(verts, pv_faces)
            plotter.add_mesh(block_mesh, color='lightgray', opacity=0.5, smooth_shading=True, label=f'Block {target_label}')
        except Exception as e:
            print(f"    [WARN] Block surface generation failed: {e}")

    # (C) 균열 평면 패치 메쉬 생성 및 렌더링
    rng = np.random.default_rng(target_label)
    normals = fracture_data['normals']
    
    # 그리드 좌표 준비 (복셀 중심 좌표 추출용)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    for i, fid in enumerate(unique_ids):
        # 해당 균열의 접촉 복셀 좌표 추출
        fid_mask = (fracture_owner == fid) & shell
        pts = np.column_stack([XX[fid_mask], YY[fid_mask], ZZ[fid_mask]])
        
        n = normals[fid]
        patch = _create_fracture_patch_mesh(pts, n)
        
        if patch:
            color = rng.random(3) # 랜덤 색상
            plotter.add_mesh(patch, color=color, opacity=0.9, 
                             label=f'Frac {fid} (vox={counts[i]})',
                             show_edges=True if len(unique_ids) < 10 else False)
            print(f"      - Frac {fid:4d}: contact voxels = {counts[i]:4d} (Plane Patch)")
        else:
            print(f"      - Frac {fid:4d}: points too few for patch.")

    plotter.show_grid(color='gray', font_size=8)
    plotter.add_legend(size=(0.15, 0.15))
    plotter.view_isometric()
    
    if save_path:
        # 렌더링 후 스크린샷 저장
        plotter.show(auto_close=False, interactive_update=True)
        plotter.screenshot(save_path)
        print(f"    - Visualization saved to: {save_path}")
    
    if interactive:
        print(f"  [Viz] 블록 시괄화 완료 (ID: {target_label}) - 인터랙티브 창을 엽니다.")
        plotter.show()
    else:
        plotter.close()


def plot_block_3d_scatter(
    labels: np.ndarray,
    state: np.ndarray,
    grid_info: dict,
    block_info: list,
    tunnel_poly_YZ: np.ndarray | None = None,
    max_voxels_per_block: int = 500,
    save_path: str = "block_3d_scatter.png",
):
    """3D 점묘법(Scatter)으로 블록 분포 시각화 (matplotlib) + 터널 형상 추가"""
    if hasattr(labels, 'get'): labels = labels.get()
    if hasattr(state, 'get'): state = state.get()
    
    if not block_info:
        print("  [Viz] 탐지된 블록 없음 - 3D Scatter 배경/터널 렌더링 중...")

    print("  [Viz] 3D Scatter (matplotlib) 렌더링 중...")
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    fig = plt.figure(figsize=(12, 9), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    try:
        cmap = cm.get_cmap('tab20', max(len(block_info), 1))
    except AttributeError:
        cmap = matplotlib.colormaps['tab20']
        
    rng = np.random.default_rng(42)

    for i, b in enumerate(block_info):
        mask = labels == b['label']
        xi = XX[mask]; yi = YY[mask]; zi = ZZ[mask]
        if len(xi) > max_voxels_per_block:
            idx = rng.choice(len(xi), max_voxels_per_block, replace=False)
            xi, yi, zi = xi[idx], yi[idx], zi[idx]
        ax.scatter(xi, yi, zi, c=[cmap(i % 20)], s=4, alpha=0.9, label=f"Block {b['label']}")

    # 터널 3D 와이어프레임 렌더링
    if tunnel_poly_YZ is not None:
        xmin, xmax = xs[0], xs[-1]
        y_poly = np.append(tunnel_poly_YZ[:,0], tunnel_poly_YZ[0,0])
        z_poly = np.append(tunnel_poly_YZ[:,1], tunnel_poly_YZ[0,1])
        
        ax.plot(np.full_like(y_poly, xmin), y_poly, z_poly, color='darkred', linewidth=1.5, alpha=0.5)
        ax.plot(np.full_like(y_poly, xmax), y_poly, z_poly, color='darkred', linewidth=1.5, alpha=0.5)
        for y, z in tunnel_poly_YZ:
            ax.plot([xmin, xmax], [y, y], [z, z], color='darkred', linewidth=1.0, alpha=0.3)

    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('Z (m)', labelpad=6)
    ax.set_title(f'3D Block Distribution with Tunnel ({len(block_info)} blocks)', fontsize=14)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [Viz] 저장: {save_path}")
    plt.close(fig)


def plot_block_overview(
    labels: np.ndarray,
    state: np.ndarray,
    grid_info: dict,
    block_info: list,
    tunnel_poly_YZ: np.ndarray | None = None,
    save_path: str = "block_overview.png",
):
    """블록 3D 분포 개요 (4-panel) 대시보드 시각화"""
    if hasattr(labels, 'get'): labels = labels.get()
    if hasattr(state, 'get'): state = state.get()
    
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    Nx, Ny, Nz = labels.shape

    n_blocks = len(block_info)
    if n_blocks == 0:
        return

    print("  [Viz] 2D Overview Dashboard 렌더링 중...")

    try:
        cmap = cm.get_cmap('tab20', max(n_blocks, 1))
    except AttributeError:
        cmap = matplotlib.colormaps['tab20']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a2e')
    fig.suptitle(f'3D DFN Block Detection Results\n{n_blocks} blocks detected',
                 color='white', fontsize=14, fontweight='bold')

    for ax in axes.ravel():
        ax.set_facecolor('#16213e')

    # Panel 1: YZ slice 
    ax1 = axes[0, 0]
    xmid = Nx // 2
    slice_yz = labels[xmid, :, :]
    im1 = ax1.imshow(slice_yz.T, origin='lower',
                     extent=[ys[0], ys[-1], zs[0], zs[-1]],
                     cmap='tab20', aspect='equal', interpolation='nearest',
                     vmin=0, vmax=max(n_blocks, 1))
    if tunnel_poly_YZ is not None:
        ax1.plot(tunnel_poly_YZ[:, 0], tunnel_poly_YZ[:, 1],
                 'w-', linewidth=1.5, label='Tunnel')
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax1.set_xlabel('Y (m)', color='white')
    ax1.set_ylabel('Z (m)', color='white')
    ax1.set_title(f'YZ slice @ X={xs[xmid]:.1f}m', color='white', fontsize=10)
    ax1.tick_params(colors='white')
    cb = plt.colorbar(im1, ax=ax1, label='Block ID')
    cb.ax.yaxis.set_tick_params(color='white')
    cb.set_label('Block ID', color='white')
    cb.ax.yaxis.set_tick_params(labelcolor='white')

    # Panel 2: XY slice
    ax2 = axes[0, 1]
    zmid = Nz // 2
    slice_xy = labels[:, :, zmid]
    im2 = ax2.imshow(slice_xy.T, origin='lower',
                     extent=[xs[0], xs[-1], ys[0], ys[-1]],
                     cmap='tab20', aspect='auto', interpolation='nearest',
                     vmin=0, vmax=max(n_blocks, 1))
    ax2.set_xlabel('X (m)', color='white')
    ax2.set_ylabel('Y (m)', color='white')
    ax2.set_title(f'XY slice @ Z={zs[zmid]:.1f}m', color='white', fontsize=10)
    ax2.tick_params(colors='white')

    # Panel 3: Volume Histogram
    ax3 = axes[1, 0]
    vols = [b['volume_m3'] for b in block_info]
    ax3.hist(vols, bins=min(30, max(n_blocks, 1)), color='#e94560', edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Block Volume (m³)', color='white')
    ax3.set_ylabel('Count', color='white')
    ax3.set_title('Block Volume Distribution', color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white')
    if len(vols) > 0:
        ax3.axvline(np.median(vols), color='yellow', linestyle='--',
                    linewidth=1.5, label=f'Median: {np.median(vols):.2f} m³')
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # Panel 4: Table summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    top_n = min(10, n_blocks)
    table_data = []
    for i, b in enumerate(block_info[:top_n]):
        cx, cy, cz = b['centroid']
        table_data.append([
            f"{b['label']}",
            f"{b['n_voxels']:,}",
            f"{b['volume_m3']:.3f}",
            f"({cx:.1f},{cy:.1f},{cz:.1f})",
        ])
    if table_data:
        col_labels = ['Block ID', 'Voxels', 'Volume (m³)', 'Centroid (m)']
        tbl = ax4.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#0f3460' if r == 0 else '#16213e')
            cell.set_text_props(color='white')
            cell.set_edgecolor('#e94560')
    ax4.set_title(f'Top-{top_n} Blocks by Volume', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  [Viz] 저장: {save_path}")
    plt.close(fig)
