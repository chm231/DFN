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
except ImportError:
    pv = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_block_3d_pyvista_interactive(
    labels: np.ndarray,       # (Nx, Ny, Nz) int32
    block_info: list,
    grid_info: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
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
        print("  [Viz] 탐지된 블록 없음.")
        return

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

    # 블록 메쉬 (Marching Cubes)
    for i, b in enumerate(block_info):
        label_id = b['label']
        mask = (labels == label_id)
        
        try:
            verts, faces_mc, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
            verts += origin
            pv_faces = np.pad(faces_mc, ((0, 0), (1, 0)), constant_values=3).flatten()
            block_mesh = pv.PolyData(verts, pv_faces)
            
            cidx = i % 20
            color_val = cmap(cidx)[:3]  # RGB tuple
            
            plotter.add_mesh(block_mesh, color=color_val, smooth_shading=True, label=f'Block {label_id}')
        except ValueError:
            pass

    # x,y,z 그리드 표시
    plotter.show_grid(color='black', font_size=10)
    print("  [Viz] 인터랙티브 뷰어 창이 뜹니다. 마우스 회전/확대 가능.")
    plotter.show()


def plot_block_3d_scatter(
    labels: np.ndarray,
    block_info: list,
    grid_info: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
    max_voxels_per_block: int = 500,
    save_path: str = "block_3d_scatter.png",
):
    """3D 점묘법(Scatter)으로 블록 분포 시각화 (matplotlib) + 터널 형상 추가"""
    if not block_info:
        return

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
    block_info: list,
    grid_info: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
    save_path: str = "block_overview.png",
):
    """블록 3D 분포 개요 (4-panel) 대시보드 시각화"""
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
