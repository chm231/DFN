"""
visualize_blocks.py
블록 탐지 결과 시각화

matplotlib 기반:
 - 3D 산점도 (블록별 색상)
 - YZ 단면 슬라이스
 - XY/XZ 슬라이스
 - 블록 볼륨 히스토그램
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경 대응 (GUI 있으면 'TkAgg' 또는 제거)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm


def plot_block_overview(
    labels: np.ndarray,       # (Nx, Ny, Nz) int32
    block_info: list,
    grid_info: dict,
    tunnel_poly_YZ: np.ndarray | None = None,
    save_path: str = "block_overview.png",
):
    """블록 3D 분포 개요 (4-panel)."""
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    Nx, Ny, Nz = labels.shape

    n_blocks = len(block_info)
    if n_blocks == 0:
        print("  [Viz] 탐지된 블록 없음.")
        return

    cmap = cm.get_cmap('tab20', max(n_blocks, 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a2e')
    fig.suptitle(f'3D DFN Block Detection Results\n{n_blocks} blocks detected',
                 color='white', fontsize=14, fontweight='bold')

    for ax in axes.ravel():
        ax.set_facecolor('#16213e')

    # ── Panel 1: YZ 단면 (X 중앙) ──────────────────────────────────────
    ax1 = axes[0, 0]
    xmid = Nx // 2
    slice_yz = labels[xmid, :, :]  # (Ny, Nz)
    im1 = ax1.imshow(slice_yz.T, origin='lower',
                     extent=[ys[0], ys[-1], zs[0], zs[-1]],
                     cmap='tab20', aspect='equal', interpolation='nearest',
                     vmin=0, vmax=max(n_blocks, 1))
    if tunnel_poly_YZ is not None:
        ax1.plot(tunnel_poly_YZ[:, 0], tunnel_poly_YZ[:, 1],
                 'w-', linewidth=1.5, label='Tunnel')
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax1.set_xlabel('Y (m)', color='white'); ax1.set_ylabel('Z (m)', color='white')
    ax1.set_title(f'YZ slice @ X={xs[xmid]:.1f}m', color='white', fontsize=10)
    ax1.tick_params(colors='white')
    plt.colorbar(im1, ax=ax1, label='Block ID').ax.yaxis.set_tick_params(color='white')

    # ── Panel 2: XY 단면 (Z 중앙) ──────────────────────────────────────
    ax2 = axes[0, 1]
    zmid = Nz // 2
    slice_xy = labels[:, :, zmid]  # (Nx, Ny)
    im2 = ax2.imshow(slice_xy.T, origin='lower',
                     extent=[xs[0], xs[-1], ys[0], ys[-1]],
                     cmap='tab20', aspect='auto', interpolation='nearest',
                     vmin=0, vmax=max(n_blocks, 1))
    ax2.set_xlabel('X (m)', color='white'); ax2.set_ylabel('Y (m)', color='white')
    ax2.set_title(f'XY slice @ Z={zs[zmid]:.1f}m', color='white', fontsize=10)
    ax2.tick_params(colors='white')

    # ── Panel 3: 블록 볼륨 히스토그램 ──────────────────────────────────
    ax3 = axes[1, 0]
    vols = [b['volume_m3'] for b in block_info]
    ax3.hist(vols, bins=min(30, n_blocks), color='#e94560', edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Block Volume (m³)', color='white')
    ax3.set_ylabel('Count', color='white')
    ax3.set_title('Block Volume Distribution', color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white')
    if len(vols) > 0:
        ax3.axvline(np.median(vols), color='yellow', linestyle='--',
                    linewidth=1.5, label=f'Median: {np.median(vols):.2f} m³')
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # ── Panel 4: 블록 통계 테이블 ──────────────────────────────────────
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
            f"({cx:.1f}, {cy:.1f}, {cz:.1f})",
        ])
    col_labels = ['Block ID', 'Voxels', 'Volume (m³)', 'Centroid (m)']
    tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                    loc='center', cellLoc='center')
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


def plot_block_3d_scatter(
    labels: np.ndarray,
    block_info: list,
    grid_info: dict,
    max_voxels_per_block: int = 500,
    save_path: str = "block_3d_scatter.png",
):
    """3D 산점도로 블록 위치 시각화."""
    if not block_info:
        return

    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    fig = plt.figure(figsize=(12, 9), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#16213e')

    cmap = cm.get_cmap('tab20', max(len(block_info), 1))
    rng = np.random.default_rng(42)

    for i, b in enumerate(block_info):
        mask = labels == b['label']
        xi = XX[mask]; yi = YY[mask]; zi = ZZ[mask]
        if len(xi) > max_voxels_per_block:
            idx = rng.choice(len(xi), max_voxels_per_block, replace=False)
            xi, yi, zi = xi[idx], yi[idx], zi[idx]
        ax.scatter(xi, yi, zi, c=[cmap(i % 20)], s=4, alpha=0.6)

    ax.set_xlabel('X (m)', color='white', labelpad=6)
    ax.set_ylabel('Y (m)', color='white', labelpad=6)
    ax.set_zlabel('Z (m)', color='white', labelpad=6)
    ax.set_title(f'3D Block Distribution ({len(block_info)} blocks)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  [Viz] 저장: {save_path}")
    plt.close(fig)
