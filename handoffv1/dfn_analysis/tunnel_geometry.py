"""
tunnel_geometry.py
터널 형상 → 3D 복셀 마스크 생성 (CPU)

_archive/block_detection/code/tunnel_geometry.py 에서 가져와, 블록 탐지에 쓰지 않는
전체 격자 좌표 배열(voxel_centers)·halo 마스크를 뺀 판본이다.

좌표 규약: x = 터널 굴진 방향, 터널 단면 다각형은 y-z 평면([y, z])에 주어지고
x 방향으로 압출(extrude)된다.
"""

from __future__ import annotations
import numpy as np


def point_in_polygon_2d(py: np.ndarray, pz: np.ndarray,
                         poly_y: np.ndarray, poly_z: np.ndarray) -> np.ndarray:
    """Ray-casting 알고리즘으로 2D 점이 폴리곤 내부인지 판별 (CPU, boolean array)."""
    n = len(poly_y)
    inside = np.zeros(len(py), dtype=bool)
    j = n - 1
    for i in range(n):
        yi, zi = poly_y[i], poly_z[i]
        yj, zj = poly_y[j], poly_z[j]
        cond = ((zi > pz) != (zj > pz)) & \
               (py < (yj - yi) * (pz - zi) / (zj - zi + 1e-15) + yi)
        inside ^= cond
        j = i
    return inside


def build_voxel_masks(
    poly_Y: np.ndarray,
    poly_Z: np.ndarray,
    domain_box: np.ndarray,
    voxel_size: float = 0.5,
    tunnel_xmin: float | None = None,
    tunnel_xmax: float | None = None,
) -> tuple:
    """
    Parameters
    ----------
    poly_Y, poly_Z : 터널 단면 폴리곤 좌표 (m)
    domain_box     : [xmin, xmax, ymin, ymax, zmin, zmax]
    voxel_size     : 복셀 한 변 길이 (m)
    tunnel_xmin/xmax : 터널 X 방향 범위 (None이면 도메인 전체)

    Returns
    -------
    tunnel_mask : (Nx,Ny,Nz) bool – 터널 내부(굴착 공간)
    grid_info   : dict (원점, 복셀 수, 복셀 크기, 복셀 중심 좌표축 xs/ys/zs)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = np.asarray(domain_box, dtype=float)

    # 그리드 생성 (복셀 중심 좌표)
    xs = np.arange(xmin + voxel_size / 2, xmax, voxel_size, dtype=np.float32)
    ys = np.arange(ymin + voxel_size / 2, ymax, voxel_size, dtype=np.float32)
    zs = np.arange(zmin + voxel_size / 2, zmax, voxel_size, dtype=np.float32)

    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    print(f"  Grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} voxels  (voxel={voxel_size}m)")

    # YZ 평면 격자 – 터널 단면 마스크
    YY, ZZ = np.meshgrid(ys, zs, indexing='ij')  # (Ny, Nz)
    inside_yz = point_in_polygon_2d(YY.ravel(), ZZ.ravel(), poly_Y, poly_Z).reshape(Ny, Nz)

    # 터널 X 범위 마스크
    if tunnel_xmin is None:
        tunnel_xmin = xmin
    if tunnel_xmax is None:
        tunnel_xmax = xmax
    x_in = (xs >= tunnel_xmin) & (xs <= tunnel_xmax)  # (Nx,)

    # 3D 터널 마스크: X 범위 & YZ 단면 내부
    tunnel_mask = x_in[:, np.newaxis, np.newaxis] & inside_yz[np.newaxis, :, :]

    grid_info = dict(
        origin=np.array([xmin, ymin, zmin], dtype=np.float32),
        shape=(Nx, Ny, Nz),
        voxel_size=voxel_size,
        xs=xs, ys=ys, zs=zs,
    )
    return tunnel_mask, grid_info
