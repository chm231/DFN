"""
tunnel_geometry.py
터널 형상 → 3D 복셀 마스크 생성 (GPU)

MATLAB에서 내보낸 HDF5의 터널 단면 폴리곤을 읽어
- tunnel_mask  : 터널 내부 복셀 (excavated)
- halo_mask    : 터널 외벽 기준 halo_dist m 이내 복셀 (분석 영역)
을 GPU 배열로 반환합니다.
"""

from __future__ import annotations
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = np
    HAS_GPU = False


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
    halo_dist: float = 6.0,
    tunnel_xmin: float | None = None,
    tunnel_xmax: float | None = None,
) -> tuple:
    """
    Parameters
    ----------
    poly_Y, poly_Z : 터널 단면 폴리곤 좌표 (m)
    domain_box     : [xmin, xmax, ymin, ymax, zmin, zmax]
    voxel_size     : 복셀 한 변 길이 (m)
    halo_dist      : 터널 경계로부터 분석 영역 거리 (m)
    tunnel_xmin/xmax : 터널 X 방향 범위 (None이면 도메인 전체)

    Returns
    -------
    voxel_centers : (Nx,Ny,Nz,3) float32 GPU 배열 – 복셀 중심 좌표
    tunnel_mask   : (Nx,Ny,Nz) bool GPU 배열 – 터널 내부
    halo_mask     : (Nx,Ny,Nz) bool GPU 배열 – halo 영역 (터널 제외)
    grid_info     : dict (원점, 복셀 수, 복셀 크기)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = domain_box.astype(float)

    # 그리드 생성
    xs = np.arange(xmin + voxel_size / 2, xmax, voxel_size, dtype=np.float32)
    ys = np.arange(ymin + voxel_size / 2, ymax, voxel_size, dtype=np.float32)
    zs = np.arange(zmin + voxel_size / 2, zmax, voxel_size, dtype=np.float32)

    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    print(f"  Grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} voxels  (voxel={voxel_size}m)")

    # YZ 평면 격자 – 터널 단면 마스크
    YY, ZZ = np.meshgrid(ys, zs, indexing='ij')  # (Ny, Nz)
    py_flat = YY.ravel()
    pz_flat = ZZ.ravel()

    inside_yz = point_in_polygon_2d(py_flat, pz_flat, poly_Y, poly_Z)
    inside_yz = inside_yz.reshape(Ny, Nz)  # (Ny, Nz)

    # 터널 X 범위 마스크
    if tunnel_xmin is None:
        tunnel_xmin = xmin
    if tunnel_xmax is None:
        tunnel_xmax = xmax
    x_in = (xs >= tunnel_xmin) & (xs <= tunnel_xmax)  # (Nx,)

    # 3D 터널 마스크: X 범위 & YZ 단면 내부
    # tunnel_mask[ix, iy, iz] = x_in[ix] & inside_yz[iy, iz]
    tunnel_mask_np = x_in[:, np.newaxis, np.newaxis] & inside_yz[np.newaxis, :, :]  # (Nx,Ny,Nz)

    # === Halo 마스크: 터널 경계로부터 halo_dist m 이내 (터널 외부) ===
    # YZ 평면에서 각 외부 점과 폴리곤 경계까지의 최소 거리
    dist_yz = _signed_dist_to_polygon_yz(py_flat, pz_flat, poly_Y, poly_Z, inside_yz.ravel())
    dist_yz = dist_yz.reshape(Ny, Nz)  # 양수=외부, 음수=내부

    halo_yz = (dist_yz > 0) & (dist_yz <= halo_dist)  # 외부이면서 halo 이내
    halo_mask_np = x_in[:, np.newaxis, np.newaxis] & halo_yz[np.newaxis, :, :]

    # GPU로 전송
    xp = cp if HAS_GPU else np
    tunnel_mask = xp.asarray(tunnel_mask_np)
    halo_mask   = xp.asarray(halo_mask_np)

    # 복셀 중심 좌표 배열 (Nx,Ny,Nz,3)
    XX3, YY3, ZZ3 = np.meshgrid(xs, ys, zs, indexing='ij')
    voxel_centers = xp.stack([
        xp.asarray(XX3.astype(np.float32)),
        xp.asarray(YY3.astype(np.float32)),
        xp.asarray(ZZ3.astype(np.float32)),
    ], axis=-1)

    grid_info = dict(
        origin=np.array([xmin, ymin, zmin], dtype=np.float32),
        shape=(Nx, Ny, Nz),
        voxel_size=voxel_size,
        xs=xs, ys=ys, zs=zs,
    )

    return voxel_centers, tunnel_mask, halo_mask, grid_info


def _signed_dist_to_polygon_yz(py, pz, poly_y, poly_z, inside_flag):
    """각 점에서 폴리곤 경계까지 최소 거리 (외부=양수, 내부=음수)."""
    n = len(poly_y)
    min_dist = np.full(len(py), np.inf, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        ay, az = poly_y[i], poly_z[i]
        by, bz = poly_y[j], poly_z[j]
        edy, edz = by - ay, bz - az
        len2 = edy**2 + edz**2
        if len2 < 1e-15:
            continue
        t = np.clip(((py - ay) * edy + (pz - az) * edz) / len2, 0, 1)
        cy = ay + t * edy
        cz = az + t * edz
        d = np.sqrt((py - cy)**2 + (pz - cz)**2)
        min_dist = np.minimum(min_dist, d)

    signed = np.where(inside_flag, -min_dist, min_dist)
    return signed.astype(np.float32)
