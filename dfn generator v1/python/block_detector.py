"""
block_detector.py
GPU 가속 블록 형성 판별 모듈

파이프라인:
1. 복셀 그리드 위에서 균열-복셀 교차 판별 (GPU 병렬)
2. 교차한 균열은 인접 복셀 간의 연결을 끊음 (face-adjacency 6-connectivity)
3. Connected component labeling → 고립된 클러스터 = 잠재적 블록
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpndi
    HAS_GPU = True
    print("[BlockDetector] CuPy GPU backend 활성화")
except ImportError:
    cp = np
    cpndi = ndi
    HAS_GPU = False
    print("[BlockDetector] CuPy 없음 – CPU 폴백")


# ────────────────────────────────────────────────────────────────────────────
#  균열-면(Face) 교차 판별
# ────────────────────────────────────────────────────────────────────────────

def _disc_intersects_plane_slab(
    centers: np.ndarray,   # (N, 3) CPU float32
    normals: np.ndarray,   # (N, 3) CPU float32
    radii:   np.ndarray,   # (N,)   CPU float32
    plane_axis: int,       # 0=X, 1=Y, 2=Z
    plane_positions: np.ndarray,  # (M,) CPU – 각 면의 위치 (m)
    voxel_size: float,
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    균열 원판과 축-정렬 평면 슬랩의 교차 여부를 GPU 배치로 계산.

    Returns
    -------
    hit : bool array shape (M, N) – hit[m, n] = True이면 균열 n이 면 m에 교차
    """
    xp = cp if HAS_GPU else np
    N = len(radii)
    M = len(plane_positions)

    # GPU 상수 (전체)
    c_gpu = xp.asarray(centers, dtype=xp.float32)  # (N,3)
    n_gpu = xp.asarray(normals, dtype=xp.float32)  # (N,3)
    r_gpu = xp.asarray(radii,   dtype=xp.float32)  # (N,)
    p_gpu = xp.asarray(plane_positions, dtype=xp.float32)  # (M,)

    # 균열 축 성분
    c_ax = c_gpu[:, plane_axis]   # (N,)
    n_ax = n_gpu[:, plane_axis]   # (N,)

    hit = xp.zeros((M, N), dtype=xp.bool_)

    # ── 조건 1: AABB 중심 거리 ────────────────────────────────────────────
    # |center_ax - plane| <= radius
    # p_gpu (M,) - c_ax (N,) → (M, N) broadcast
    dist_to_plane = xp.abs(p_gpu[:, None] - c_ax[None, :])  # (M, N)
    within_radius = dist_to_plane <= r_gpu[None, :]           # (M, N)

    # ── 조건 2: 실제 원판 반경 투영 ──────────────────────────────────────
    # 원판의 법선이 평면에 거의 평행하면 (n_ax ≈ 0) 교차 없음
    # 원판과 평면의 교차원: r_eff = r * sqrt(1 - n_ax^2)
    #   단 n_ax가 크면 (법선이 평면 법선과 거의 평행) 작아짐
    #   디스크 평면의 평면과의 거리 = |dot(plane_pos - center, n)| / 1
    #   실제 교차 조건: 거리 <= r * sqrt(1 - n_ax²) (원판 높이, 근사)
    n_ax_clamp = xp.clip(xp.abs(n_ax), 0, 1)           # (N,)
    r_proj = r_gpu * xp.sqrt(xp.maximum(1 - n_ax_clamp**2, 0))  # (N,) – 투영 반경

    # 평면과 원판 중심 거리를 법선 방향으로
    # (M, N): |(p - c_ax) * n_ax_signed| → 실제 교차 두께
    signed_n = n_gpu[:, plane_axis]  # (N,)
    # 원판이 평면을 실제로 통과하는 조건:
    # |c_ax - p| <= |r| * |n_ax| 이면 원판이 평면을 통과
    #  (디스크가 축에 수직인 경우 n_ax=1이면 원점 근접만으로 충분)
    pierces = dist_to_plane <= (r_gpu[None, :] * xp.abs(signed_n)[None, :] + voxel_size * 0.5)

    hit = within_radius | pierces  # 둘 중 하나라도 해당하면 교차 후보

    if HAS_GPU:
        return cp.asnumpy(hit)
    return hit


# ────────────────────────────────────────────────────────────────────────────
#  주방향 연결 행렬 세버기 (Face Sever)
# ────────────────────────────────────────────────────────────────────────────

def build_face_adjacency(
    grid_shape: tuple,          # (Nx, Ny, Nz)
    fracture_centers: np.ndarray,  # (N,3) float32
    fracture_normals: np.ndarray,  # (N,3) float32
    fracture_radii:   np.ndarray,  # (N,) float32
    grid_info: dict,
    halo_mask_cpu: np.ndarray,  # (Nx,Ny,Nz) bool – 분석 대상 복셀
    batch_fractures: int = 200_000,
) -> tuple:
    """
    각 축(X, Y, Z)에 대해 인접 복셀 쌍 간의 face가 균열에 의해
    끊겨 있는지 계산합니다.

    Returns
    -------
    conn_x, conn_y, conn_z : (Nx-1,Ny,Nz), (Nx,Ny-1,Nz), (Nx,Ny,Nz-1) bool
        True = 해당 face가 연결되어 있음 (균열 없음)
    """
    Nx, Ny, Nz = grid_shape
    vs = grid_info['voxel_size']
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']

    xp = cp if HAS_GPU else np

    # 각 면의 위치: 복셀 경계 (복셀 중심 사이 중간)
    face_x = (xs[:-1] + xs[1:]) / 2   # (Nx-1,)
    face_y = (ys[:-1] + ys[1:]) / 2   # (Ny-1,)
    face_z = (zs[:-1] + zs[1:]) / 2   # (Nz-1,)

    # 초기값: 모두 연결됨
    conn_x = np.ones((Nx-1, Ny, Nz), dtype=bool)
    conn_y = np.ones((Nx, Ny-1, Nz), dtype=bool)
    conn_z = np.ones((Nx, Ny, Nz-1), dtype=bool)

    N_frac = len(fracture_radii)
    n_batches = int(np.ceil(N_frac / batch_fractures))

    for axis, face_positions, conn in [
        (0, face_x, conn_x),
        (1, face_y, conn_y),
        (2, face_z, conn_z),
    ]:
        axis_names = ['X', 'Y', 'Z']
        print(f"  [{axis_names[axis]}축] 균열-면 교차 계산 ({len(face_positions)} faces × {N_frac:,} 균열)...")

        # 배치로 균열 처리
        sever_any = np.zeros((len(face_positions),), dtype=bool)

        for bi in tqdm(range(n_batches), desc=f"  Batch ({axis_names[axis]})", leave=False):
            sl = slice(bi * batch_fractures, (bi + 1) * batch_fractures)
            c_b = fracture_centers[sl]
            n_b = fracture_normals[sl]
            r_b = fracture_radii[sl]
            if len(r_b) == 0:
                continue
            hit = _disc_intersects_plane_slab(c_b, n_b, r_b, axis, face_positions, vs)
            # hit: (M, n_b) → 어느 균열이든 교차하면 face는 절단됨
            sever_any |= hit.any(axis=1)  # (M,)

        # conn 배열에 적용
        # axis=0: conn_x shape (Nx-1, Ny, Nz), face_positions shape (Nx-1,)
        #   sever_any[m] → conn_x[m, :, :] = False
        if axis == 0:
            conn_x[sever_any, :, :] = False
        elif axis == 1:
            conn_y[:, sever_any, :] = False
        else:
            conn_z[:, :, sever_any] = False

    return conn_x, conn_y, conn_z


# ────────────────────────────────────────────────────────────────────────────
#  연결성 분석 → 블록 레이블링
# ────────────────────────────────────────────────────────────────────────────

def detect_blocks(
    conn_x: np.ndarray,   # (Nx-1, Ny, Nz)
    conn_y: np.ndarray,   # (Nx, Ny-1, Nz)
    conn_z: np.ndarray,   # (Nx, Ny, Nz-1)
    halo_mask: np.ndarray,  # (Nx, Ny, Nz) bool
    grid_info: dict,
    min_voxels: int = 8,
) -> tuple:
    """
    Face 연결 정보로부터 그래프 기반 connected component labeling을 수행.

    Returns
    -------
    labels    : (Nx, Ny, Nz) int32 – 각 복셀의 블록 번호 (0 = 배경)
    block_info: list of dict – 각 블록의 통계
    """
    Nx, Ny, Nz = halo_mask.shape
    vs = grid_info['voxel_size']
    voxel_volume = vs ** 3  # m³

    print("  Connected component labeling (Union-Find)...")

    # ── Union-Find (CPU) ─────────────────────────────────────────────────
    # halo 영역 내 복셀만 분석 대상
    flat = halo_mask.ravel()        # (Nx*Ny*Nz,) bool
    n_total = Nx * Ny * Nz
    parent = np.arange(n_total, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # X 방향 연결 (stride = Ny*Nz)
    stride_x = Ny * Nz
    ix_range = np.where(conn_x.ravel())[0]  # conn_x: (Nx-1, Ny, Nz) flatten
    for f in tqdm(ix_range, desc="  Union X", leave=False, miniters=1000):
        # f 번째 face → voxel ix(f//(Ny*Nz)), iy((f%(Ny*Nz))//Nz), iz(f%Nz)
        ix = f // (Ny * Nz)
        rest = f % (Ny * Nz)
        iy = rest // Nz
        iz = rest % Nz
        a = ix * Ny * Nz + iy * Nz + iz
        b = a + stride_x
        if flat[a] and flat[b]:
            union(a, b)

    # Y 방향 연결 (stride = Nz)
    stride_y = Nz
    iy_range = np.where(conn_y.ravel())[0]  # conn_y: (Nx, Ny-1, Nz)
    for f in tqdm(iy_range, desc="  Union Y", leave=False, miniters=1000):
        ix = f // ((Ny - 1) * Nz)
        rest = f % ((Ny - 1) * Nz)
        iy = rest // Nz
        iz = rest % Nz
        a = ix * Ny * Nz + iy * Nz + iz
        b = a + stride_y
        if flat[a] and flat[b]:
            union(a, b)

    # Z 방향 연결 (stride = 1)
    iz_range = np.where(conn_z.ravel())[0]  # conn_z: (Nx, Ny, Nz-1)
    for f in tqdm(iz_range, desc="  Union Z", leave=False, miniters=1000):
        ix = f // (Ny * (Nz - 1))
        rest = f % (Ny * (Nz - 1))
        iy = rest // (Nz - 1)
        iz = rest % (Nz - 1)
        a = ix * Ny * Nz + iy * Nz + iz
        b = a + 1
        if flat[a] and flat[b]:
            union(a, b)

    # 루트 정규화
    for i in range(n_total):
        parent[i] = find(i)

    # halo 내부 복셀만 레이블링
    labels_flat = np.zeros(n_total, dtype=np.int32)
    halo_idx = np.where(flat)[0]
    roots = parent[halo_idx]
    unique_roots, inv = np.unique(roots, return_inverse=True)
    labels_flat[halo_idx] = inv + 1  # 1부터 시작
    labels = labels_flat.reshape(Nx, Ny, Nz)

    # ── 블록 통계 ─────────────────────────────────────────────────────────
    print("  블록 통계 계산...")
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    n_labels = len(unique_roots)
    block_info = []
    for lbl in range(1, n_labels + 1):
        mask = labels == lbl
        cnt = mask.sum()
        if cnt < min_voxels:
            labels[mask] = 0
            continue
        vol = cnt * voxel_volume
        cx = XX[mask].mean()
        cy = YY[mask].mean()
        cz = ZZ[mask].mean()
        block_info.append(dict(
            label=lbl,
            n_voxels=int(cnt),
            volume_m3=float(vol),
            centroid=(float(cx), float(cy), float(cz)),
        ))

    block_info.sort(key=lambda d: d['volume_m3'], reverse=True)
    print(f"  → 검출된 블록: {len(block_info)}개 (최소 {min_voxels} 복셀 기준)")
    return labels, block_info
