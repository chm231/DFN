"""
block_detector.py  –  Voxel-based GPU DFN Block Detection

Algorithm (per user specification):
  1. Voxelization: classify each voxel as ROCK / FRACTURE / TUNNEL
  2. CCA (26-connectivity) on ROCK voxels only
  3. Filter:
       Cond A – component (dilated 1 voxel) touches TUNNEL
       Cond B – component does NOT touch grid boundary
  4. Post-process: volume, centroid, contact area
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    import cupy as cp
    HAS_GPU = True
    print("[BlockDetector] CuPy GPU backend 활성화")
except ImportError:
    cp = np
    HAS_GPU = False
    print("[BlockDetector] CuPy 없음 - CPU 폴백")

# ── Voxel state labels ────────────────────────────────────────────────────
ROCK     = np.uint8(0)
FRACTURE = np.uint8(1)
TUNNEL   = np.uint8(2)

# 26-connectivity structuring element
STRUCT26 = np.ones((3, 3, 3), dtype=bool)
STRUCT6  = ndi.generate_binary_structure(3, 1)

# AABB size threshold: fractures with local grid > this use GPU
_GPU_AABB_THRESH = 32_768   # 32³


def to_numpy(x) -> np.ndarray:
    if HAS_GPU:
        return getattr(cp, 'asnumpy')(x)
    return np.asarray(x)


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 – Voxel Classification
# ══════════════════════════════════════════════════════════════════════════

def classify_voxels(
    grid_info: dict,
    fracture_centers: np.ndarray,   # (N, 3) float32
    fracture_normals: np.ndarray,   # (N, 3) float32
    fracture_radii:   np.ndarray,   # (N,)   float32
    tunnel_mask:      np.ndarray,   # (Nx,Ny,Nz) bool  CPU
    tol_factor: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (state, fracture_owner):
      state: (Nx,Ny,Nz) uint8 array (ROCK=0, FRACTURE=1, TUNNEL=2)
      fracture_owner: (Nx,Ny,Nz) int32 array (Fracture ID or -1)
    Priority: TUNNEL > FRACTURE > ROCK
    For FRACTURE voxels, owner is assigned to the fracture with smallest distance to plane.
    """
    Nx, Ny, Nz = grid_info['shape']
    vs   = float(grid_info['voxel_size'])
    xs   = grid_info['xs']
    ys   = grid_info['ys']
    zs   = grid_info['zs']
    tol  = vs * tol_factor          # fracture half-thickness [m]

    state = np.zeros((Nx, Ny, Nz), dtype=np.uint8)   # all ROCK
    fracture_owner = np.full((Nx, Ny, Nz), -1, dtype=np.int32)
    min_dist = np.full((Nx, Ny, Nz), np.inf, dtype=np.float32)
    
    state[tunnel_mask.astype(bool)] = TUNNEL

    N = len(fracture_radii)
    print(f"  균열 복셀화 시작: {N:,}개 (tol={tol:.3f}m)")

    for i in tqdm(range(N), desc="  균열 복셀화", miniters=max(1, N // 200)):
        cx = float(fracture_centers[i, 0])
        cy = float(fracture_centers[i, 1])
        cz = float(fracture_centers[i, 2])
        nx_ = float(fracture_normals[i, 0])
        ny_ = float(fracture_normals[i, 1])
        nz_ = float(fracture_normals[i, 2])
        r   = float(fracture_radii[i])

        # ── AABB in voxel-index space ──────────────────────────────────
        ix0 = max(0,  int(np.searchsorted(xs, cx - r - vs)))
        ix1 = min(Nx, int(np.searchsorted(xs, cx + r + vs)) + 1)
        iy0 = max(0,  int(np.searchsorted(ys, cy - r - vs)))
        iy1 = min(Ny, int(np.searchsorted(ys, cy + r + vs)) + 1)
        iz0 = max(0,  int(np.searchsorted(zs, cz - r - vs)))
        iz1 = min(Nz, int(np.searchsorted(zs, cz + r + vs)) + 1)

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue

        lNx, lNy, lNz = ix1-ix0, iy1-iy0, iz1-iz0
        aabb_size = lNx * lNy * lNz

        lx = xs[ix0:ix1]
        ly = ys[iy0:iy1]
        lz = zs[iz0:iz1]

        if HAS_GPU and aabb_size >= _GPU_AABB_THRESH:
            # ── GPU path ───────────────────────────────────────────────
            LX, LY, LZ = cp.meshgrid(
                cp.asarray(lx), cp.asarray(ly), cp.asarray(lz), indexing='ij')
            dx = LX - cp.float32(cx)
            dy = LY - cp.float32(cy)
            dz = LZ - cp.float32(cz)
            d_plane = dx*cp.float32(nx_) + dy*cp.float32(ny_) + dz*cp.float32(nz_)
            d_rad_sq = dx**2 + dy**2 + dz**2 - d_plane**2
            hit = (cp.abs(d_plane) <= cp.float32(tol)) & \
                  (d_rad_sq <= cp.float32(r * r))
            hit_cpu = to_numpy(hit)
        else:
            # ── CPU path ───────────────────────────────────────────────
            LX, LY, LZ = np.meshgrid(lx, ly, lz, indexing='ij')
            dx = LX - cx;  dy = LY - cy;  dz = LZ - cz
            d_plane  = dx*nx_ + dy*ny_ + dz*nz_
            d_rad_sq = dx**2 + dy**2 + dz**2 - d_plane**2
            hit_cpu  = (np.abs(d_plane) <= tol) & (d_rad_sq <= r*r)

        local = state[ix0:ix1, iy0:iy1, iz0:iz1]
        local_owner = fracture_owner[ix0:ix1, iy0:iy1, iz0:iz1]
        local_min_dist = min_dist[ix0:ix1, iy0:iy1, iz0:iz1]

        # smallest distance logic
        dist_abs = np.abs(d_plane) if not (HAS_GPU and aabb_size >= _GPU_AABB_THRESH) else to_numpy(cp.abs(d_plane))
        
        # update if hit AND (dist < min_dist)
        update_mask = hit_cpu & (dist_abs < local_min_dist)
        local[update_mask & (local == ROCK)] = FRACTURE
        local_owner[update_mask] = i
        local_min_dist[update_mask] = dist_abs[update_mask]

    n_rock = int((state == ROCK).sum())
    n_frac = int((state == FRACTURE).sum())
    n_tunn = int((state == TUNNEL).sum())
    print(f"  분류 결과:  ROCK={n_rock:,}  FRACTURE={n_frac:,}  TUNNEL={n_tunn:,}")
    return state, fracture_owner


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 – Connected Component Analysis (26-connectivity)
# ══════════════════════════════════════════════════════════════════════════

def run_cca(state: np.ndarray, connectivity: int = 26) -> tuple:
    """
    CCA on ROCK voxels with 6 or 26 connectivity.
    Tries GPU (cupyx.scipy.ndimage.label), falls back to CPU scipy.
    Returns: labels (Nx,Ny,Nz) int32,  n_labels int
    """
    struct = STRUCT26 if connectivity == 26 else STRUCT6
    rock_mask = (state == ROCK)
    print(f"  CCA 시작 (ROCK 복셀: {rock_mask.sum():,}개, {connectivity}-connectivity)...")

    try:
        if not HAS_GPU:
            raise RuntimeError("no GPU")
        from cupyx.scipy import ndimage as cpndi
        rock_gpu   = cp.asarray(rock_mask)
        struct_gpu = cp.asarray(struct)
        from typing import Any
        res: Any = cpndi.label(rock_gpu, structure=struct_gpu)
        labels_gpu = res[0]
        n_labels = res[1]
        labels = to_numpy(labels_gpu).astype(np.int32)
        print(f"  GPU CCA 완료: {n_labels:,} 컴포넌트")
    except Exception:
        labels, n_labels = ndi.label(rock_mask.astype(np.int8), structure=struct)
        labels = labels.astype(np.int32)
        print(f"  CPU CCA 완료: {n_labels:,} 컴포넌트")

    return labels, int(n_labels)


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 – Filter & Statistics
# ══════════════════════════════════════════════════════════════════════════

def filter_and_stat_blocks(
    labels:      np.ndarray,   # (Nx,Ny,Nz) int32
    n_labels:    int,
    state:       np.ndarray,   # (Nx,Ny,Nz) uint8  – full state array
    grid_info:   dict,
    min_voxels:  int = 8,
    connectivity:int = 26,     # Dilation 구조체
) -> list:
    """
    Conditions:
      A – dilated component overlaps TUNNEL voxels (vectorized via tunnel mask dilation)
      B – component does NOT touch any grid boundary face (vectorized via boundary mask)

    Returns list of dicts, sorted by volume descending.
    (Vectorized approach for extreme speed up)
    """
    Nx, Ny, Nz   = grid_info['shape']
    vs           = float(grid_info['voxel_size'])
    voxel_vol    = vs ** 3
    xs, ys, zs   = grid_info['xs'], grid_info['ys'], grid_info['zs']

    tunnel_bool  = (state == TUNNEL)
    
    print(f"  블록 필터링 병목 개선 파이프라인 시작 (전체 {n_labels:,} 컴포넌트)...")

    # ── 1. Voxel Count 계산 및 소형 제거 ────────────────────────────
    try:
        if HAS_GPU:
            counts_gpu = cp.bincount(cp.asarray(labels).ravel())
            voxel_counts = to_numpy(counts_gpu)
        else:
            voxel_counts = np.bincount(labels.ravel())
    except Exception:
        voxel_counts = np.bincount(labels.ravel())

    if len(voxel_counts) <= n_labels:
        voxel_counts = np.pad(voxel_counts, (0, n_labels + 1 - len(voxel_counts)))

    surviving_labels = np.where(voxel_counts >= min_voxels)[0]
    surviving_labels = surviving_labels[surviving_labels > 0]
    print(f"    - 1단계: min_voxels(>={min_voxels}) 필터 통과 컴포넌트: {len(surviving_labels):,} 개")

    # ── 2. Boundary Touch 벡터화 판정 ──────────────────────────────
    boundary_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    boundary_mask[0,:,:] = True
    boundary_mask[-1,:,:] = True
    boundary_mask[:,0,:] = True
    boundary_mask[:,-1,:] = True
    boundary_mask[:,:,0] = True
    boundary_mask[:,:,-1] = True
    
    boundary_labels = np.unique(labels[boundary_mask])
    boundary_labels = boundary_labels[boundary_labels > 0]
    print(f"    - 2단계: boundary-touch labels 수: {len(boundary_labels):,} 개")

    # ── 3. Tunnel Touch 벡터화 판정 (1회 Dilation) ─────────────────
    struct = STRUCT26 if connectivity == 26 else STRUCT6
    try:
        if HAS_GPU:
            from cupyx.scipy import ndimage as cpndi
            tmask_gpu = cp.asarray(tunnel_bool)
            struct_gpu = cp.asarray(struct)
            dilated_tmask_gpu = cpndi.binary_dilation(tmask_gpu, structure=struct_gpu)
            dilated_tunnel_mask = to_numpy(dilated_tmask_gpu)
        else:
            dilated_tunnel_mask = ndi.binary_dilation(tunnel_bool, structure=struct)
    except Exception:
        dilated_tunnel_mask = ndi.binary_dilation(tunnel_bool, structure=struct)

    tunnel_touch_labels = np.unique(labels[dilated_tunnel_mask])
    tunnel_touch_labels = tunnel_touch_labels[tunnel_touch_labels > 0]
    print(f"    - 3단계: tunnel-touch labels 수: {len(tunnel_touch_labels):,} 개")

    # ── 4. 최종 유효 라벨 교집합 필터링 ────────────────────────────
    surviving_set = set(surviving_labels)
    boundary_set = set(boundary_labels)
    tunnel_set = set(tunnel_touch_labels)
    
    final_labels = surviving_set - boundary_set
    final_labels = final_labels.intersection(tunnel_set)
    final_labels = list(final_labels)
    print(f"    - 4단계: 최종 통과 labels (surviving - boundary & tunnel): {len(final_labels):,} 개")

    # ── 5. 통계 계산 (최종 라벨 대상만 for loop) ───────────────────
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    block_info = []

    for lbl in tqdm(final_labels, desc="  최종 통계 산출", miniters=max(1, len(final_labels)//10)):
        mask = (labels == lbl)
        cnt = int(voxel_counts[lbl])
        
        vol_ = cnt * voxel_vol
        cx_ = float(XX[mask].mean())
        cy_ = float(YY[mask].mean())
        cz_ = float(ZZ[mask].mean())
        
        # symmetric dilation에 의해 mask & dilated_tunnel_mask 교집합이 접촉 Voxel 수
        contact_voxels = int(np.sum(mask & dilated_tunnel_mask))
        contact_area   = contact_voxels * (vs ** 2)
        
        block_info.append(dict(
            label          = int(lbl),
            n_voxels       = cnt,
            volume_m3      = vol_,
            centroid       = (cx_, cy_, cz_),
            contact_area_m2= contact_area,
        ))

    block_info.sort(key=lambda d: d['volume_m3'], reverse=True)
    for rank, b in enumerate(block_info):
        b['rank'] = rank + 1

    rejected_small = n_labels - len(surviving_labels)
    rejected_boundary = len(surviving_set.intersection(boundary_set))
    rejected_no_tunnel = len(surviving_set - boundary_set) - len(final_labels)

    print(f"  필터 결과: {len(block_info)}개 블록 확정")
    print(f"    제외 - 소형:{rejected_small}  경계접촉:{rejected_boundary}  터널미접촉:{rejected_no_tunnel}")
    return block_info
