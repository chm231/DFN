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
    print("[BlockDetector] CuPy 없음 – CPU 폴백")

# ── Voxel state labels ────────────────────────────────────────────────────
ROCK     = np.uint8(0)
FRACTURE = np.uint8(1)
TUNNEL   = np.uint8(2)

# 26-connectivity structuring element
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

# AABB size threshold: fractures with local grid > this use GPU
_GPU_AABB_THRESH = 32_768   # 32³


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
) -> np.ndarray:
    """
    Returns (Nx,Ny,Nz) uint8 array:
      ROCK=0,  FRACTURE=1,  TUNNEL=2
    Priority:  TUNNEL > FRACTURE > ROCK
    """
    Nx, Ny, Nz = grid_info['shape']
    vs   = float(grid_info['voxel_size'])
    xs   = grid_info['xs']
    ys   = grid_info['ys']
    zs   = grid_info['zs']
    tol  = vs * tol_factor          # fracture half-thickness [m]

    state = np.zeros((Nx, Ny, Nz), dtype=np.uint8)   # all ROCK
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
            hit_cpu = cp.asnumpy(hit)
        else:
            # ── CPU path ───────────────────────────────────────────────
            LX, LY, LZ = np.meshgrid(lx, ly, lz, indexing='ij')
            dx = LX - cx;  dy = LY - cy;  dz = LZ - cz
            d_plane  = dx*nx_ + dy*ny_ + dz*nz_
            d_rad_sq = dx**2 + dy**2 + dz**2 - d_plane**2
            hit_cpu  = (np.abs(d_plane) <= tol) & (d_rad_sq <= r*r)

        local = state[ix0:ix1, iy0:iy1, iz0:iz1]
        local[hit_cpu & (local == ROCK)] = FRACTURE

    n_rock = int((state == ROCK).sum())
    n_frac = int((state == FRACTURE).sum())
    n_tunn = int((state == TUNNEL).sum())
    print(f"  분류 결과:  ROCK={n_rock:,}  FRACTURE={n_frac:,}  TUNNEL={n_tunn:,}")
    return state


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 – Connected Component Analysis (26-connectivity)
# ══════════════════════════════════════════════════════════════════════════

def run_cca(state: np.ndarray) -> tuple:
    """
    26-connectivity CCA on ROCK voxels.
    Tries GPU (cupyx.scipy.ndimage.label), falls back to CPU scipy.
    Returns: labels (Nx,Ny,Nz) int32,  n_labels int
    """
    rock_mask = (state == ROCK)
    print(f"  CCA 시작 (ROCK 복셀: {rock_mask.sum():,}개, 26-connectivity)...")

    try:
        if not HAS_GPU:
            raise RuntimeError("no GPU")
        from cupyx.scipy import ndimage as cpndi
        rock_gpu  = cp.asarray(rock_mask)
        struct_gpu = cp.asarray(STRUCT26)
        labels_gpu, n_labels = cpndi.label(rock_gpu, structure=struct_gpu)
        labels = cp.asnumpy(labels_gpu).astype(np.int32)
        print(f"  GPU CCA 완료: {n_labels:,} 컴포넌트")
    except Exception:
        labels, n_labels = ndi.label(rock_mask.astype(np.int8), structure=STRUCT26)
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
) -> list:
    """
    Conditions:
      A – dilated component overlaps TUNNEL voxels
      B – component does NOT touch any grid boundary face

    Returns list of dicts, sorted by volume descending.
    """
    Nx, Ny, Nz   = grid_info['shape']
    vs           = float(grid_info['voxel_size'])
    voxel_vol    = vs ** 3
    xs, ys, zs   = grid_info['xs'], grid_info['ys'], grid_info['zs']

    tunnel_bool  = (state == TUNNEL)
    XX, YY, ZZ   = np.meshgrid(xs, ys, zs, indexing='ij')

    block_info = []
    rejected_small = rejected_boundary = rejected_no_tunnel = 0

    print(f"  블록 필터링 ({n_labels:,} 컴포넌트)...")
    for lbl in tqdm(range(1, n_labels + 1),
                    desc="  필터링", miniters=max(1, n_labels // 100)):

        mask = (labels == lbl)
        cnt  = int(mask.sum())

        # ── 최소 크기 ───────────────────────────────────────────────────
        if cnt < min_voxels:
            rejected_small += 1
            continue

        # ── Condition B: 경계 접촉 없어야 함 ────────────────────────────
        if (mask[0,:,:].any() or mask[-1,:,:].any() or
            mask[:,0,:].any() or mask[:,-1,:].any() or
            mask[:,:,0].any() or mask[:,:,-1].any()):
            rejected_boundary += 1
            continue

        # ── Condition A: 터널과 접촉해야 함 (1-voxel dilation) ──────────
        dilated = ndi.binary_dilation(mask, structure=STRUCT26)
        if not bool(np.any(dilated & tunnel_bool)):
            rejected_no_tunnel += 1
            continue

        # ── 통계 계산 ───────────────────────────────────────────────────
        vol             = cnt * voxel_vol
        cx_             = float(XX[mask].mean())
        cy_             = float(YY[mask].mean())
        cz_             = float(ZZ[mask].mean())
        contact_voxels  = int(np.sum(dilated & tunnel_bool))
        contact_area    = contact_voxels * (vs ** 2)

        block_info.append(dict(
            label          = int(lbl),
            n_voxels       = cnt,
            volume_m3      = float(vol),
            centroid       = (cx_, cy_, cz_),
            contact_area_m2= float(contact_area),
        ))

    block_info.sort(key=lambda d: d['volume_m3'], reverse=True)
    for rank, b in enumerate(block_info):
        b['rank'] = rank + 1

    print(f"  필터 결과: {len(block_info)}개 블록 확정")
    print(f"    제외 – 소형:{rejected_small}  경계접촉:{rejected_boundary}"
          f"  터널미접촉:{rejected_no_tunnel}")
    return block_info
