"""
block_stability.py – 복셀 블록의 제거가능성(removability)과 단순 안전율(FS)

block_detector 가 확정한 블록(터널 접촉 ∩ 도메인 경계 미접촉)마다:

  1. 경계 균열면 추출
     블록 복셀의 6-이웃 중 FRACTURE 복셀의 fracture_owner 로 경계면을 찾는다.
     바깥 법선 n_out 은 블록 무게중심이 놓인 쪽의 반대 방향이다.
     면적 = (접촉 복셀면 수 × vs²) / ||n||_1   (계단형 복셀 표면의 면적 과대 보정)

  2. 이동 가능성 (voxel sweep)
     블록 표면 복셀을 방향 d 로 0.5 복셀 간격으로 이동시키며, 블록 자신의 길이(d 방향 투영폭)
     +1 복셀만큼 이동하는 동안 다른 ROCK 복셀(다른 블록·주변 암반)과 겹치지 않으면 "d 방향 자유".
     FRACTURE·TUNNEL 복셀은 통과 가능. 격자 밖은 미지 영역이므로 통과 가능으로 본다
     (불안정 쪽으로 보수적). 주변 블록은 고정으로 본다(키블록 가정).
     오목(non-convex) 블록도 그대로 판정된다.

  3. 중력 하 파괴 모드와 FS (Goodman & Shi 1985 / Hoek & Bray 한계평형, 지보·수압 없음)
     g = (0,0,-1),  W = γ·V_corr
     V_corr = V_voxel + tol·ΣA_j : 균열 slab(두께 2·tol)에 빼앗긴 블록 쪽 절반을 되돌린 부피.
     (해석해 쐐기 검증에서 V_voxel 은 O(vs) 로 과소, V_corr 는 편향이 크게 줄어듦)
       - 낙하(fall)     : -z 방향 자유           → FS = 0 (절리 인장강도 0)
       - 단일면 활동     : g·n_j > 0 인 면 j 의 경사방향 s_j 로 자유
                          FS = (W cosα tanφ + c A_j) / (W sinα)
       - 쐐기(2면) 활동  : 교선 방향 t (t·g>0) 로 자유이고 두 면 수직력 N_i, N_j > 0
                          FS = ((N_i+N_j) tanφ + c (A_i+A_j)) / (W t·g)
     허용 모드가 여럿이면 최소 FS 를 채택(보수적). 허용 모드가 없으면 mode='stable', FS=inf.

  4. removable = 표본 방향(Fibonacci 구면 + 위 모드 방향) 중 하나라도 자유이면 True.
     (위상적 제거가능성: 중력과 무관하게 터널 쪽으로 빠질 수 있는 블록)

좌표: 전역 [x, y, z] (x = 터널 굴진, z = Up). 복셀은 정육면체(vs)로 가정한다.
"""

from __future__ import annotations
import numpy as np

from dfn_analysis.block_detector import ROCK, FRACTURE

GRAVITY = np.array([0.0, 0.0, -1.0])


def fibonacci_directions(n: int) -> np.ndarray:
    """단위구 위 거의 균일한 n 개 방향."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def _expanded_slices(bbox, shape):
    i0, i1, j0, j1, k0, k1 = bbox
    return (slice(max(i0 - 1, 0), min(i1 + 1, shape[0])),
            slice(max(j0 - 1, 0), min(j1 + 1, shape[1])),
            slice(max(k0 - 1, 0), min(k1 + 1, shape[2])))


def bounding_planes(lbl, bbox, labels, state, fracture_owner,
                    normals, centers, centroid_xyz, voxel_size):
    """블록 경계 균열면 목록: [{owner, n_out, area_m2, n_faces}]."""
    sl = _expanded_slices(bbox, labels.shape)
    m = labels[sl] == lbl
    st = state[sl]
    ow = fracture_owner[sl]

    owners = []
    for axis in range(3):
        for shift in (1, -1):
            nb_state = np.roll(st, -shift, axis=axis)
            nb_owner = np.roll(ow, -shift, axis=axis)
            faces = m & (nb_state == FRACTURE)
            owners.append(nb_owner[faces])
    owners = np.concatenate(owners)
    owners = owners[owners >= 0]
    ids, counts = np.unique(owners, return_counts=True)

    planes = []
    for o, cnt in zip(ids, counts):
        n = normals[o] / np.linalg.norm(normals[o])
        side = np.sign(np.dot(centroid_xyz - centers[o], n)) or 1.0
        n_out = -side * n
        planes.append(dict(owner=int(o), n_out=n_out,
                           area_m2=float(cnt) * voxel_size ** 2 / np.abs(n).sum(),
                           n_faces=int(cnt)))
    return planes


def surface_voxels(lbl, bbox, labels):
    """블록 표면 복셀의 전역 인덱스 (n,3)."""
    sl = _expanded_slices(bbox, labels.shape)
    m = labels[sl] == lbl
    interior = m.copy()
    for axis in range(3):
        for shift in (1, -1):
            interior &= np.roll(m, shift, axis=axis)
    idx = np.argwhere(m & ~interior)
    return idx + np.array([s.start for s in sl])


def is_free(surf_idx, d, lbl, labels, state):
    """방향 d(단위벡터)로 블록 길이만큼 이동하는 동안 다른 ROCK 과 겹치지 않으면 True."""
    shape = np.array(labels.shape)
    proj = surf_idx @ d
    travel = proj.max() - proj.min() + 1.0      # 복셀 단위
    p0 = surf_idx.astype(np.float64)
    for t in np.arange(0.5, travel + 0.5, 0.5):
        q = np.rint(p0 + t * d).astype(np.int64)
        q = q[((q >= 0) & (q < shape)).all(axis=1)]      # 격자 밖 = 통과 가능
        qi, qj, qk = q[:, 0], q[:, 1], q[:, 2]
        if np.any((state[qi, qj, qk] == ROCK) & (labels[qi, qj, qk] != lbl)):
            return False
    return True


def analyze_block(block, labels, state, fracture_owner, normals, centers, grid_info,
                  gamma_kN_m3=26.0, phi_deg=30.0, cohesion_kPa=0.0, tol_factor=0.6, n_dirs=64):
    """블록 1개의 경계면·제거가능성·파괴모드·FS. 결과 dict 반환."""
    vs = float(grid_info['voxel_size'])
    lbl, bbox = block['label'], block['bbox']
    centroid = np.asarray(block['centroid'], dtype=np.float64)
    tan_phi = np.tan(np.radians(phi_deg))

    planes = bounding_planes(lbl, bbox, labels, state, fracture_owner,
                             normals, centers, centroid, vs)
    volume_corr = block['volume_m3'] + tol_factor * vs * sum(p['area_m2'] for p in planes)
    W = gamma_kN_m3 * volume_corr
    surf = surface_voxels(lbl, bbox, labels)
    free_cache = {}

    def free(d):
        key = tuple(np.round(d, 6))
        if key not in free_cache:
            free_cache[key] = is_free(surf, d, lbl, labels, state)
        return free_cache[key]

    cands = []   # (mode, FS, plane owners, direction)
    if free(GRAVITY):
        cands.append(('fall', 0.0, (), GRAVITY))

    for p in planes:
        cos_a = float(GRAVITY @ p['n_out'])
        if cos_a <= 1e-9:
            continue                                   # 중력이 면을 누르지 않음
        s = GRAVITY - cos_a * p['n_out']
        if np.linalg.norm(s) < 1e-6:
            continue                                   # 수평면: 활동 불가
        s /= np.linalg.norm(s)
        if free(s):
            sin_a = np.sqrt(max(0.0, 1.0 - cos_a ** 2))
            fs = (W * cos_a * tan_phi + cohesion_kPa * p['area_m2']) / (W * sin_a)
            cands.append(('slide_1plane', float(fs), (p['owner'],), s))

    for a in range(len(planes)):
        for b in range(a + 1, len(planes)):
            pa, pb = planes[a], planes[b]
            t = np.cross(pa['n_out'], pb['n_out'])
            if np.linalg.norm(t) < 1e-6:
                continue
            t /= np.linalg.norm(t)
            if t @ GRAVITY < 0:
                t = -t
            if t @ GRAVITY <= 1e-9:
                continue                               # 수평 교선
            g_perp = W * (GRAVITY - (GRAVITY @ t) * t)
            M = np.stack([pa['n_out'], pb['n_out']], axis=1)
            N = np.linalg.lstsq(M, g_perp, rcond=None)[0]
            if N[0] <= 0 or N[1] <= 0:
                continue                               # 한 면이 떨어짐 → 단일면/낙하 모드 영역
            if free(t):
                fs = ((N[0] + N[1]) * tan_phi
                      + cohesion_kPa * (pa['area_m2'] + pb['area_m2'])) / (W * (t @ GRAVITY))
                cands.append(('slide_2plane', float(fs), (pa['owner'], pb['owner']), t))

    if cands:
        mode, fs, mode_planes, mode_dir = min(cands, key=lambda c: c[1])
        removable = True
    else:
        mode, fs, mode_planes, mode_dir = 'stable', float('inf'), (), None
        removable = any(free(d) for d in fibonacci_directions(n_dirs))

    return dict(
        n_planes=len(planes),
        plane_owners=[p['owner'] for p in planes],
        plane_areas_m2=[round(p['area_m2'], 4) for p in planes],
        removable=bool(removable),
        mode=mode,
        fs=fs,
        mode_planes=list(mode_planes),
        mode_dir=None if mode_dir is None else [round(float(v), 4) for v in mode_dir],
        volume_corr_m3=float(volume_corr),
        weight_kN=float(W),
    )
