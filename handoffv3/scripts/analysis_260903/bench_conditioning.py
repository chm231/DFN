# -*- coding: utf-8 -*-
"""부재 조건화(remove_face_intersecting) 최적화 방안 벤치마크.

원본(파이썬 이중 루프) vs 벡터화(기존 clip_segments_to_convex_polygon_vectorized 재사용)를
같은 실현(seed 7)에서 비교: ① 부분표본 5만 개에서 판정 완전 일치 검증
② 전체 121만 개에서 소요 시간 비교.
"""
import time

import h5py
import numpy as np

from dfn_analysis import generate_conditional_hidden_dfn as G
from dfn_analysis.export_domain_dfn_json import yz_bounds
from dfn_analysis.radius_powerlaw_likelihood import clip_segments_to_convex_polygon_vectorized

PDIR = "demo_output/dfm_demo"
visible = G.load_visible_discs(f"{PDIR}/reconstruct/reconstructed_discs.csv",
                               ("deterministic_disc", "orientation_only"))
observed, _ = G.load_observed_traces(f"{PDIR}/trace_dataset/trace_dataset_3d.csv")
params = G.load_inverted_params(f"{PDIR}/kr/kr_summary_by_set.csv", f"{PDIR}/p32/p32_summary.csv")
face_xs = sorted(observed.keys())
with h5py.File(f"{PDIR}/dfn_export_for_python.h5") as f:
    poly = np.array(f["tunnel/poly_YZ"], dtype=np.float64)
poly_ccw = G._ccw_polygon(poly)

x0, x1 = max(face_xs), max(face_xs) + 10.0
RMAX = 25.0
ymin, ymax, zmin, zmax = yz_bounds(poly_ccw, 5.0, 7.48)
box = dict(x0=x0 - RMAX, dx=(x1 - x0) + 2 * RMAX,
           y0=ymin - RMAX, dy=(ymax - ymin) + 2 * RMAX,
           z0=zmin - RMAX, dz=(zmax - zmin) + 2 * RMAX)

t = time.perf_counter()
hidden = G.generate_hidden_discs(params, visible, box, RMAX, 7, [1, 2, 3, 4])
print(f"[gen] {len(hidden):,}개 생성 ({time.perf_counter()-t:.1f} s)")

C = np.array([d["center"] for d in hidden])
N = np.array([d["normal"] for d in hidden])
N = N / np.linalg.norm(N, axis=1, keepdims=True)
R = np.array([d["radius"] for d in hidden])
LMIN = 0.5
TOL = 1e-9


def detected_vectorized(C, N, R):
    """면별 벡터화: disc-면 현 계산 + 볼록다각형 클리핑 후 길이 >= LMIN 판정."""
    det = np.zeros(len(R), dtype=bool)
    sin_phi = np.sqrt(np.maximum(1.0 - N[:, 0] ** 2, 0.0))
    ok_phi = sin_phi >= TOL
    for xf in face_xs:
        t_in = np.where(ok_phi, (C[:, 0] - xf) / np.where(ok_phi, sin_phi, 1.0), np.inf)
        cand = ~det & ok_phi & (np.abs(t_in) < R - TOL)
        # (현 전체 길이 < LMIN 이면 클리핑 후에도 < LMIN — 저렴한 사전 배제)
        half = np.sqrt(np.maximum(R[cand] ** 2 - t_in[cand] ** 2, 0.0))
        long_enough = 2.0 * half >= LMIN
        idx = np.nonzero(cand)[0][long_enough]
        if len(idx) == 0:
            continue
        n_s, c_s, r_s = N[idx], C[idx], R[idx]
        sp = sin_phi[idx]
        ti = t_in[idx]
        hf = half[long_enough]
        # 현 방향 = n × e_x (yz 성분만 필요; x성분은 0)
        dir_yz = np.column_stack([n_s[:, 2], -n_s[:, 1]]) / sp[:, None]
        # 현 중점 = c − t_in · (e_x − n_x n)/sinφ
        proj = (np.array([1.0, 0, 0])[None, :] - n_s[:, 0:1] * n_s) / sp[:, None]
        mid = c_s - ti[:, None] * proj
        vis_len, _ = clip_segments_to_convex_polygon_vectorized(
            mid[:, 1:3], dir_yz, 2.0 * hf, poly_ccw)
        # 원본과 동일 기준: 길이 1e-6 초과(비퇴화)이면서 lmin 이상
        det[idx[(vis_len >= LMIN) & (vis_len > 1e-6)]] = True
    return det


# ── ① 부분표본 정확성 검증 ──────────────────────────────────────
NS = 50_000
sub = hidden[:NS]
t = time.perf_counter()
kept_orig, nrem_orig = G.remove_face_intersecting(sub, face_xs, poly_ccw, LMIN)
t_orig_sub = time.perf_counter() - t
det_orig = np.ones(NS, dtype=bool)
kept_ids = set(id(d) for d in kept_orig)
det_orig = np.array([id(d) not in kept_ids for d in sub])

t = time.perf_counter()
det_vec_sub = detected_vectorized(C[:NS], N[:NS], R[:NS])
t_vec_sub = time.perf_counter() - t
same = (det_orig == det_vec_sub).all()
print(f"[검증] 5만 개: 원본 제거 {det_orig.sum()} vs 벡터화 {det_vec_sub.sum()} — "
      f"판정 {'완전 일치' if same else '불일치 ' + str(int((det_orig != det_vec_sub).sum())) + '개'}")
print(f"[시간] 5만 개: 원본 {t_orig_sub:.1f} s, 벡터화 {t_vec_sub:.3f} s "
      f"(x{t_orig_sub/t_vec_sub:.0f})")

# ── ② 전체 벤치마크 ───────────────────────────────────────────
t = time.perf_counter()
det_full = detected_vectorized(C, N, R)
t_vec_full = time.perf_counter() - t
est_orig_full = t_orig_sub * len(hidden) / NS
print(f"[전체] {len(hidden):,}개: 벡터화 {t_vec_full:.1f} s "
      f"(원본 외삽 {est_orig_full:.0f} s → 약 x{est_orig_full/t_vec_full:.0f} 단축)")
print(f"[전체] 제거 {det_full.sum():,}개")
