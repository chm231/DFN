# -*- coding: utf-8 -*-
"""블라인드 검증: 1~6면 MC50 제자리 블록 확률장 vs 면 07~12 실측 기반 블록.

정답측: 12면 파이프라인의 관측 복원 원판 중 중심 x > 11.95 (면 07~12 증거 포함,
        면 01~06 단독 원판은 중심이 11.31 이하이므로 예측 입력 누출 없음)
        → 예측과 동일 격자(x 11.368~21.368, 복셀 0.1, 터널 없음)에서
        제자리 닫힌 블록(도메인 외곽 비접촉) 판정.
비교: 실제 블록 복셀에서의 예측 P 분포, 임계값별 적중/농축(lift), P>=0.5 예측
      클러스터의 실측 확인 여부.

사용: (handoffv2에서, PYTHONPATH=.) python validate_blind_f06.py
"""
import json

import numpy as np
import pandas as pd
import pyvista as pv

from dfn_analysis.detect_blocks_from_domain_json import (
    compute_edge_cuts, run_cca_edgecut, tunnel_geometry, block_detector)

H2 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
X_SPLIT = 11.95  # 면 06(11.31)과 07(12.63) 중간 — 블라인드 경계

# ── 정답측 원판: 면 07~12 증거 포함 관측 복원 원판 ─────────────
df = pd.read_csv(f"{H2}/demo_output/dfm_demo/reconstruct/reconstructed_discs.csv")
sel = df[df.cx > X_SPLIT].reset_index(drop=True)
print(f"[정답측] 12면 복원 원판 {len(df)}개 중 면07~12 증거(cx>{X_SPLIT}) {len(sel)}개")
print(f"  n_faces 분포: {sel.n_faces.value_counts().to_dict()}, "
      f"반지름 중앙값 {sel.radius.median():.2f} m")

centers = sel[["cx", "cy", "cz"]].to_numpy(float)
normals = sel[["nx", "ny", "nz"]].to_numpy(float)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
radii = sel["radius"].to_numpy(float)

# ── 예측과 동일 도메인/격자 ──────────────────────────────────
dom = json.load(open(f"{H2}/example_io/f06/dfn_domain_f06_forward.json",
                     encoding="utf-8"))["meta"]["domain"]
x0, x1 = [float(v) for v in dom["x_range_m"]]
yz = dom["yz_bounds_m"]
box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], float)

_, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
    poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=0.1, halo_dist=0.0,
    tunnel_xmin=x0, tunnel_xmax=x1)
tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
rock = np.ones(tunnel_mask.shape, dtype=bool)
vs = float(grid_info["voxel_size"])
xs, ys, zs = grid_info["xs"], grid_info["ys"], grid_info["zs"]

cutx, cuty, cutz = compute_edge_cuts(grid_info, centers, normals, radii)
labels3d, ncomp = run_cca_edgecut(rock, cutx, cuty, cutz)
counts = np.bincount(labels3d[labels3d >= 0], minlength=ncomp)
boundary = set()
for sl in (labels3d[0], labels3d[-1], labels3d[:, 0], labels3d[:, -1],
           labels3d[:, :, 0], labels3d[:, :, -1]):
    u = np.unique(sl)
    boundary.update(int(v) for v in u[u >= 0])
keep = [l for l in range(ncomp) if counts[l] >= 8 and l not in boundary]
actual = np.isin(labels3d, keep)
print(f"[정답측] 실측 기반 제자리 블록 {len(keep)}개, 부피 {actual.sum()*vs**3:.2f} m³")

# ── 예측 확률장 로드 (동일 격자 확인) ────────────────────────
g = pv.read(f"{H2}/example_io/blocks_edgecut/f06/block_prob_insitu_mc50.vti")
p = np.asarray(g.point_data["p_block"]).reshape(labels3d.shape, order="F")
assert p.shape == labels3d.shape

# ── 비교 통계 ────────────────────────────────────────────────
pin = p[actual]
print("\n[비교] 실제 블록 복셀에서의 예측 P:")
print(f"  평균 {pin.mean():.3f} / 중앙값 {np.median(pin):.3f} / 최대 {pin.max():.2f}"
      f"   (도메인 전체 평균 P = {p.mean():.4f})")
print(f"  농축비(lift, 평균 기준) = {pin.mean()/p.mean():.1f}배")
dom_v = p.size
for thr in (0.05, 0.1, 0.2, 0.5):
    a = (pin >= thr - 1e-9).mean()
    b = (p >= thr - 1e-9).mean()
    lift = a / b if b > 0 else np.nan
    print(f"  P>={thr:g}: 실제블록 부피의 {a*100:5.1f}% vs 도메인의 {b*100:5.2f}%"
          f"  (lift {lift:.0f}배)")

# 블록별 적중표
rows = []
for l in keep:
    m = labels3d == l
    iw = np.nonzero(m)
    cx_b = xs[iw[0]].mean(); cy_b = ys[iw[1]].mean(); cz_b = zs[iw[2]].mean()
    rows.append((counts[l]*vs**3, cx_b, cy_b, cz_b,
                 p[m].mean(), p[m].max()))
rows.sort(reverse=True)
print("\n[블록별] 부피 m³ | 중심(x,y,z) | 예측P 평균/최대")
for v, cx_b, cy_b, cz_b, pm, px in rows[:15]:
    print(f"  {v:6.3f} | ({cx_b:5.2f},{cy_b:6.2f},{cz_b:5.2f}) | {pm:.3f}/{px:.2f}")

# 역방향: 예측 P>=0.5 영역이 실측으로 확인되는가
hi = p >= 0.5
if hi.any():
    inb = actual[hi].mean()
    # 실제 블록까지 최근접 거리
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~actual, sampling=vs)
    print(f"\n[역방향] 예측 P>=0.5 복셀 {hi.sum():,}개 중 실제 블록 내부 {inb*100:.0f}%, "
          f"실제 블록까지 거리 중앙값 {np.median(dist[hi]):.2f} m / p90 {np.percentile(dist[hi],90):.2f} m")

# ── 그림: 예측 필드 + 실제 블록 윤곽 오버레이 ────────────────
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
lv = [0.05, 0.1, 0.2, 0.3, 0.5]

ax = axes[0]
proj = gaussian_filter(p.max(axis=0), 2.0)
cf = ax.contourf(ys, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
am = actual.max(axis=0)
ax.contour(ys, zs, am.T.astype(float), levels=[0.5], colors="blue", linewidths=1.6)
ax.plot(np.append(poly_yz[:, 0], poly_yz[0, 0]), np.append(poly_yz[:, 1], poly_yz[0, 1]),
        "k--", lw=1.0)
ax.set_title("yz 투영 — 채색: 예측 P(1~6면), 파란 윤곽: 실측 블록(면 07~12)")
ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")

ax = axes[1]
proj = gaussian_filter(p.max(axis=1), 2.0)
cf2 = ax.contourf(xs, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
am = actual.max(axis=1)
ax.contour(xs, zs, am.T.astype(float), levels=[0.5], colors="blue", linewidths=1.6)
for fx in (12.63, 15.4338, 16.9103, 18.2338, 20.9309):
    ax.axvline(fx, color="grey", lw=0.7, ls=":")
ax.set_title("xz 투영 — 점선: 실측 면 07~11 위치")
ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")

fig.colorbar(cf, ax=axes, shrink=0.85, label="예측 P(블록)")
fig.suptitle("블라인드 검증: 1~6면 MC50 블록 확률 예측 vs 면 07~12 실측 기반 제자리 블록",
             fontsize=12)
out = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/blind_validation_f06_mc50.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
