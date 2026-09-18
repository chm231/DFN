# -*- coding: utf-8 -*-
"""(a) 몬테카를로→확률장 개념도(흑백, 캡션 없음)
   (b) 1~6면 제자리 확률장 2D 투영(데이터 그림, 제목 없음, 계획 터널 윤곽 점선)
   (c) 면 08 블라인드: MC(lmin 0.5) 예측 분포 vs 실측(굵은 세로선)"""
import glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K, DG, MG, LG, VLG = "#111111", "#444444", "#9A9A9A", "#CFCFCF", "#EFEFEF"
H2 = r"C:\Users\user\OneDrive\2026-1\3D DFN modeling\handoffv2"
SP = r"C:\Users\user\AppData\Local\Temp\claude\c--Users-user-OneDrive-2026-1-3D-DFN-modeling\c501f425-c178-491d-b844-fdbebd260c97\scratchpad"

# ---------------- (a) 개념도: 실현 1, 실현 2 → 누적 확률 ----------------
n = 10
def realization(seed):
    r = np.random.default_rng(seed); m = np.zeros((n, n), bool)
    m[1:4, 6:9] = True                                   # 관측 원판 기반 블록: 막장면(왼쪽) 직후 천장부 — 모든 실현에 존재
    for _ in range(r.integers(2, 4)):                    # 확률 균열 기반 블록: 실현마다 위치가 다름
        i, j = r.integers(0, n - 2), r.integers(0, n - 2); m[i:i + r.integers(1, 3), j:j + r.integers(1, 3)] = True
    return m
reals = [realization(s) for s in (11, 22, 33, 44, 55, 66, 77, 88)]
P = np.mean(reals, axis=0)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
def base(ax, label):
    ax.set_aspect("equal"); ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(MG); s.set_linewidth(0.8)
    ax.text(-0.35, n - 0.6, label, fontsize=13, fontweight="bold", color=K, va="top")
    ax.plot([-0.5, -0.5], [-0.5, n - 0.5], color=K, lw=4)   # 왼쪽 굵은 선 = 막장면
for ax, m, lab in [(axes[0], reals[0], "실현 1"), (axes[1], reals[1], "실현 2")]:
    base(ax, lab)
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=(DG if m[i, j] else "white"), edgecolor=LG, lw=0.6))
ax = axes[2]; base(ax, "P = 출현 횟수 / N")
for i in range(n):
    for j in range(n):
        g = 1 - 0.85 * P[i, j]; ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=(g, g, g), edgecolor=LG, lw=0.6))
fig.tight_layout(w_pad=2.0); fig.savefig("fig_mc_concept.png", dpi=200, facecolor="white"); plt.close(fig)

# ---------------- (b) 제자리 확률장 2D 투영 ----------------
g = pv.read(H2 + r"\example_io\blocks_edgecut\f06\block_prob_insitu_lmin05_mc50.vti")
dims, sp, org = g.dimensions, g.spacing, g.origin
p = g.point_data["p_block"].reshape(dims, order="F")
xs = org[0] + sp[0] * np.arange(dims[0]); ys = org[1] + sp[1] * np.arange(dims[1]); zs = org[2] + sp[2] * np.arange(dims[2])
dom = json.load(open(H2 + r"\example_io\f06\dfn_domain_f06_forward.json", encoding="utf-8"))["meta"]["domain"]
poly = np.asarray(dom["tunnel_polygon_yz_m"], float); x0 = dom["x_range_m"][0]
cmap = LinearSegmentedColormap.from_list("pb", [(0.0, "white"), (0.2, "#FFE082"), (0.5, "#FB8C00"), (0.8, "#D32F2F"), (1.0, "#7B0000")])  # 0~0.5
yz = p.max(axis=0).T; xz = p.max(axis=1).T
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(width_ratios=[1, 1.05]))
ax = axes[0]
im = ax.imshow(yz, origin="lower", extent=[ys[0], ys[-1], zs[0], zs[-1]], cmap=cmap, vmin=0, vmax=0.5, aspect="equal")
ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="#1F3A8A", ls="--", lw=1.4))
ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]")
ax = axes[1]
im = ax.imshow(xz, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap=cmap, vmin=0, vmax=0.5, aspect="equal")
ax.axhline(poly[:, 1].max(), color="#1F3A8A", ls="--", lw=1.2); ax.axhline(poly[:, 1].min(), color="#1F3A8A", ls="--", lw=1.2)
ax.axvline(x0, color=K, lw=2.0)
ax.set_xlabel("x [m]  (굴진 방향)"); ax.set_ylabel("z [m]")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, extend="max"); cb.set_label("P(블록)")
fig.savefig("fig_prob_insitu_clean.png", dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("prob field ok | P>=0.5 voxels", int((p >= 0.5).sum()))

# ---------------- (c) 면 08 블라인드 ----------------
area = 0.5 * abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
df = pd.read_csv(H2 + r"\demo_output\dfm_demo\trace_dataset\trace_dataset_3d.csv")
f8 = df[(df.face_id == 8) & (df.observed_length_m >= 0.5)]
n_a, p21_a = len(f8), f8.observed_length_m.sum() / area
counts, p21s = [], []
for f in sorted(glob.glob(SP + r"\face08mc_l05\tr_seed*.npz")):
    z = np.load(f); L = z["lengths"]; m = L >= 0.5
    counts.append(int(m.sum())); p21s.append(L[m].sum() / area)
counts, p21s = np.array(counts), np.array(p21s)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, v, obs, xl in [(axes[0], counts, n_a, "면 08 절리선 개수 (≥ 0.5 m)"), (axes[1], p21s, p21_a, "면 08 P21 [m/m²] (≥ 0.5 m)")]:
    ax.hist(v, bins=12, color=LG, edgecolor=DG, lw=0.6)
    ax.axvline(obs, color=K, lw=2.4)
    ax.set_xlabel(xl); ax.set_ylabel("실현 수 (N = 50)")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(w_pad=2.0); fig.savefig("fig_blind_face08_clean.png", dpi=200, facecolor="white"); plt.close(fig)
print(f"face08: 실측 {n_a}개 P21 {p21_a:.2f} | MC {counts.mean():.0f}±{counts.std():.0f}, P21 {p21s.mean():.2f}±{p21s.std():.2f}, 실측 백분위 개수 {(counts<n_a).mean()*100:.0f}% P21 {(p21s<p21_a).mean()*100:.0f}%")
