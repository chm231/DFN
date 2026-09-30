# -*- coding: utf-8 -*-
"""면(다면체) 복원 개념도 — 캡션 없음, 흑백. ① 계단형 복셀 블록 ② 경계 원판 판별(굵은 실선=채택, 점선=관통 슬릿 제외) ③ 반공간 클리핑 → 볼록 다면체."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K, DG, MG, LG, VLG = "#111111", "#444444", "#9A9A9A", "#CFCFCF", "#EFEFEF"
vs = 0.5
planes = [(np.array([1.0, 1.0]), np.array([0.2, -1.0])), (np.array([5.5, 1.5]), np.array([1.0, 0.35])), (np.array([3.0, 5.2]), np.array([-0.3, 1.0]))]
planes = [(p0, n_ / np.linalg.norm(n_)) for p0, n_ in planes]
def inside(p): return all(((p - p0) @ n_) <= 0 for p0, n_ in planes) and p[0] >= 0.2
gx = np.arange(0, 8, vs) + vs/2
vox = [(x, y) for x in gx for y in gx if inside(np.array([x, y]))]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
def base(ax, label):
    ax.set_aspect("equal"); ax.set_xlim(0, 8); ax.set_ylim(0, 7); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(MG); s.set_linewidth(0.8)
    ax.text(0.15, 6.85, label, fontsize=14, fontweight="bold", color=K, va="top")
def draw_vox(ax, fc=LG):
    for x, y in vox: ax.add_patch(Rectangle((x - vs/2, y - vs/2), vs, vs, facecolor=fc, edgecolor="white", lw=0.6))
ax = axes[0]; base(ax, ""); draw_vox(ax)
ax = axes[1]; base(ax, "①"); draw_vox(ax)
for p0, n_ in planes:
    tt = np.array([-n_[1], n_[0]]); a, b = p0 - 6 * tt, p0 + 6 * tt
    ax.plot([a[0], b[0]], [a[1], b[1]], color=K, lw=2.6)
ps, ns = np.array([3.0, 3.0]), np.array([1.0, -0.6]); ns /= np.linalg.norm(ns); ts = np.array([-ns[1], ns[0]])
ax.plot([ps[0] - 1.2*ts[0], ps[0] + 1.2*ts[0]], [ps[1] - 1.2*ts[1], ps[1] + 1.2*ts[1]], color=DG, lw=2.4, ls=(0, (3, 2)))
ax.plot([0.2, 0.2], [0, 7], color=MG, lw=1.6, ls=(0, (6, 3)))
poly = [np.array([0.2, 0.2]), np.array([7.8, 0.2]), np.array([7.8, 6.8]), np.array([0.2, 6.8])]
def clip(poly, p0, n_):
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]; da, db = (a - p0) @ n_, (b - p0) @ n_
        if da <= 0: out.append(a)
        if (da <= 0) != (db <= 0): t = da / (da - db); out.append(a + t * (b - a))
    return out
for p0, n_ in planes: poly = clip(poly, p0, n_)
poly = np.array(poly)
ax = axes[2]; base(ax, "②"); draw_vox(ax, VLG)
ax.plot([0.2, 0.2], [0, 7], color=MG, lw=1.6, ls=(0, (6, 3)))
ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=K, lw=3.2))
fig.tight_layout(w_pad=2.0)
fig.savefig("fig_polyhedra_clean.png", dpi=200, facecolor="white"); print("ok")
