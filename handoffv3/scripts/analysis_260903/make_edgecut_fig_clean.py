# -*- coding: utf-8 -*-
"""edge-cut 개념도(캡션 없음, 흑백). 균열 원판 3개(2D 선분)가 협력해 블록을 닫는 예:
   ① 격자 + 원판  ② 원판을 가로지르는 연결선 절단  ③ 남은 연결선의 실제 연결성분(닫힌 블록 = 진회색)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K, DG, MG, LG, VLG, DK = "#111111", "#444444", "#9A9A9A", "#CFCFCF", "#F1F1F1", "#8A8A8A"
vs = 1.0; n = 10; xs = np.arange(n) * vs
# 원판 3개: 삼각형 꼭짓점 A,B,C 를 잇는 변을 양끝으로 0.9 씩 연장 (서로 겹쳐 닫힌 블록 형성)
V = [np.array([1.4, 2.2]), np.array([8.2, 3.4]), np.array([4.3, 8.1])]
segs = []
for i in range(3):
    a, b = V[i], V[(i + 1) % 3]; t = (b - a) / np.linalg.norm(b - a); nrm = np.array([-t[1], t[0]])
    c = (a + b) / 2; r = np.linalg.norm(b - a) / 2 + 0.9
    segs.append((c, nrm, t, r))
def cut_edges():
    cuts, kept = [], []
    for x in xs:
        for y in xs:
            for dx, dy in ((vs, 0), (0, vs)):
                x2, y2 = x + dx, y + dy
                if x2 > xs[-1] + 1e-9 or y2 > xs[-1] + 1e-9: continue
                pa, pb = np.array([x, y]), np.array([x2, y2]); hit = None
                for c, nrm, t, r in segs:
                    d1, d2 = (pa - c) @ nrm, (pb - c) @ nrm
                    if (d1 > 0) != (d2 > 0):
                        tt = d1 / (d1 - d2); q = pa + tt * (pb - pa)
                        if np.sum((q - c) ** 2) <= r * r: hit = q; break
                (cuts if hit is not None else kept).append(((x, y), (x2, y2), hit))
    return cuts, kept
cuts, kept = cut_edges()
# 연결성분
idx = {(x, y): i for i, (x, y) in enumerate((x, y) for x in xs for y in xs)}
rows_, cols_ = zip(*[(idx[a], idx[b]) for a, b, _ in kept])
g = coo_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(idx), len(idx)))
ncomp, lab = connected_components(g, directed=False)
sizes = np.bincount(lab); big = np.argmax(sizes)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
def base(ax, label):
    ax.set_aspect("equal"); ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.6, n - 0.4); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(MG); s.set_linewidth(0.8)
    ax.text(-0.45, n - 0.6, label, fontsize=14, fontweight="bold", color=K, va="top")
def grid(ax):
    for x in xs:
        ax.plot([x, x], [xs[0], xs[-1]], color=MG, lw=1.3, zorder=1); ax.plot([xs[0], xs[-1]], [x, x], color=MG, lw=1.3, zorder=1)
def nodes(ax): ax.scatter(*np.meshgrid(xs, xs), s=22, color=DG, zorder=4)
def draw_discs(ax, lw):
    for c, nrm, t, r in segs:
        p0, p1 = c - r * t, c + r * t
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=K, lw=lw, solid_capstyle="round", zorder=5)
def draw_cuts(ax):
    for a, b, q in cuts:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=3.4, zorder=2)
        perp = np.array([-(b[1]-a[1]), b[0]-a[0]]) * 0.16
        ax.plot([q[0]-perp[0], q[0]+perp[0]], [q[1]-perp[1], q[1]+perp[1]], color=K, lw=2.6, zorder=6)
ax = axes[0]; base(ax, "①"); grid(ax); nodes(ax); draw_discs(ax, 4.0)
ax = axes[1]; base(ax, "②"); grid(ax); nodes(ax); draw_discs(ax, 2.0); draw_cuts(ax)
ax = axes[2]; base(ax, "③")
for (x, y), i in idx.items():
    ax.add_patch(Rectangle((x - vs/2, y - vs/2), vs, vs, facecolor=(VLG if lab[i] == big else DK), edgecolor="none", zorder=0))
grid(ax); nodes(ax); draw_discs(ax, 2.0); draw_cuts(ax)
fig.tight_layout(w_pad=2.0)
fig.savefig("fig_edgecut_clean.png", dpi=200, facecolor="white")
print("components:", ncomp, "| block voxels:", int((lab != big).sum()))
