# -*- coding: utf-8 -*-
"""개념도 2장 — 흑백(그레이스케일) 테마. 규약: 모식도는 색 대신 명도·선 굵기·선 종류로 구분."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K, DG, MG, LG, VLG = "#111111", "#444444", "#8C8C8C", "#C8C8C8", "#EDEDED"

# ---------- Fig 1: edge-cut ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
vs = 1.0; n = 8; xs = np.arange(n) * vs
c = np.array([3.6, 3.3]); ang = np.deg2rad(35); nrm = np.array([np.cos(ang), np.sin(ang)]); t = np.array([-nrm[1], nrm[0]]); r = 2.6
p0, p1 = c - r * t, c + r * t
def draw_grid(ax, title):
    for x in xs:
        ax.plot([x, x], [xs[0], xs[-1]], color=LG, lw=0.8); ax.plot([xs[0], xs[-1]], [x, x], color=LG, lw=0.8)
    ax.scatter(*np.meshgrid(xs, xs), s=14, color=DG, zorder=3)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=K, lw=3.2, label="유한 균열 원판 (반지름 r)")
    ax.set_aspect("equal"); ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.6, n - 0.4); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
ax = axes[0]; draw_grid(ax, "(a) 종전: 균열 복셀 (두께 = 0.6·복셀)")
tol = 0.6 * vs
for x in xs:
    for y in xs:
        p = np.array([x, y]); d = (p - c) @ nrm; rho = abs((p - c) @ t)
        if abs(d) <= tol and rho <= r:
            ax.add_patch(Rectangle((x - vs/2, y - vs/2), vs, vs, facecolor=LG, hatch="///", edgecolor=MG, lw=0, zorder=1))
ax.text(0.2, 7.2, "빗금 복셀 = 암반에서 제거 → 부피 손실·두께<복셀이면 구멍", fontsize=8.5, color=DG)
ax = axes[1]; draw_grid(ax, "(b) edge-cut: 원판을 가로지르는 이웃 연결선만 절단")
ncut = 0
for x in xs:
    for y in xs:
        for dx, dy in ((vs, 0), (0, vs)):
            x2, y2 = x + dx, y + dy
            if x2 > xs[-1] + 1e-9 or y2 > xs[-1] + 1e-9: continue
            pa, pb = np.array([x, y]), np.array([x2, y2]); d1, d2 = (pa - c) @ nrm, (pb - c) @ nrm
            if (d1 > 0) != (d2 > 0):
                tt = d1 / (d1 - d2); q = pa + tt * (pb - pa)
                if np.sum((q - c) ** 2) <= r * r:
                    ax.plot([x, x2], [y, y2], color=K, lw=3.4, zorder=2); ncut += 1
                else:
                    ax.plot([x, x2], [y, y2], color=MG, lw=2.0, ls=(0, (2, 2)), zorder=2)
ax.text(0.2, 7.2, f"절단된 연결선 {ncut}개 (굵은 실선) · 원판 밖은 유지 (점선)", fontsize=8.5, color=DG)
ax = axes[2]; draw_grid(ax, "(c) 남은 연결선의 연결성분 = 블록 후보")
for x in xs:
    for y in xs:
        p = np.array([x, y]); d = (p - c) @ nrm
        ax.add_patch(Rectangle((x - vs/2, y - vs/2), vs, vs, color=(VLG if d > 0 else LG), zorder=0))
ax.text(0.2, 7.2, "원판 끝(crack tip) 너머로는 연결 유지 → 유한 균열은 스스로 블록을 닫지 못함", fontsize=8.5, color=DG)
axes[0].legend(loc="lower right", fontsize=8.5)
fig.suptitle("블록 판정 원리 — 복셀 격자에서 균열을 '두께 0의 분리면'으로 다루는 edge-cut (2D 단면 개념도)", fontsize=12)
fig.tight_layout(); fig.savefig("fig_edgecut_concept.png", dpi=170); print("fig1 ok")

# ---------- Fig 2: polyhedra ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
vs = 0.5
planes = [(np.array([1.0, 1.0]), np.array([0.2, -1.0])), (np.array([5.5, 1.5]), np.array([1.0, 0.35])), (np.array([3.0, 5.2]), np.array([-0.3, 1.0]))]
planes = [(p0_, n_ / np.linalg.norm(n_)) for p0_, n_ in planes]
def inside(p): return all(((p - p0_) @ n_) <= 0 for p0_, n_ in planes) and p[0] >= 0.2
gx = np.arange(0, 8, vs) + vs/2
vox = [(x, y) for x in gx for y in gx if inside(np.array([x, y]))]
def base(ax, title, sub):
    ax.set_aspect("equal"); ax.set_xlim(0, 8); ax.set_ylim(0, 7); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11, pad=6)
    ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color=DG)
def draw_vox(ax, fc=LG):
    for x, y in vox: ax.add_patch(Rectangle((x - vs/2, y - vs/2), vs, vs, facecolor=fc, edgecolor="white", lw=0.5))
ax = axes[0]; base(ax, "① edge-cut CCA 결과", "닫힌 컴포넌트의 복셀 집합 — 표면이 계단형"); draw_vox(ax)
ax = axes[1]; base(ax, "② 경계 원판 판별", "굵은 실선 = 채택된 경계 원판(95 % 한쪽 · 경계 접촉 ≤1.2h · 반경 내)\n점선 = 블록 안에서 끝나는 슬릿 → 볼록 재구성에서는 무시"); draw_vox(ax)
for p0_, n_ in planes:
    tt = np.array([-n_[1], n_[0]]); a, b = p0_ - 6 * tt, p0_ + 6 * tt
    ax.plot([a[0], b[0]], [a[1], b[1]], color=K, lw=2.4)
ps, ns = np.array([3.0, 3.0]), np.array([1.0, -0.6]); ns /= np.linalg.norm(ns); ts = np.array([-ns[1], ns[0]])
ax.plot([ps[0] - 1.2*ts[0], ps[0] + 1.2*ts[0]], [ps[1] - 1.2*ts[1], ps[1] + 1.2*ts[1]], color=DG, lw=2.2, ls=(0, (3, 2)))
poly = [np.array([0.2, 0.2]), np.array([7.8, 0.2]), np.array([7.8, 6.8]), np.array([0.2, 6.8])]
def clip(poly, p0_, n_):
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]; da, db = (a - p0_) @ n_, (b - p0_) @ n_
        if da <= 0: out.append(a)
        if (da <= 0) != (db <= 0): tq = da / (da - db); out.append(a + tq * (b - a))
    return out
for p0_, n_ in planes: poly = clip(poly, p0_, n_)
poly = np.array(poly)
av = len(vox) * vs * vs
ap = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
ax = axes[2]; base(ax, "③ 반공간 순차 클리핑 → 볼록 다면체", f"복셀 bbox에서 시작해 경계 평면(+터널 벽·도메인 면)으로 되깎음\n이 예의 면적비 다면체/복셀 = {ap/av:.2f} (실데이터 부피비 중앙값 0.90, >2이면 'open')")
draw_vox(ax, VLG)
ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=K, lw=3.0))
fig.suptitle("다면체 재구성 — 계단형 복셀 블록을 경계 균열 평면·터널 벽·도메인 면으로 되깎아 정확한 쐐기 기하 복원 (2D 개념도)", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("fig_polyhedra_concept.png", dpi=170); print("fig2 ok")
