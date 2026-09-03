# -*- coding: utf-8 -*-
"""전체 도메인(x -0.145~21.368, 1~6면 시나리오) MC 블록 확률장 집계.

출력:
  example_io/blocks_edgecut/f06/block_prob_full_mc50.vti  (p_block, p_smooth)
  docs/figures/block_probability_full_mc50.png            (기굴착/전방/xz 3패널)
"""
import glob
import json

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

SP = "C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
H2 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"
X_FACE = 11.368  # 막장면 (기굴착/전방 경계)

files = sorted(glob.glob(f"{SP}/fullmc/blk_seed*_labels.npz"))
print(f"실현 {len(files)}개 집계")
count = None
nblk, vols = [], []
for f in files:
    z = np.load(f)
    hit = (z["labels"] > 0)
    count = hit.astype(np.uint8) if count is None else count + hit
    origin, vs = z["origin"], float(z["voxel_size"])
    nblk.append(len(np.unique(z["labels"])) - 1)
    vols.append(hit.sum() * vs ** 3)
p = count.astype(np.float32) / len(files)
p_smooth = gaussian_filter(p, sigma=3.0)

xs = origin[0] + vs * (np.arange(p.shape[0]) + 0.5)
ys = origin[1] + vs * (np.arange(p.shape[1]) + 0.5)
zs = origin[2] + vs * (np.arange(p.shape[2]) + 0.5)
exc = xs < X_FACE  # 기굴착 구간 마스크 (x방향)

print(f"실현별 블록 수 {min(nblk)}~{max(nblk)}, 총부피 {min(vols):.1f}~{max(vols):.1f} m³")
for name, sel in (("전체", slice(None)), ("기굴착(x<11.37)", exc), ("전방(x>11.37)", ~exc)):
    q = p[sel]
    nz = q > 0
    line = f"[{name}] ≥1회 {nz.sum()*vs**3:.1f} m³ ({nz.mean()*100:.2f}%)"
    for thr in (0.1, 0.3, 0.5, 0.9):
        line += f" | P≥{thr:g}: {(q >= thr-1e-9).sum()*vs**3:.2f} m³"
    print(line + f" | 최대 {q.max():.2f}")
n_det = (p >= 0.999).sum()
print(f"P=1.0(50/50) 복셀 {n_det:,}개 ({n_det*vs**3:.2f} m³)")

grid = pv.ImageData(dimensions=p.shape, spacing=(vs, vs, vs),
                    origin=tuple(float(v) + vs / 2 for v in origin))
grid.point_data["p_block"] = p.ravel(order="F")
grid.point_data["p_smooth"] = p_smooth.ravel(order="F")
out_vti = f"{H2}/example_io/blocks_edgecut/f06/block_prob_full_mc50.vti"
grid.save(out_vti)
print("saved", out_vti)

# ── 그림: 기굴착 yz / 전방 yz / 전체 xz ─────────────────────
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

poly = np.array(json.load(open(
    f"{H2}/example_io/f06/dfn_domain_f06_forward.json",
    encoding="utf-8"))["meta"]["domain"]["tunnel_polygon_yz_m"])

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
lv = [0.05, 0.1, 0.2, 0.3, 0.5, 0.9]

for ax, (title, mask) in zip(axes[:2], [
        ("기굴착 구간 (x -0.1~11.4, x방향 최대값)", exc),
        ("전방 미굴착 구간 (x 11.4~21.4, x방향 최대값)", ~exc)]):
    proj = gaussian_filter(p[mask].max(axis=0), 2.0)
    cf = ax.contourf(ys, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
    ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
            "k-", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")

ax = axes[2]
proj = gaussian_filter(p.max(axis=1), 2.0)
cf2 = ax.contourf(xs, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
ax.axvline(X_FACE, color="b", lw=1.5, ls="--", label="막장면 x=11.37")
ax.axhline(poly[:, 1].min(), color="k", lw=1, ls="--")
ax.axhline(poly[:, 1].max(), color="k", lw=1, ls="--")
ax.set_title("xz 평면 (y방향 최대값 — 점선: 터널 천장/바닥)")
ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)

fig.colorbar(cf, ax=axes, shrink=0.85, label="P(블록)")
fig.suptitle(f"전체 도메인 블록 존재확률 — 1~6면 관측, 몬테카를로 {len(files)}개 실현 "
             f"(굴착 가정, 터널 접촉 블록, 복셀 0.1 m)", fontsize=13)
out_png = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/block_probability_full_mc50.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved", out_png)
