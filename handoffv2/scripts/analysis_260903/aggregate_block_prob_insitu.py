# -*- coding: utf-8 -*-
"""전방 미굴착 도메인(x 11.37~21.37) '제자리 닫힌 블록' MC 확률장 집계.

블록 정의: 터널 없음(굴착 전), 도메인 외곽 6면에 닿지 않는 완전 닫힌 컴포넌트.

출력:
  example_io/blocks_edgecut/f06/block_prob_insitu_mc50.vti  (p_block, p_smooth)
  docs/figures/block_probability_insitu_mc50.png
"""
import glob
import json

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

SP = "C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
H2 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"

files = sorted(glob.glob(f"{SP}/insitumc/blk_seed*_labels.npz"))
print(f"실현 {len(files)}개 집계")
count = None
nblk, vols = [], []
for f in files:
    z = np.load(f)
    hit = (z["labels"] > 0)
    count = hit.astype(np.uint8) if count is None else count + hit
    origin, vs = z["origin"], float(z["voxel_size"])
    nblk.append(int(z["labels"].max()))
    vols.append(hit.sum() * vs ** 3)
p = count.astype(np.float32) / len(files)
p_smooth = gaussian_filter(p, sigma=3.0)

print(f"실현별 블록 수 {min(nblk)}~{max(nblk)}, 총부피 {min(vols):.1f}~{max(vols):.1f} m³")
nz = p > 0
print(f"≥1회 {nz.sum():,}복셀 ({nz.mean()*100:.2f}%, {nz.sum()*vs**3:.1f} m³)")
for thr in (0.1, 0.2, 0.3, 0.5, 0.9):
    n = (p >= thr - 1e-9).sum()
    print(f"  P >= {thr:.1f} : {n:,} 복셀 ({n*vs**3:.2f} m³)")
print(f"최대 P = {p.max():.2f} ({(p == p.max()).sum()} 복셀)")
print(f"p_smooth 최대 {p_smooth.max():.3f}")

grid = pv.ImageData(dimensions=p.shape, spacing=(vs, vs, vs),
                    origin=tuple(float(v) + vs / 2 for v in origin))
grid.point_data["p_block"] = p.ravel(order="F")
grid.point_data["p_smooth"] = p_smooth.ravel(order="F")
out_vti = f"{H2}/example_io/blocks_edgecut/f06/block_prob_insitu_mc50.vti"
grid.save(out_vti)
print("saved", out_vti)

# ── 그림 ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

poly = np.array(json.load(open(
    f"{H2}/example_io/f06/dfn_domain_f06_forward.json",
    encoding="utf-8"))["meta"]["domain"]["tunnel_polygon_yz_m"])
xs = origin[0] + vs * (np.arange(p.shape[0]) + 0.5)
ys = origin[1] + vs * (np.arange(p.shape[1]) + 0.5)
zs = origin[2] + vs * (np.arange(p.shape[2]) + 0.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
lv = [0.05, 0.1, 0.2, 0.3, 0.5]

ax = axes[0]
proj = gaussian_filter(p.max(axis=0), 2.0)
cf = ax.contourf(ys, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
        "b--", lw=1.2, label="계획 터널 단면 (참고)")
ax.set_title("제자리 블록 존재확률 (yz 평면, x방향 최대값)")
ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
proj = gaussian_filter(p.max(axis=1), 2.0)
cf2 = ax.contourf(xs, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
ax.axhline(poly[:, 1].min(), color="b", lw=1, ls="--")
ax.axhline(poly[:, 1].max(), color="b", lw=1, ls="--")
ax.set_title("제자리 블록 존재확률 (xz 평면, y방향 최대값)")
ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")

fig.colorbar(cf, ax=axes, shrink=0.85, label="P(블록)")
fig.suptitle(f"막장면(x=11.37) 뒤 미굴착 도메인 — 제자리 닫힌 블록 존재확률, "
             f"몬테카를로 {len(files)}개 실현 (터널 없음, 복셀 0.1 m)", fontsize=12)
out_png = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/block_probability_insitu_mc50.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved", out_png)
