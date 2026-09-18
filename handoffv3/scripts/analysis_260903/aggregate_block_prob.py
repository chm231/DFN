# -*- coding: utf-8 -*-
"""MC 10 실현의 블록 라벨을 집계해 복셀별 블록 존재확률 장(field)을 만든다.

출력:
  example_io/blocks_edgecut/block_prob_mc10.vti  (point data: p_block, p_smooth)
  docs/figures/block_probability_mc10.png        (yz/xz 투영 컨투어)
"""
import glob
import json

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

SP = "C:/Users/user/AppData/Local/Temp/claude/c--Users-user-OneDrive-2026-1-3D-DFN-modeling/c501f425-c178-491d-b844-fdbebd260c97/scratchpad"
H2 = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/handoffv2"

files = sorted(glob.glob(f"{SP}/mc10/blk_seed*_labels.npz"))
print(f"실현 {len(files)}개 집계")
count = None
for f in files:
    z = np.load(f)
    hit = (z["labels"] > 0)
    count = hit.astype(np.uint8) if count is None else count + hit
    origin, vs = z["origin"], float(z["voxel_size"])
p = count.astype(np.float32) / len(files)
p_smooth = gaussian_filter(p, sigma=3.0)  # 0.3 m 평활 — 컨투어용

nz = p > 0
print(f"블록이 1회 이상 나타난 복셀 {nz.sum():,}개 ({nz.mean()*100:.2f}%)")
for k in range(1, len(files) + 1):
    n = (count >= k).sum()
    if n:
        print(f"  P >= {k/len(files):.1f} : {n:,} 복셀 ({n*vs**3:.2f} m³)")

grid = pv.ImageData(dimensions=p.shape, spacing=(vs, vs, vs),
                    origin=tuple(float(v) + vs / 2 for v in origin))
grid.point_data["p_block"] = p.ravel(order="F")
grid.point_data["p_smooth"] = p_smooth.ravel(order="F")
out_vti = f"{H2}/example_io/blocks_edgecut/block_prob_mc50.vti"
grid.save(out_vti)
print("saved", out_vti)

# ── 2D 투영 컨투어 그림 ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

poly = np.array(json.load(open(
    f"{H2}/example_io/dfn_domain_x22.204-32.204_halo5_hz7.48.json",
    encoding="utf-8"))["meta"]["domain"]["tunnel_polygon_yz_m"])
ys = origin[1] + vs * (np.arange(p.shape[1]) + 0.5)
zs = origin[2] + vs * (np.arange(p.shape[2]) + 0.5)
xs = origin[0] + vs * (np.arange(p.shape[0]) + 0.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
lv = [0.05, 0.1, 0.2, 0.3, 0.5]

ax = axes[0]
proj = gaussian_filter(p.max(axis=0), 2.0)  # yz: x 방향 최대
cf = ax.contourf(ys, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
        "k-", lw=1.5, label="터널 단면")
ax.set_title("블록 존재확률 (yz 평면, x방향 최대값)")
ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
proj = gaussian_filter(p.max(axis=1), 2.0)  # xz: y 방향 최대
cf2 = ax.contourf(xs, zs, proj.T, levels=lv, cmap="YlOrRd", extend="max")
ax.axhline(poly[:, 1].min(), color="k", lw=1, ls="--")
ax.axhline(poly[:, 1].max(), color="k", lw=1, ls="--")
ax.set_title("블록 존재확률 (xz 평면, y방향 최대값 — 점선: 터널 천장/바닥)")
ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")

fig.colorbar(cf, ax=axes, shrink=0.85, label="P(블록)")
fig.suptitle(f"몬테카를로 {len(files)}개 실현의 복셀별 블록 존재확률 (굴착 가정, 터널 접촉 블록)",
             fontsize=12)
out_png = "c:/Users/user/OneDrive/2026-1/3D DFN modeling/docs/figures/block_probability_mc50.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved", out_png)
