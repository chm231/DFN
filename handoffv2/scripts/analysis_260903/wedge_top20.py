# -*- coding: utf-8 -*-
"""상위 N개 균열을 무한평면으로 확장했을 때 터널 주변 제거가능 쐐기 블록 판정.

한양대 해석 조건 재현: 반지름 상위 N개(기본 20) → 무한평면(반지름 1e4) →
굴착 터널(도메인 전 구간) 주변에서 [터널 접촉 ∧ 도메인 외곽 비접촉] 컴포넌트
= 제거가능 쐐기 후보. 위치(천장/측벽/바닥)도 보고.

사용: (handoffv2에서, PYTHONPATH=.)
  python wedge_top20.py <json> [n_top] [out_npz]
출력 1줄: WEDGE <json명> n_top=<N> wedges=<개수> vols=<부피목록> pos=<위치목록>
"""
import json
import sys

import numpy as np

from dfn_analysis.detect_blocks_from_domain_json import (
    compute_edge_cuts, run_cca_edgecut, tunnel_geometry, block_detector)

json_path = sys.argv[1]
n_top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out_npz = sys.argv[3] if len(sys.argv) > 3 else None

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
fr = data["fractures"]
R_all = np.array([f_["radius_m"] for f_ in fr], dtype=np.float64)
idx = np.argsort(R_all)[::-1][:n_top]
centers = np.array([fr[i]["center_xyz_m"] for i in idx], dtype=np.float64)
normals = np.array([fr[i]["normal_xyz"] for i in idx], dtype=np.float64)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
radii = np.full(n_top, 1.0e4)  # 무한평면 근사

dom = data["meta"]["domain"]
x0, x1 = [float(v) for v in dom["x_range_m"]]
yz = dom["yz_bounds_m"]
box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)
z_top = poly_yz[:, 1].max()
z_bot = poly_yz[:, 1].min()

_, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
    poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=0.1, halo_dist=0.0,
    tunnel_xmin=x0, tunnel_xmax=x1)
tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
rock = ~tunnel_mask
vs = float(grid_info["voxel_size"])
zs = grid_info["zs"]

cutx, cuty, cutz = compute_edge_cuts(grid_info, centers, normals, radii)
labels3d, ncomp = run_cca_edgecut(rock, cutx, cuty, cutz)
counts = np.bincount(labels3d[labels3d >= 0], minlength=ncomp)

boundary = set()
for sl in (labels3d[0], labels3d[-1], labels3d[:, 0], labels3d[:, -1],
           labels3d[:, :, 0], labels3d[:, :, -1]):
    u = np.unique(sl)
    boundary.update(int(v) for v in u[u >= 0])
adj = np.zeros(labels3d.shape, bool)
tm = tunnel_mask
adj[:-1] |= tm[1:]; adj[1:] |= tm[:-1]
adj[:, :-1] |= tm[:, 1:]; adj[:, 1:] |= tm[:, :-1]
adj[:, :, :-1] |= tm[:, :, 1:]; adj[:, :, 1:] |= tm[:, :, :-1]
touch = set(int(v) for v in np.unique(labels3d[adj & (labels3d >= 0)]))

wedges = [l for l in sorted(touch - boundary) if counts[l] >= 8]
vols, poss = [], []
for l in wedges:
    iw = np.nonzero(labels3d == l)
    zc = zs[iw[2]].mean()
    vols.append(round(counts[l] * vs ** 3, 2))
    poss.append("crown" if zc > z_top - 1.5 else ("floor" if zc < z_bot + 1.5 else "wall"))
name = json_path.replace("\\", "/").split("/")[-1]
print(f"WEDGE {name} n_top={n_top} wedges={len(wedges)} vols={vols} pos={poss}")
crown_v = [v for v, p_ in zip(vols, poss) if p_ == "crown"]
wall_v = [v for v, p_ in zip(vols, poss) if p_ == "wall"]
print(f"SUM\t{name}\t{len(wedges)}\t{len(crown_v)}\t{max(crown_v) if crown_v else 0}"
      f"\t{len(wall_v)}\t{round(sum(vols),1)}")
if out_npz and wedges:
    out = np.zeros(labels3d.shape, np.int32)
    for k, l in enumerate(wedges):
        out[labels3d == l] = k + 1
    origin = np.array([grid_info["xs"][0], grid_info["ys"][0], grid_info["zs"][0]])
    np.savez_compressed(out_npz, labels=out, origin=origin, voxel_size=vs,
                        top_idx=idx)
