# -*- coding: utf-8 -*-
"""전방 도메인에 설계 단면을 압출해 굴착했을 때 풀리는 블록 라벨 저장 (MC용).

detect_interior_labels.py 와 한 줄만 다르다: rock = ~tunnel_mask.
전방 미굴착 암반에 설계 터널 단면을 압출해 넣고(굴착 가정), edge-cut CCA 후
도메인 6면에 닿지 않는 컴포넌트를 '제거가능 블록'으로 판정한다.
떨어지는 블록은 굴착 전에는 닫혀 있지 않고 자유면이 생겨야 풀리므로,
조기경보가 답해야 하는 양은 이쪽이다(키블록 개념, Goodman & Shi).

사용: (handoffv2에서, PYTHONPATH=.)
  python detect_interior_labels.py <domain_json> <voxel> <out_npz>
"""
import json
import sys

import numpy as np

from dfn_analysis.detect_blocks_from_domain_json import (
    compute_edge_cuts, run_cca_edgecut, tunnel_geometry, block_detector)

json_path, voxel, out_npz = sys.argv[1], float(sys.argv[2]), sys.argv[3]

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
fr = data["fractures"]
centers = np.array([f_["center_xyz_m"] for f_ in fr], dtype=np.float64)
normals = np.array([f_["normal_xyz"] for f_ in fr], dtype=np.float64)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
radii = np.array([f_["radius_m"] for f_ in fr], dtype=np.float64)
dom = data["meta"]["domain"]
x0, x1 = [float(v) for v in dom["x_range_m"]]
yz = dom["yz_bounds_m"]
box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)

_, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
    poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=voxel, halo_dist=0.0,
    tunnel_xmin=x0, tunnel_xmax=x1)
tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
rock = ~tunnel_mask  # 설계 단면을 전방으로 압출해 굴착한 상태
vs = float(grid_info["voxel_size"])

cutx, cuty, cutz = compute_edge_cuts(grid_info, centers, normals, radii)
labels3d, ncomp = run_cca_edgecut(rock, cutx, cuty, cutz)
counts = np.bincount(labels3d[labels3d >= 0], minlength=ncomp)

boundary = set()
for sl in (labels3d[0], labels3d[-1], labels3d[:, 0], labels3d[:, -1],
           labels3d[:, :, 0], labels3d[:, :, -1]):
    u = np.unique(sl)
    boundary.update(int(v) for v in u[u >= 0])
keep = np.array([l for l in range(ncomp) if counts[l] >= 8 and l not in boundary],
                dtype=np.int64)

out = np.zeros(labels3d.shape, dtype=np.int32)
if len(keep):
    remap = np.zeros(ncomp, dtype=np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    valid = labels3d >= 0
    out[valid] = remap[labels3d[valid]]
vol = int((out > 0).sum()) * vs ** 3
origin = np.array([grid_info["xs"][0], grid_info["ys"][0], grid_info["zs"][0]])
np.savez_compressed(out_npz, labels=out, origin=origin, voxel_size=vs)
print(f"result 제거가능블록 {len(keep)}개, 부피 {vol:.2f} m3 -> {out_npz}")
