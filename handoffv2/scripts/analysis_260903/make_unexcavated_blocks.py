# -*- coding: utf-8 -*-
"""미굴착 도메인(터널 없음) 내부 닫힌 블록: edge-cut CCA + 다면체 재구성.

전방 생성 도메인을 미굴착 암반으로 취급(rock=전체), 도메인 6면 어디에도 닿지 않는
컴포넌트를 내부 닫힌 블록으로 판정하고 경계 원판 평면으로 다면체 복원.

사용: (handoffv2에서, PYTHONPATH=.)
  python make_unexcavated_blocks.py <domain_json> <voxel> <out_prefix>
"""
import json
import sys
from pathlib import Path

import numpy as np

from dfn_analysis.detect_blocks_from_domain_json import (
    compute_edge_cuts, run_cca_edgecut, tunnel_geometry, block_detector)
from dfn_analysis.cut_blocks_polyhedral import hull_of
from dfn_analysis.reconstruct_block_polyhedra import clip_hull

json_path, voxel, prefix = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])

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
print(f"[input] 균열 {len(fr):,}개, 도메인 z [{box[4]:.2f},{box[5]:.2f}] "
      f"(dz={box[5]-box[4]:.2f} m), ε = {voxel} m")

_, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
    poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=voxel, halo_dist=0.0,
    tunnel_xmin=x0, tunnel_xmax=x1)
tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
rock = np.ones(tunnel_mask.shape, dtype=bool)  # 미굴착 — 전부 암반
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
vols = [counts[l] * vs ** 3 for l in keep]
print(f"[interior] 내부 닫힌 블록 {len(keep):,}개, 부피 합 {sum(vols):.2f} m³, "
      f"최대 {max(vols):.3f}" if keep else "[interior] 없음")

disc_lo = centers - radii[:, None]
disc_hi = centers + radii[:, None]
import pyvista as pv
parts = []
n_ok = 0
for k, lbl in enumerate(keep):
    iw = np.nonzero(labels3d == lbl)
    P = np.column_stack([xs[iw[0]], ys[iw[1]], zs[iw[2]]]).astype(np.float64)
    lo, hi = P.min(axis=0), P.max(axis=0)
    m = 0.75 * vs
    corners = np.array([[a, b, c] for a in (lo[0]-m, hi[0]+m)
                        for b in (lo[1]-m, hi[1]+m) for c in (lo[2]-m, hi[2]+m)])
    hull = hull_of(corners)
    cand = np.nonzero(np.all(disc_hi >= lo - vs, axis=1)
                      & np.all(disc_lo <= hi + vs, axis=1))[0]
    for j in cand:
        d = (P - centers[j]) @ normals[j]
        pos_frac = float((d > 0).mean())
        if min(pos_frac, 1 - pos_frac) > 0.05:
            continue
        absd = np.abs(d)
        if absd.min() > 1.2 * vs:
            continue
        near = absd <= 1.2 * vs
        inplane2 = np.sum((P[near] - centers[j]) ** 2, axis=1) - d[near] ** 2
        if not (inplane2 <= (radii[j] + vs) ** 2).any():
            continue
        sgn = 1.0 if pos_frac >= 0.5 else -1.0
        new = clip_hull(hull, centers[j], -sgn * normals[j])
        if new is not None:
            hull = new
    if hull is None:
        continue
    for p0, nrm in [((x0, 0, 0), (-1, 0, 0)), ((x1, 0, 0), (1, 0, 0)),
                    ((0, box[3], 0), (0, 1, 0)), ((0, box[2], 0), (0, -1, 0)),
                    ((0, 0, box[5]), (0, 0, 1)), ((0, 0, box[4]), (0, 0, -1))]:
        new = clip_hull(hull, np.array(p0, float), np.array(nrm, float))
        if new is None:
            hull = None
            break
        hull = new
    if hull is None:
        continue
    n_ok += 1
    faces = np.column_stack([np.full(len(hull.simplices), 3), hull.simplices]).ravel()
    mm = pv.PolyData(hull.points, faces=faces)
    mm.cell_data["block_id"] = np.full(mm.n_cells, k + 1, dtype=np.int32)
    parts.append(mm)

if parts:
    merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    merged.save(f"{prefix}_polyhedra.vtp")
print(f"[recon] 다면체 재구성 {n_ok}/{len(keep)}")
print(f"[out] {prefix}_polyhedra.vtp")
