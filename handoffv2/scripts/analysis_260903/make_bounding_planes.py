# -*- coding: utf-8 -*-
"""쐐기별 경계평면(top20 rank) 목록 CSV 생성.

사용: (handoffv2, PYTHONPATH=.) python make_bounding_planes.py <json> <out_csv>
"""
import csv
import json
import sys

import numpy as np

from dfn_analysis.detect_blocks_from_domain_json import (
    compute_edge_cuts, run_cca_edgecut, tunnel_geometry, block_detector)

jp, out = sys.argv[1], sys.argv[2]
N_TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 20
with open(jp, encoding="utf-8") as f:
    data = json.load(f)
fr = data["fractures"]
R_all = np.array([f_["radius_m"] for f_ in fr])
idx = np.argsort(R_all)[::-1][:N_TOP]
C = np.array([fr[i]["center_xyz_m"] for i in idx], float)
N = np.array([fr[i]["normal_xyz"] for i in idx], float)
N /= np.linalg.norm(N, axis=1, keepdims=True)
dom = data["meta"]["domain"]
x0, x1 = [float(v) for v in dom["x_range_m"]]
yz = dom["yz_bounds_m"]
box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
poly = np.asarray(dom["tunnel_polygon_yz_m"], float)
_, tm, _, gi = tunnel_geometry.build_voxel_masks(
    poly[:, 0], poly[:, 1], box, voxel_size=0.1, halo_dist=0.0,
    tunnel_xmin=x0, tunnel_xmax=x1)
tm = np.asarray(block_detector.to_numpy(tm))
cutx, cuty, cutz = compute_edge_cuts(gi, C, N, np.full(N_TOP, 1e4))
lab, nc = run_cca_edgecut(~tm, cutx, cuty, cutz)
cnt = np.bincount(lab[lab >= 0], minlength=nc)
bd = set()
for sl in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
    u = np.unique(sl)
    bd.update(int(v) for v in u[u >= 0])
adj = np.zeros(lab.shape, bool)
adj[:-1] |= tm[1:]; adj[1:] |= tm[:-1]
adj[:, :-1] |= tm[:, 1:]; adj[:, 1:] |= tm[:, :-1]
adj[:, :, :-1] |= tm[:, :, 1:]; adj[:, :, 1:] |= tm[:, :, :-1]
touch = set(int(v) for v in np.unique(lab[adj & (lab >= 0)]))
wedges = [l for l in sorted(touch - bd) if cnt[l] >= 8]
xs, ys, zs = gi["xs"], gi["ys"], gi["zs"]
rows = []
for k, l in enumerate(wedges):
    iw = np.nonzero(lab == l)
    P = np.column_stack([xs[iw[0]], ys[iw[1]], zs[iw[2]]]).astype(float)
    planes = []
    for j in range(N_TOP):
        d = (P - C[j]) @ N[j]
        pf = (d > 0).mean()
        if min(pf, 1 - pf) <= 0.02 and (np.abs(d) <= 0.12).sum() >= 5:
            planes.append(j + 1)
    rows.append([k + 1, len(planes), " ".join(map(str, planes))])
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["wedge_id", "n_bounding_planes", f"plane_ranks(top{N_TOP}_fractures.csv)"])
    w.writerows(rows)
print("saved", out)
