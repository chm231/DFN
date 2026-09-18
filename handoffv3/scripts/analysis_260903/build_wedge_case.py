# -*- coding: utf-8 -*-
"""쐐기 발생 케이스 패키지 생성: 상위20 무한평면 쐐기 다면체 + 상위20 표 + 장면 vtp.

사용: (handoffv2에서, PYTHONPATH=.)
  python build_wedge_case.py <json> <out_dir>
출력: <out_dir>/wedges_polyhedra.vtp, top20_planes.vtp, tunnel.vtp,
      top20_fractures.csv, wedge_summary.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from dfn_analysis.detect_blocks_from_domain_json import (
    compute_edge_cuts, run_cca_edgecut, tunnel_geometry, block_detector)
from dfn_analysis.cut_blocks_polyhedral import hull_of
from dfn_analysis.reconstruct_block_polyhedra import clip_hull, tunnel_edge_planes
from select_top import N_OBS_DEFAULT, select_top_idx

json_path, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
N_TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_OBS = int(sys.argv[4]) if len(sys.argv) > 4 else N_OBS_DEFAULT

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
fr = data["fractures"]
R_all = np.array([f_["radius_m"] for f_ in fr])
idx, sel_summary = select_top_idx(fr, N_TOP, N_OBS)
print(sel_summary)
centers = np.array([fr[i]["center_xyz_m"] for i in idx], float)
normals = np.array([fr[i]["normal_xyz"] for i in idx], float)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
sets = [fr[i].get("set_id") for i in idx]
labels = [fr[i].get("label", "unobserved") for i in idx]
radii_true = R_all[idx]
radii_inf = np.full(len(idx), 1.0e4)

dom = data["meta"]["domain"]
x0, x1 = [float(v) for v in dom["x_range_m"]]
yz = dom["yz_bounds_m"]
box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], float)
z_top, z_bot = poly_yz[:, 1].max(), poly_yz[:, 1].min()

# ── 상위 20 표 (규약 명시: trend=arctan2(ny,nx), x=굴진방향, z=상방) ──
pole = normals.copy()
flip = pole[:, 2] < 0
pole[flip] *= -1.0
trend_pole = np.degrees(np.arctan2(pole[:, 1], pole[:, 0])) % 360
plunge_pole = np.degrees(np.arcsin(np.clip(pole[:, 2], -1, 1)))
dip = 90.0 - plunge_pole
dipdir = (trend_pole + 180.0) % 360
with open(out_dir / f"top{N_TOP}_fractures.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rank", "label", "set_id", "cx_m", "cy_m", "cz_m",
                "nx", "ny", "nz", "radius_m",
                "dip_deg", "dipdir_deg_trendconv", "pole_trend_deg", "pole_plunge_deg"])
    for k in range(N_TOP):
        w.writerow([k + 1, labels[k], sets[k], *np.round(centers[k], 3),
                    *np.round(normals[k], 4), round(float(radii_true[k]), 2),
                    round(dip[k], 1), round(dipdir[k], 1),
                    round(trend_pole[k], 1), round(plunge_pole[k], 1)])

# ── 쐐기 판정 (무한평면) ─────────────────────────────────────
_, tunnel_mask, _, gi = tunnel_geometry.build_voxel_masks(
    poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=0.1, halo_dist=0.0,
    tunnel_xmin=x0, tunnel_xmax=x1)
tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
rock = ~tunnel_mask
vs = 0.1
xs, ys, zs = gi["xs"], gi["ys"], gi["zs"]
cutx, cuty, cutz = compute_edge_cuts(gi, centers, normals, radii_inf)
lab, nc = run_cca_edgecut(rock, cutx, cuty, cutz)
cnt = np.bincount(lab[lab >= 0], minlength=nc)
bd = set()
for sl in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
    u = np.unique(sl)
    bd.update(int(v) for v in u[u >= 0])
adj = np.zeros(lab.shape, bool)
tm = tunnel_mask
adj[:-1] |= tm[1:]; adj[1:] |= tm[:-1]
adj[:, :-1] |= tm[:, 1:]; adj[:, 1:] |= tm[:, :-1]
adj[:, :, :-1] |= tm[:, :, 1:]; adj[:, :, 1:] |= tm[:, :, :-1]
touch = set(int(v) for v in np.unique(lab[adj & (lab >= 0)]))
wedges = [l for l in sorted(touch - bd) if cnt[l] >= 8]
print(f"[wedge] {len(wedges)}개")

# ── 매끈한 재구성: 절리 평면으로만 볼록체 → 터널 프리즘 곡면으로 절단 ──
# (대형 천장 쐐기는 터널 아치를 감싸는 비볼록 형상 — 터널면은 평면 클리핑이
#  아니라 닫힌 프리즘 표면과의 clip_surface 로 잘라내야 매끈+정확)
n = len(poly_yz)
xa, xb = x0 - 1.0, x1 + 1.0  # 캡이 도메인 밖에 오도록 연장
ring_a = np.column_stack([np.full(n, xa), poly_yz])
ring_b = np.column_stack([np.full(n, xb), poly_yz])
faces_side = [[4, i, (i+1) % n, n+(i+1) % n, n+i] for i in range(n)]
prism = pv.PolyData(np.vstack([ring_a, ring_b]),
                    faces=np.array([v for q in faces_side for v in q]))
cap_a = pv.PolyData(ring_a, faces=np.concatenate([[n], np.arange(n)])).triangulate()
cap_b = pv.PolyData(ring_b, faces=np.concatenate([[n], np.arange(n)])).triangulate()
tunnel_solid = prism.triangulate().merge(cap_a).merge(cap_b).clean()
tunnel_solid = tunnel_solid.compute_normals(auto_orient_normals=True)

# 터널 '내부'의 implicit distance 부호 판별 (최종 안전 절단용)
import vtk
_ipd = vtk.vtkImplicitPolyDataDistance()
_ipd.SetInput(tunnel_solid)
_cen_yz = poly_yz.mean(axis=0)
inside_sign = _ipd.EvaluateFunction([(x0 + x1) / 2, _cen_yz[0], _cen_yz[1]])


def clip_out_tunnel(mesh):
    """터널 내부로 들어간 부분을 확실히 제거 (implicit distance 0 절단)."""
    m2 = mesh.compute_implicit_distance(tunnel_solid)
    # inside_sign<0 이면 내부=음수 → 양수(외부) 유지 = invert False
    out = m2.clip_scalar(scalars="implicit_distance", value=0.0,
                         invert=(inside_sign > 0))
    return out.extract_surface() if out.n_cells else mesh

parts, rows = [], []
for k, l in enumerate(wedges):
    iw = np.nonzero(lab == l)
    P = np.column_stack([xs[iw[0]], ys[iw[1]], zs[iw[2]]]).astype(float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    m = 0.75 * vs
    corners = np.array([[a, b, c] for a in (lo[0]-m, hi[0]+m)
                        for b in (lo[1]-m, hi[1]+m) for c in (lo[2]-m, hi[2]+m)])
    hull = hull_of(corners)
    for j in range(N_TOP):
        d = (P - centers[j]) @ normals[j]
        pos_frac = float((d > 0).mean())
        if min(pos_frac, 1 - pos_frac) > 0.05 or np.abs(d).min() > 1.2 * vs:
            continue
        sgn = 1.0 if pos_frac >= 0.5 else -1.0
        new = clip_hull(hull, centers[j], -sgn * normals[j])
        if new is not None:
            hull = new
    if hull is None:
        continue
    faces = np.column_stack([np.full(len(hull.simplices), 3), hull.simplices]).ravel()
    hm = pv.PolyData(hull.points, faces=faces).triangulate()
    # 터널 밖(암반 쪽)만 유지 — invert 방향은 부피로 자동 판별
    cut = hm.clip_surface(tunnel_solid, invert=False)
    vol_vox = cnt[l] * vs ** 3
    if cut.n_cells == 0 or not (0.05 * vol_vox < abs(cut.volume) < 3.0 * hull.volume):
        cut2 = hm.clip_surface(tunnel_solid, invert=True)
        if cut2.n_cells and abs(abs(cut2.volume) - vol_vox) < abs(abs(cut.volume) - vol_vox):
            cut = cut2
    if cut.n_cells == 0:
        continue
    # 분리 조각 중 실제 블록 복셀을 포함하는 조각만 유지 (터널 반대편 조각 제거)
    bodies = cut.connectivity("all").split_bodies()
    if len(bodies) > 1:
        step = max(1, len(P) // 300)
        sample = pv.PolyData(P[::step])
        keep_bodies = []
        for b in bodies:
            surf = b.extract_surface().triangulate()
            try:
                enc = sample.select_enclosed_points(surf, tolerance=1e-6,
                                                    check_surface=False)
                if enc.point_data["SelectedPoints"].sum() > 0:
                    keep_bodies.append(surf)
            except Exception:
                keep_bodies.append(surf)
        if keep_bodies:
            cut = keep_bodies[0]
            for b in keep_bodies[1:]:
                cut = cut.merge(b)
            cut = cut.extract_surface()
    # 얇은 슬리버는 볼록 근사가 두께를 과장 → 평활 복셀 표면으로 대체
    if abs(cut.volume) > 2.0 * vol_vox or abs(cut.volume) < 0.3 * vol_vox:
        from scipy.ndimage import gaussian_filter
        i0 = [max(int(a.min()) - 3, 0) for a in iw]
        i1 = [int(a.max()) + 4 for a in iw]
        sub = np.zeros([b - a for a, b in zip(i0, i1)], np.float32)
        sub[iw[0] - i0[0], iw[1] - i0[1], iw[2] - i0[2]] = 1.0
        sub = gaussian_filter(sub, sigma=1.0)
        gsub = pv.ImageData(dimensions=sub.shape, spacing=(vs, vs, vs),
                            origin=(xs[i0[0]], ys[i0[1]], zs[i0[2]]))
        gsub.point_data["m"] = sub.ravel(order="F")
        cut = gsub.contour([0.35], scalars="m").extract_surface()
        if cut.n_cells == 0:
            continue
    cut = clip_out_tunnel(cut)
    if cut.n_cells == 0:
        continue
    zc = zs[iw[2]].mean()
    pos = "crown" if zc > z_top - 1.5 else ("floor" if zc < z_bot + 1.5 else "wall")
    rows.append([k + 1, pos, round(vol_vox, 2), round(abs(cut.volume), 2),
                 round(float(xs[iw[0]].mean()), 2), round(float(ys[iw[1]].mean()), 2),
                 round(float(zc), 2)])
    mm = cut
    mm.cell_data["block_id"] = np.full(mm.n_cells, k + 1, np.int32)
    rgb = {"crown": [200, 30, 30], "wall": [230, 140, 30], "floor": [90, 110, 190]}[pos]
    mm.cell_data["rgb"] = np.tile(np.array(rgb, np.uint8), (mm.n_cells, 1))
    parts.append(mm)
merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
merged.save(out_dir / "wedges_polyhedra.vtp")

# 복셀 기반 표면(과절단 없는 정확한 형상): pos_code 1=crown 2=wall 3=floor
pos_code = np.zeros(lab.shape, np.uint8)
for l in wedges:
    m = lab == l
    iw = np.nonzero(m)
    zc = zs[iw[2]].mean()
    pos_code[m] = 1 if zc > z_top - 1.5 else (3 if zc < z_bot + 1.5 else 2)
gridv = pv.ImageData(dimensions=np.array(lab.shape) + 1, spacing=(vs, vs, vs),
                     origin=(xs[0] - vs/2, ys[0] - vs/2, zs[0] - vs/2))
gridv.cell_data["pos_code"] = pos_code.ravel(order="F")
gridv.save(out_dir / "wedges_voxels.vti")
with open(out_dir / "wedge_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["wedge_id", "position", "vol_voxel_m3", "vol_poly_m3",
                "cx_m", "cy_m", "cz_m"])
    w.writerows(rows)
print(f"[recon] 다면체 {len(parts)}/{len(wedges)}")

# ── 장면: 상위20 평면(디스플레이용 대형 원판) + 터널 ─────────
disc_parts = []
theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
for k in range(N_TOP):
    n = normals[k]
    ref = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    pts = centers[k] + 18.0 * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    d = pv.PolyData(pts, faces=np.concatenate([[32], np.arange(32)])).triangulate()
    d.cell_data["set_id"] = np.full(d.n_cells, sets[k] or 0, np.int32)
    disc_parts.append(d)
discs = disc_parts[0].merge(disc_parts[1:])
discs.save(out_dir / f"top{N_TOP}_planes.vtp")

n = len(poly_yz)
ring0 = np.column_stack([np.full(n, x0), poly_yz])
ring1 = np.column_stack([np.full(n, x1), poly_yz])
quads = [[4, i, (i+1) % n, n+(i+1) % n, n+i] for i in range(n)]
pv.PolyData(np.vstack([ring0, ring1]), faces=np.array(quads).ravel()).save(
    out_dir / "tunnel.vtp")
print("[out]", out_dir)
