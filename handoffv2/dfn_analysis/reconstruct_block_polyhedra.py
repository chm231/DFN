"""하이브리드 블록 재구성: edge-cut CCA로 찾은 '진짜 닫힌 블록'의 정확한 다면체 복원.

절차:
  1) edge-cut CCA (detect_blocks_from_domain_json --method edgecut 와 동일)로
     완전히 둘러싸인 블록의 복셀 집합을 얻는다.
  2) 각 블록에 대해 그 경계를 이루는 원판(경계 평면)을 판별한다:
     - 블록 복셀 중심의 95% 이상이 평면 한쪽에 있고,
     - 평면이 블록 경계에 접해 있으며(최소 |거리| <= 1.2*복셀),
     - 접촉부가 원판 반지름 안에 드는 원판.
  3) 블록 복셀 bbox(+여유)에서 시작해 경계 원판 평면·인접 터널 벽면·도메인 면으로
     순차 클리핑 → 볼록 다면체 복원 (계단 없는 정확한 쐐기 기하).
  4) 검증: 다면체 부피 vs 복셀 부피 비율(ratio) 기록. 비율이 과도하면(기본 >2)
     경계 평면 누락으로 보고 flag=open 처리.

한계: 볼록 재구성이므로 블록 안으로 파고들다 끝나는 슬릿(부분 절입)과 비볼록
형상은 외곽 형상으로 근사된다. 터널 벽은 폴리곤 변 평면으로 국소 근사.

실행 예:
  python -m dfn_analysis.reconstruct_block_polyhedra \
      --json example_io/dfn_domain_*.json --voxel 0.1 \
      --out-prefix example_io/blocks_edgecut/poly168 [--include-interior]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from dfn_analysis.detect_blocks_from_domain_json import (
    load_domain_json, compute_edge_cuts, run_cca_edgecut,
    tunnel_geometry, block_detector)
from dfn_analysis.cut_blocks_polyhedral import hull_of, hull_edges


def clip_hull(hull, p0_xyz, n_xyz, eps=1e-9):
    """볼록 hull에서 (x-p0)·n <= 0 쪽만 남긴다. 전부 잘리면 None."""
    pts = hull.points
    d = (pts - p0_xyz) @ n_xyz
    if d.max() <= eps:
        return hull
    if d.min() >= -eps:
        return None
    ipts = []
    for a, b in hull_edges(hull):
        da, db = d[a], d[b]
        if (da > eps and db < -eps) or (da < -eps and db > eps):
            t = da / (da - db)
            ipts.append(pts[a] + t * (pts[b] - pts[a]))
    keep = pts[d <= eps]
    allp = np.vstack([keep, np.asarray(ipts)]) if ipts else keep
    return hull_of(allp)


def tunnel_edge_planes(poly_yz):
    """터널 폴리곤 각 변의 (기준점 yz, 바깥쪽 법선 yz) 목록. CCW 보장."""
    area = 0.0
    n = len(poly_yz)
    for i in range(n):
        a, b = poly_yz[i], poly_yz[(i + 1) % n]
        area += a[0] * b[1] - b[0] * a[1]
    if area < 0:
        poly_yz = poly_yz[::-1]
    cen = poly_yz.mean(axis=0)
    planes = []
    for i in range(n):
        a, b = poly_yz[i], poly_yz[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]])
        nrm /= np.linalg.norm(nrm)
        if np.dot((a + b) / 2 - cen, nrm) < 0:
            nrm = -nrm  # 바깥쪽으로
        planes.append((a.copy(), nrm, b.copy()))
    return planes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True)
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--min-voxels", type=int, default=8)
    ap.add_argument("--include-interior", action="store_true",
                    help="터널 비접촉 내부 블록도 재구성 (기본: 터널 접촉만)")
    ap.add_argument("--ratio-max", type=float, default=2.0,
                    help="다면체/복셀 부피비 상한 — 초과 시 flag=open (기본 2.0)")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    centers, normals, radii, _, box, poly_yz, _ = load_domain_json(Path(args.json), None, None)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    _, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
        poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=args.voxel, halo_dist=0.0,
        tunnel_xmin=float(box[0]), tunnel_xmax=float(box[1]))
    tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
    rock = ~tunnel_mask
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
    adj = np.zeros(labels3d.shape, bool)
    tm = tunnel_mask
    adj[:-1] |= tm[1:]; adj[1:] |= tm[:-1]
    adj[:, :-1] |= tm[:, 1:]; adj[:, 1:] |= tm[:, :-1]
    adj[:, :, :-1] |= tm[:, :, 1:]; adj[:, :, 1:] |= tm[:, :, :-1]
    tunnel_touch = set(int(v) for v in np.unique(labels3d[adj & (labels3d >= 0)]))

    targets = [l for l in range(ncomp)
               if counts[l] >= args.min_voxels and l not in boundary
               and (args.include_interior or l in tunnel_touch)]
    print(f"[target] 재구성 대상 블록 {len(targets)}개 "
          f"({'터널접촉+내부' if args.include_interior else '터널 접촉만'})")

    # 원판 AABB (블록별 후보 선별용)
    disc_lo = centers - radii[:, None]
    disc_hi = centers + radii[:, None]
    t_planes = tunnel_edge_planes(np.asarray(poly_yz, dtype=float))
    x0, x1, y0, y1, z0, z1 = box

    rows, meshes = [], []
    n_open = 0
    for lbl in targets:
        iw = np.nonzero(labels3d == lbl)
        P = np.column_stack([xs[iw[0]], ys[iw[1]], zs[iw[2]]]).astype(np.float64)
        lo, hi = P.min(axis=0), P.max(axis=0)
        vol_vox = len(P) * vs ** 3
        # 시작 다면체: 복셀 bbox + 여유
        m = 0.75 * vs
        corners = np.array([[a, b, c] for a in (lo[0]-m, hi[0]+m)
                            for b in (lo[1]-m, hi[1]+m) for c in (lo[2]-m, hi[2]+m)])
        hull = hull_of(corners)

        # (1) 경계 원판 평면 클리핑
        cand = np.nonzero(np.all(disc_hi >= lo - vs, axis=1)
                          & np.all(disc_lo <= hi + vs, axis=1))[0]
        n_planes = 0
        for j in cand:
            d = (P - centers[j]) @ normals[j]
            pos_frac = float((d > 0).mean())
            if min(pos_frac, 1 - pos_frac) > 0.05:
                continue  # 블록 관통(슬릿) — 볼록 재구성에서는 무시
            absd = np.abs(d)
            if absd.min() > 1.2 * vs:
                continue  # 경계에 안 닿음
            near = absd <= 1.2 * vs
            inplane2 = np.sum((P[near] - centers[j]) ** 2, axis=1) - d[near] ** 2
            if not (inplane2 <= (radii[j] + vs) ** 2).any():
                continue  # 접촉부가 원판 밖
            sgn = 1.0 if pos_frac >= 0.5 else -1.0
            new = clip_hull(hull, centers[j], -sgn * normals[j])
            if new is not None:
                hull = new
                n_planes += 1
        # (2) 인접 터널 벽면 클리핑 (암반쪽 유지)
        if hull is not None:
            for a_yz, n_out, b_yz in t_planes:
                seg_lo = np.minimum(a_yz, b_yz) - 2 * vs
                seg_hi = np.maximum(a_yz, b_yz) + 2 * vs
                if (hi[1] < seg_lo[0] or lo[1] > seg_hi[0]
                        or hi[2] < seg_lo[1] or lo[2] > seg_hi[1]):
                    continue
                p0 = np.array([lo[0], a_yz[0], a_yz[1]])
                new = clip_hull(hull, p0, np.array([0.0, -n_out[0], -n_out[1]]))
                if new is not None:
                    hull = new
        # (3) 도메인 면 클리핑
        if hull is not None:
            for p0, nrm in [((x1, 0, 0), (1, 0, 0)), ((x0, 0, 0), (-1, 0, 0)),
                            ((0, y1, 0), (0, 1, 0)), ((0, y0, 0), (0, -1, 0)),
                            ((0, 0, z1), (0, 0, 1)), ((0, 0, z0), (0, 0, -1))]:
                new = clip_hull(hull, np.array(p0, float), np.array(nrm, float))
                if new is None:
                    hull = None
                    break
                hull = new
        if hull is None:
            n_open += 1
            continue
        ratio = hull.volume / vol_vox
        flag = "ok" if ratio <= args.ratio_max else "open"
        if flag == "open":
            n_open += 1
        rows.append(dict(label=lbl, n_planes=n_planes, vol_voxel=vol_vox,
                         vol_poly=hull.volume, ratio=ratio, flag=flag,
                         tunnel_contact=int(lbl in tunnel_touch)))
        meshes.append((lbl, hull, flag, lbl in tunnel_touch))

    ok = [r for r in rows if r["flag"] == "ok"]
    ratios = np.array([r["ratio"] for r in ok])
    print(f"[recon] 성공 {len(ok)} / 대상 {len(targets)} (open/실패 {n_open})")
    if len(ratios):
        print(f"[recon] 부피비(다면체/복셀): 중앙값 {np.median(ratios):.3f}, "
              f"p10 {np.percentile(ratios,10):.3f}, p90 {np.percentile(ratios,90):.3f}")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "n_planes", "vol_voxel",
                                           "vol_poly", "ratio", "flag", "tunnel_contact"])
        w.writeheader()
        for r in sorted(rows, key=lambda q: -q["vol_poly"]):
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    import pyvista as pv
    parts = []
    for k, (lbl, hull, flag, tc) in enumerate(meshes):
        faces = np.column_stack([np.full(len(hull.simplices), 3), hull.simplices]).ravel()
        mm = pv.PolyData(hull.points, faces=faces)
        mm.cell_data["block_id"] = np.full(mm.n_cells, k + 1, dtype=np.int32)
        mm.cell_data["flag_open"] = np.full(mm.n_cells, int(flag == "open"), dtype=np.int8)
        mm.cell_data["tunnel_contact"] = np.full(mm.n_cells, int(tc), dtype=np.int8)
        parts.append(mm)
    if parts:
        merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        merged.save(f"{prefix}_polyhedra.vtp")
    print(f"[out] {prefix}_summary.csv / _polyhedra.vtp")


if __name__ == "__main__":
    main()
