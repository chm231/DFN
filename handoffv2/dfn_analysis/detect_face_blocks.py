"""마지막 막장면(굴진면) 쐐기 블록 판정 + 다면체 재구성.

구성 (전방 도메인을 미굴착 암반으로 취급):
  - 암반 = 도메인 전체 (터널 연장 없음 — 아직 굴착 전)
  - 자유면 = 막장면 x = x0 평면 중 터널 단면 다각형 내부 패치 (실제 노출면)
  - 막장면 블록 = edge-cut CCA 컴포넌트 중
      ① 자유면 패치에 접하고
      ② 막장면의 다각형 밖 부분(뒤로 암반이 이어지는 절단 경계)에는 닿지 않고
      ③ 다른 도메인 외곽에도 닿지 않는 것
  - 각 블록은 경계 원판 평면 + 막장면 평면(+도메인 면)으로 정확한 다면체 재구성

--labels observed 면 관측 복원 균열만 사용(실측 증거 기반 쐐기), all 이면 전체 DFN.

실행 예:
  python -m dfn_analysis.detect_face_blocks --json example_io/dfn_domain_*.json \
      --voxel 0.05 --labels observed --out-prefix example_io/blocks_edgecut/face_obs
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dfn_analysis.detect_blocks_from_domain_json import (
    load_domain_json, compute_edge_cuts, run_cca_edgecut,
    tunnel_geometry, block_detector)
from dfn_analysis.cut_blocks_polyhedral import hull_of
from dfn_analysis.reconstruct_block_polyhedra import clip_hull


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True)
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="복셀(닫힘 판정 스케일 ε) [m]. 관측 균열은 소형이라 0.05 권장")
    ap.add_argument("--labels", choices=["observed", "all"], default="observed")
    ap.add_argument("--min-voxels", type=int, default=8)
    ap.add_argument("--ratio-max", type=float, default=2.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    fr = data["fractures"]
    if args.labels == "observed":
        fr = [f_ for f_ in fr if f_["label"] == "observed"]
    centers = np.array([f_["center_xyz_m"] for f_ in fr], dtype=np.float64)
    normals = np.array([f_["normal_xyz"] for f_ in fr], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    radii = np.array([f_["radius_m"] for f_ in fr], dtype=np.float64)
    dom = data["meta"]["domain"]
    x0, x1 = [float(v) for v in dom["x_range_m"]]
    yz = dom["yz_bounds_m"]
    box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
    poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)
    print(f"[input] 균열 {len(fr)}개 ({args.labels}), 막장면 x = {x0:.3f} m, ε = {args.voxel} m")

    # 격자: tunnel_mask 는 막장면 자유면 패치(다각형 내부) 판정용으로만 사용
    _, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
        poly_yz[:, 0], poly_yz[:, 1], box, voxel_size=args.voxel, halo_dist=0.0,
        tunnel_xmin=x0, tunnel_xmax=x1)
    tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))
    face_inside_yz = tunnel_mask[0]          # (Ny,Nz): 막장면에서 터널 단면 내부
    rock = np.ones(tunnel_mask.shape, dtype=bool)   # 전방은 미굴착 — 전부 암반
    vs = float(grid_info["voxel_size"])
    xs, ys, zs = grid_info["xs"], grid_info["ys"], grid_info["zs"]

    cutx, cuty, cutz = compute_edge_cuts(grid_info, centers, normals, radii)
    labels3d, ncomp = run_cca_edgecut(rock, cutx, cuty, cutz)
    counts = np.bincount(labels3d[labels3d >= 0], minlength=ncomp)

    face_touch = set(int(v) for v in np.unique(labels3d[0][face_inside_yz]))
    x0_outside = set(int(v) for v in np.unique(labels3d[0][~face_inside_yz]))
    other_bnd = set()
    for sl in (labels3d[-1], labels3d[:, 0], labels3d[:, -1],
               labels3d[:, :, 0], labels3d[:, :, -1]):
        u = np.unique(sl)
        other_bnd.update(int(v) for v in u[u >= 0])
    keep = [l for l in face_touch
            if counts[l] >= args.min_voxels and l not in x0_outside and l not in other_bnd]
    v_keep = [counts[l] * vs ** 3 for l in keep]
    print(f"[face] 자유면 접촉 컴포넌트 {len(face_touch):,} → 막장면 쐐기 블록 {len(keep)}개")
    if v_keep:
        print(f"[face] 부피 합 {sum(v_keep):.3f} m³, 최대 {max(v_keep):.3f}, "
              f"중앙값 {float(np.median(v_keep)):.4f}")

    # 다면체 재구성 (경계 원판 + 막장면 평면 + 도메인 면)
    disc_lo = centers - radii[:, None]
    disc_hi = centers + radii[:, None]
    rows, meshes = [], []
    for lbl in keep:
        iw = np.nonzero(labels3d == lbl)
        P = np.column_stack([xs[iw[0]], ys[iw[1]], zs[iw[2]]]).astype(np.float64)
        lo, hi = P.min(axis=0), P.max(axis=0)
        vol_vox = len(P) * vs ** 3
        m = 0.75 * vs
        corners = np.array([[a, b, c] for a in (lo[0]-m, hi[0]+m)
                            for b in (lo[1]-m, hi[1]+m) for c in (lo[2]-m, hi[2]+m)])
        hull = hull_of(corners)
        n_planes = 0
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
                n_planes += 1
        if hull is not None:  # 막장면 평면 + 도메인 면
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
        ratio = hull.volume / vol_vox
        flag = "ok" if ratio <= args.ratio_max else "open"
        rows.append(dict(label=lbl, n_planes=n_planes, vol_voxel=vol_vox,
                         vol_poly=hull.volume, ratio=ratio, flag=flag))
        meshes.append((lbl, hull, flag))

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "n_planes", "vol_voxel",
                                           "vol_poly", "ratio", "flag"])
        w.writeheader()
        for r in sorted(rows, key=lambda q: -q["vol_poly"]):
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    import pyvista as pv
    parts = []
    for k, (lbl, hull, flag) in enumerate(meshes):
        faces = np.column_stack([np.full(len(hull.simplices), 3), hull.simplices]).ravel()
        mm = pv.PolyData(hull.points, faces=faces)
        mm.cell_data["block_id"] = np.full(mm.n_cells, k + 1, dtype=np.int32)
        mm.cell_data["flag_open"] = np.full(mm.n_cells, int(flag == "open"), dtype=np.int8)
        parts.append(mm)
    if parts:
        merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        merged.save(f"{prefix}_polyhedra.vtp")
    n_ok = sum(1 for r in rows if r["flag"] == "ok")
    print(f"[recon] 다면체 재구성 {n_ok}/{len(keep)} (open {len(rows)-n_ok})")
    print(f"[out] {prefix}_summary.csv" + (f" / _polyhedra.vtp" if parts else " (블록 없음)"))


if __name__ == "__main__":
    main()
