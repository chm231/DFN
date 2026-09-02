"""도메인 DFN JSON([7] export_domain_dfn_json 출력) → 복셀 기반 블록 판정.

레거시 블록 탐지 모듈(_archive/block_detection/code 의 block_detector, tunnel_geometry)을
그대로 재사용한다. 좌표계: x = 터널 굴진 방향, 막장면 = x ≈ const 평면, 단위 m.

가정 (명시):
  - 해석 도메인(마지막 막장면 전방 구간)에 터널이 계속 굴착된다고 보고,
    터널 단면 폴리곤을 도메인 x 전 구간으로 연장한 마스크를 TUNNEL로 분류한다.
    (전방 굴진 시 터널 주변에 형성될 블록을 평가하는 목적)
  - 균열은 반지름 r 의 원판, 두께는 복셀 크기 * tol-factor (레거시와 동일 0.6).
  - CCA 기본 연결성은 6-connectivity (보수적; 모서리 접촉으로 블록이 합쳐지지 않음).
  - 블록 채택: 터널에 접하고, 도메인 외곽 경계에 닿지 않는 ROCK 컴포넌트.

실행 예:
  python -m dfn_analysis.detect_blocks_from_domain_json --json <dfn_domain.json> \
      --voxel 0.1 --connectivity 6 --out-prefix <outdir>/blocks_seed2026
  # 상위 N개 대형 균열만 사용 (클라이언트 안정성 해석 시나리오):
  ... --top-n 30
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# 레거시 블록 탐지 모듈 경로 추가 (수정 없이 재사용)
_ARCHIVE_CODE = Path(__file__).resolve().parents[2] / "_archive" / "block_detection" / "code"
sys.path.insert(0, str(_ARCHIVE_CODE))
import block_detector  # noqa: E402
import tunnel_geometry  # noqa: E402


def load_domain_json(json_path: Path, top_n: int | None, sets: list | None):
    """JSON에서 균열(center/normal/radius)과 도메인/터널 기하를 읽는다."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    dom = meta["domain"]
    fr = data["fractures"]
    if sets:
        fr = [f_ for f_ in fr if int(f_["set_id"]) in sets]
    centers_xyz = np.array([f_["center_xyz_m"] for f_ in fr], dtype=np.float32)
    normals_xyz = np.array([f_["normal_xyz"] for f_ in fr], dtype=np.float32)
    radii_m = np.array([f_["radius_m"] for f_ in fr], dtype=np.float32)
    labels = np.array([f_["label"] for f_ in fr])

    if top_n is not None and top_n < len(radii_m):
        idx = np.argsort(-radii_m)[:top_n]
        centers_xyz, normals_xyz, radii_m, labels = (
            centers_xyz[idx], normals_xyz[idx], radii_m[idx], labels[idx])

    x0, x1 = [float(v) for v in dom["x_range_m"]]
    yz = dom["yz_bounds_m"]
    domain_box = np.array([x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]])
    poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)
    return centers_xyz, normals_xyz, radii_m, labels, domain_box, poly_yz, meta


# ----------------------------------------------------------------------
# edge-cut 방식: 균열 = 두께 0의 분리면 (이웃 복셀 중심 연결을 끊음)
#   - 복셀 방식과 달리 균열 두께가 해상도에 종속되지 않고 암반 부피 손실이 없다.
#   - 두 복셀 중심을 잇는 선분이 원판을 가로지르면 그 연결(edge)만 제거.
# ----------------------------------------------------------------------

def compute_edge_cuts(grid_info, centers_xyz, normals_xyz, radii_m):
    """축방향(6-connectivity) edge 별로 원판 횡단 여부 boolean 배열 반환."""
    xs, ys, zs = grid_info["xs"], grid_info["ys"], grid_info["zs"]
    Nx, Ny, Nz = grid_info["shape"]
    vs = float(grid_info["voxel_size"])
    cutx = np.zeros((Nx - 1, Ny, Nz), dtype=bool)
    cuty = np.zeros((Nx, Ny - 1, Nz), dtype=bool)
    cutz = np.zeros((Nx, Ny, Nz - 1), dtype=bool)
    for i in range(len(radii_m)):
        c = centers_xyz[i].astype(np.float64)
        nrm = normals_xyz[i].astype(np.float64)
        nrm = nrm / np.linalg.norm(nrm)
        r = float(radii_m[i])
        ix0 = max(0, int(np.searchsorted(xs, c[0] - r - vs)))
        ix1 = min(Nx, int(np.searchsorted(xs, c[0] + r + vs)) + 1)
        iy0 = max(0, int(np.searchsorted(ys, c[1] - r - vs)))
        iy1 = min(Ny, int(np.searchsorted(ys, c[1] + r + vs)) + 1)
        iz0 = max(0, int(np.searchsorted(zs, c[2] - r - vs)))
        iz1 = min(Nz, int(np.searchsorted(zs, c[2] + r + vs)) + 1)
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue
        lx = xs[ix0:ix1].astype(np.float64) - c[0]
        ly = ys[iy0:iy1].astype(np.float64) - c[1]
        lz = zs[iz0:iz1].astype(np.float64) - c[2]
        # 복셀 중심의 평면 부호거리 D (a,b,c 로컬 격자)
        D = (lx[:, None, None] * nrm[0] + ly[None, :, None] * nrm[1]
             + lz[None, None, :] * nrm[2])
        ly2 = ly ** 2
        lz2 = lz ** 2
        r2 = r * r
        # 축별 edge: 부호 변화(평면 횡단) & 교차점의 중심거리 <= r (교차점은 평면 위
        # 이므로 3D 거리 = 면내 거리)
        for axis in range(3):
            if D.shape[axis] < 2:
                continue
            sl1 = [slice(None)] * 3
            sl2 = [slice(None)] * 3
            sl1[axis] = slice(None, -1)
            sl2[axis] = slice(1, None)
            d1, d2 = D[tuple(sl1)], D[tuple(sl2)]
            cross = (d1 > 0) != (d2 > 0)
            if not cross.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(cross, d1 / (d1 - d2), 0.0)
            if axis == 0:
                px = lx[:-1, None, None] + t * vs
                rho2 = px ** 2 + ly2[None, :, None] + lz2[None, None, :]
            elif axis == 1:
                py = ly[None, :-1, None] + t * vs
                rho2 = lx[:, None, None] ** 2 + py ** 2 + lz2[None, None, :]
            else:
                pz = lz[None, None, :-1] + t * vs
                rho2 = lx[:, None, None] ** 2 + ly2[None, :, None] + pz ** 2
            hit = cross & (rho2 <= r2)
            if axis == 0:
                cutx[ix0:ix1 - 1, iy0:iy1, iz0:iz1] |= hit
            elif axis == 1:
                cuty[ix0:ix1, iy0:iy1 - 1, iz0:iz1] |= hit
            else:
                cutz[ix0:ix1, iy0:iy1, iz0:iz1 - 1] |= hit
    return cutx, cuty, cutz


def run_cca_edgecut(rock_mask, cutx, cuty, cutz):
    """절단되지 않은 6-이웃 edge 그래프에서 연결성분. labels3d(-1=비암반), n_comp 반환."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n_rock = int(rock_mask.sum())
    idx = np.full(rock_mask.shape, -1, dtype=np.int32)
    idx[rock_mask] = np.arange(n_rock, dtype=np.int32)
    rows, cols = [], []
    for axis, cut in [(0, cutx), (1, cuty), (2, cutz)]:
        sl1 = [slice(None)] * 3
        sl2 = [slice(None)] * 3
        sl1[axis] = slice(None, -1)
        sl2[axis] = slice(1, None)
        ok = rock_mask[tuple(sl1)] & rock_mask[tuple(sl2)] & ~cut
        rows.append(idx[tuple(sl1)][ok])
        cols.append(idx[tuple(sl2)][ok])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(n_rock, n_rock))
    n_comp, comp = connected_components(graph, directed=False)
    labels3d = np.full(rock_mask.shape, -1, dtype=np.int32)
    labels3d[rock_mask] = comp
    print(f"  edge-cut CCA: rock {n_rock:,} voxels, 살아있는 edge {len(rows):,}, "
          f"{n_comp:,} 컴포넌트")
    return labels3d, n_comp


def filter_blocks_edgecut(labels3d, n_comp, tunnel_mask, grid_info, min_voxels):
    """터널 접촉 & 외곽 비접촉 컴포넌트 필터 + 통계 (레거시와 동일 기준)."""
    vs = float(grid_info["voxel_size"])
    counts = np.bincount(labels3d[labels3d >= 0], minlength=n_comp)
    boundary = set()
    for sl in (labels3d[0], labels3d[-1], labels3d[:, 0], labels3d[:, -1],
               labels3d[:, :, 0], labels3d[:, :, -1]):
        u = np.unique(sl)
        boundary.update(int(v) for v in u[u >= 0])
    # 터널 복셀과 6-이웃으로 맞닿은 암반 복셀
    adj = np.zeros(labels3d.shape, dtype=bool)
    tm = tunnel_mask
    adj[:-1] |= tm[1:];  adj[1:] |= tm[:-1]
    adj[:, :-1] |= tm[:, 1:];  adj[:, 1:] |= tm[:, :-1]
    adj[:, :, :-1] |= tm[:, :, 1:];  adj[:, :, 1:] |= tm[:, :, :-1]
    touch_lbl = labels3d[adj & (labels3d >= 0)]
    tunnel_touch = np.unique(touch_lbl)
    final = [int(l) for l in tunnel_touch
             if counts[l] >= min_voxels and int(l) not in boundary]
    xs, ys, zs = grid_info["xs"], grid_info["ys"], grid_info["zs"]
    blocks = []
    for lbl in final:
        mask = labels3d == lbl
        iw = np.nonzero(mask)
        blocks.append(dict(
            label=lbl, n_voxels=int(counts[lbl]),
            volume_m3=float(counts[lbl]) * vs ** 3,
            centroid=(float(xs[iw[0]].mean()), float(ys[iw[1]].mean()),
                      float(zs[iw[2]].mean())),
            contact_area_m2=float(np.sum(mask & adj)) * vs ** 2,
        ))
    blocks.sort(key=lambda d: d["volume_m3"], reverse=True)
    for rank, b in enumerate(blocks):
        b["rank"] = rank + 1
    print(f"  edge-cut 필터: 터널접촉 {len(tunnel_touch):,} → 최종 {len(blocks)}개 "
          f"(외곽접촉/소형 제외)")
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True, help="dfn_domain_*.json 경로")
    ap.add_argument("--voxel", type=float, default=0.1, help="복셀 크기 [m] (기본 0.1)")
    ap.add_argument("--connectivity", type=int, choices=[6, 26], default=6,
                    help="ROCK CCA 연결성 (기본 6 = 보수적)")
    ap.add_argument("--min-voxels", type=int, default=8, help="블록 최소 복셀 수 (기본 8)")
    ap.add_argument("--top-n", type=int, default=None,
                    help="반지름 상위 N개 균열만 사용 (안정성 해석 시나리오 모사)")
    ap.add_argument("--sets", type=int, nargs="*", default=None, help="사용할 절리군 (기본 전체)")
    ap.add_argument("--tol-factor", type=float, default=0.6, help="균열 두께 = voxel*tol (기본 0.6)")
    ap.add_argument("--method", choices=["voxel", "edgecut"], default="voxel",
                    help="voxel: 균열을 두께 있는 복셀로 (레거시). "
                         "edgecut: 균열을 두께 0의 분리면으로 (이웃 연결 절단; 해상도 비종속)")
    ap.add_argument("--out-prefix", required=True,
                    help="출력 프리픽스: <prefix>_blocks.csv / _summary.json / _labels.npz")
    args = ap.parse_args()

    centers_xyz, normals_xyz, radii_m, labels_arr, domain_box, poly_yz, meta = \
        load_domain_json(Path(args.json), args.top_n, args.sets)
    n_obs = int((labels_arr == "observed").sum())
    print(f"[input] 균열 {len(radii_m):,}개 (observed {n_obs}, unobserved {len(radii_m)-n_obs})"
          f"  r_max={radii_m.max():.2f} m" if len(radii_m) else "[input] 균열 0개")
    print(f"[domain] x[{domain_box[0]:.2f},{domain_box[1]:.2f}] "
          f"y[{domain_box[2]:.2f},{domain_box[3]:.2f}] z[{domain_box[4]:.2f},{domain_box[5]:.2f}]")

    # 터널을 도메인 x 전 구간으로 연장 (전방 굴착 가정)
    _, tunnel_mask, _, grid_info = tunnel_geometry.build_voxel_masks(
        poly_yz[:, 0], poly_yz[:, 1], domain_box,
        voxel_size=args.voxel, halo_dist=0.0,
        tunnel_xmin=float(domain_box[0]), tunnel_xmax=float(domain_box[1]),
    )
    tunnel_mask = np.asarray(block_detector.to_numpy(tunnel_mask))

    if args.method == "edgecut":
        rock_mask = ~tunnel_mask
        cutx, cuty, cutz = compute_edge_cuts(grid_info, centers_xyz, normals_xyz, radii_m)
        print(f"  절단 edge: x {int(cutx.sum()):,} / y {int(cuty.sum()):,} / z {int(cutz.sum()):,}")
        cca_labels, n_labels = run_cca_edgecut(rock_mask, cutx, cuty, cutz)
        blocks = filter_blocks_edgecut(cca_labels, n_labels, tunnel_mask, grid_info,
                                       args.min_voxels)
        # npz 저장용: 최종 블록만 남기고 1부터 라벨링 (0 = 배경)
        keep = np.array([b["label"] for b in blocks], dtype=np.int32)
        remap = np.zeros(n_labels, dtype=np.int32)
        remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
        cca_labels = np.where(cca_labels >= 0, remap[np.maximum(cca_labels, 0)], 0)
        for b in blocks:  # label을 재라벨 값으로 교체 (npz와 일치)
            b["label"] = int(remap[b["label"]])
    else:
        state, _owner = block_detector.classify_voxels(
            grid_info, centers_xyz, normals_xyz, radii_m, tunnel_mask,
            tol_factor=args.tol_factor,
        )
        cca_labels, n_labels = block_detector.run_cca(state, connectivity=args.connectivity)
        blocks = block_detector.filter_and_stat_blocks(
            cca_labels, n_labels, state, grid_info,
            min_voxels=args.min_voxels, connectivity=args.connectivity,
        )

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}_blocks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "label", "n_voxels", "volume_m3",
                    "centroid_x", "centroid_y", "centroid_z", "contact_area_m2"])
        for b in blocks:
            w.writerow([b["rank"], b["label"], b["n_voxels"], f"{b['volume_m3']:.6f}",
                        f"{b['centroid'][0]:.4f}", f"{b['centroid'][1]:.4f}",
                        f"{b['centroid'][2]:.4f}", f"{b['contact_area_m2']:.4f}"])

    # 블록 복셀 라벨(비교 분석용)과 요약 저장
    block_label_ids = np.array([b["label"] for b in blocks], dtype=np.int32)
    block_mask = np.isin(cca_labels, block_label_ids)
    np.savez_compressed(f"{prefix}_labels.npz",
                        labels=np.where(block_mask, cca_labels, 0).astype(np.int32),
                        origin=grid_info["origin"], voxel_size=args.voxel)

    vols = np.array([b["volume_m3"] for b in blocks])
    summary = dict(
        input_json=str(args.json), n_fractures_used=int(len(radii_m)), top_n=args.top_n,
        voxel_size_m=args.voxel, connectivity=args.connectivity, min_voxels=args.min_voxels,
        method=args.method,
        tol_factor=args.tol_factor, grid_shape=list(grid_info["shape"]),
        n_blocks=len(blocks),
        total_block_volume_m3=float(vols.sum()) if len(vols) else 0.0,
        max_block_volume_m3=float(vols.max()) if len(vols) else 0.0,
        median_block_volume_m3=float(np.median(vols)) if len(vols) else 0.0,
        top10_volumes_m3=[float(v) for v in vols[:10]],
        seed=meta.get("seed"),
    )
    with open(f"{prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"[out] {prefix}_blocks.csv / _summary.json / _labels.npz")
    print(f"[result] 블록 {len(blocks)}개, 총부피 {summary['total_block_volume_m3']:.3f} m³, "
          f"최대 {summary['max_block_volume_m3']:.3f} m³")


if __name__ == "__main__":
    main()
