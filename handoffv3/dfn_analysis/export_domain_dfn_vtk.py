"""도메인 DFN JSON([7] export_domain_dfn_json 출력) → ParaView용 VTK 파일 내보내기.

출력 (--out-dir):
  dfn_discs.vtp     : 균열 원판 폴리곤. cell data = set_id, observed(1/0), radius_m, fracture_id
  tunnel.vtp        : 터널 단면을 도메인 x 전 구간으로 연장한 표면
  domain_box.vtp    : 해석 도메인 외곽 박스 (wireframe용)
  wall_traces.vtp   : 터널 벽면 절리선 (있을 때)
  blocks.vti        : (선택, --blocks-npz) 블록 복셀 라벨 ImageData

ParaView 사용:
  dfn_discs.vtp 열기 → Coloring: set_id (Interpret Values As Categories 권장)
  observed만 보기: Threshold 필터, Scalars=observed, 범위 1~1
  blocks.vti 열기 → Threshold 필터, Scalars=block_label, 하한 1 → 블록만 표시

실행 예:
  python -m dfn_analysis.export_domain_dfn_vtk --json <dfn_domain.json> \
      --out-dir <outdir> [--blocks-npz <prefix_labels.npz>]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def build_disc_mesh(centers_xyz, normals_xyz, radii_m, n_sides=24):
    normals = np.asarray(normals_xyz, dtype=float)
    centers = np.asarray(centers_xyz, dtype=float)
    radii = np.asarray(radii_m, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    ref = np.where(np.abs(normals[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    u = np.cross(normals, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normals, u)
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    pts = centers[:, None, :] + radii[:, None, None] * (
        np.cos(theta)[None, :, None] * u[:, None, :]
        + np.sin(theta)[None, :, None] * v[:, None, :]
    )
    n = len(centers)
    faces = np.empty((n, n_sides + 1), dtype=np.int64)
    faces[:, 0] = n_sides
    faces[:, 1:] = np.arange(n * n_sides).reshape(n, n_sides)
    return pv.PolyData(pts.reshape(-1, 3), faces=faces.ravel())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True, help="dfn_domain_*.json 경로")
    ap.add_argument("--out-dir", required=True, help="VTK 출력 폴더")
    ap.add_argument("--blocks-npz", default=None,
                    help="detect_blocks_from_domain_json 출력 *_labels.npz (선택)")
    ap.add_argument("--top-n", type=int, default=None,
                    help="반지름 상위 N개 균열만 내보내기 (안정성 해석 시나리오)")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    dom = meta["domain"]
    fr = data["fractures"]
    if args.top_n is not None and args.top_n < len(fr):
        fr = sorted(fr, key=lambda f_: -f_["radius_m"])[:args.top_n]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 균열 원판 (cell data: set_id / observed / radius_m / fracture_id)
    discs = build_disc_mesh([f_["center_xyz_m"] for f_ in fr],
                            [f_["normal_xyz"] for f_ in fr],
                            [f_["radius_m"] for f_ in fr])
    discs.cell_data["set_id"] = np.array([f_["set_id"] for f_ in fr], dtype=np.int32)
    discs.cell_data["observed"] = np.array(
        [1 if f_["label"] == "observed" else 0 for f_ in fr], dtype=np.int8)
    discs.cell_data["radius_m"] = np.array([f_["radius_m"] for f_ in fr], dtype=np.float32)
    discs.cell_data["fracture_id"] = np.array([f_["id"] for f_ in fr], dtype=np.int64)
    discs.save(out / "dfn_discs.vtp")

    # 터널 표면 (도메인 x 전 구간 연장)
    poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)
    x0, x1 = [float(v) for v in dom["x_range_m"]]
    n = len(poly_yz)
    pts_xyz = np.column_stack([np.full(n, x0), poly_yz[:, 0], poly_yz[:, 1]])
    loop = pv.PolyData(pts_xyz, lines=np.concatenate([[n + 1], np.arange(n), [0]]))
    loop.extrude([x1 - x0, 0.0, 0.0], capping=False).save(out / "tunnel.vtp")

    # 도메인 박스
    yz = dom["yz_bounds_m"]
    pv.Box((x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"])).extract_all_edges() \
        .save(out / "domain_box.vtp")

    # 터널 벽면 절리선
    traces = data.get("tunnel_wall_traces", [])
    if traces:
        p0_xyz = np.array([t["p0_xyz_m"] for t in traces])
        p1_xyz = np.array([t["p1_xyz_m"] for t in traces])
        m = len(traces)
        tpts = np.empty((2 * m, 3))
        tpts[0::2], tpts[1::2] = p0_xyz, p1_xyz
        lines = np.column_stack([np.full(m, 2), np.arange(0, 2 * m, 2),
                                 np.arange(1, 2 * m, 2)]).astype(np.int64)
        tr = pv.PolyData(tpts, lines=lines.ravel())
        tr.cell_data["set_id"] = np.array([t["set_id"] for t in traces], dtype=np.int32)
        tr.save(out / "wall_traces.vtp")

    # 블록 복셀 (선택)
    if args.blocks_npz:
        z = np.load(args.blocks_npz)
        lab, origin, vs = z["labels"], z["origin"], float(z["voxel_size"])
        grid = pv.ImageData(dimensions=np.array(lab.shape) + 1,
                            spacing=(vs, vs, vs), origin=tuple(origin))
        grid.cell_data["block_label"] = lab.ravel(order="F")
        grid.save(out / "blocks.vti")

    n_obs = int(discs.cell_data["observed"].sum())
    print(f"[out] {out}  discs={discs.n_cells:,} (observed {n_obs}), "
          f"traces={len(traces)}, blocks={'yes' if args.blocks_npz else 'no'}")


if __name__ == "__main__":
    main()
