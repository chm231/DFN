# -*- coding: utf-8 -*-
"""파이프라인(국소) 좌표 산출물 → world(원자료 측량) 좌표 변환 유틸리티.

이중좌표계 규약: 변환기가 p_pipeline = R @ p_world (순수 회전)로 좌표를 만들므로
world 복원은 p_world = p_pipeline @ R (행벡터 기준).

회전 R은 다음 우선순위로 찾는다:
  1) --diagnostics 로 지정한 conversion_diagnostics.json
  2) --json 입력의 meta.coordinate_system.world_transform (신규 export가 자동 포함)

사용 예:
  # 도메인 JSON을 world 좌표로 변환
  python scripts/pipeline_to_world.py --json in.json --out-json in_world.json \
      [--diagnostics demo_output/dfm_demo/conversion_diagnostics.json]
  # vtp/vti 등 VTK 파일의 점 좌표 변환
  python scripts/pipeline_to_world.py --vtk in.vtp --out-vtk in_world.vtp \
      --diagnostics <conversion_diagnostics.json>
"""
import argparse
import json
import sys

import numpy as np


def load_rotation(diag_path, doc=None):
    cs = None
    if diag_path:
        with open(diag_path, encoding="utf-8") as fh:
            cs = json.load(fh).get("coordinate_system")
    elif doc is not None:
        cs = (doc.get("meta", {}).get("coordinate_system", {}) or {}).get("world_transform")
    if not cs or "world_to_pipeline_rotation" not in cs:
        sys.exit("world_to_pipeline_rotation 을 찾을 수 없음 — --diagnostics 를 지정하거나 "
                 "신규 변환기(이중좌표계 기록)로 만든 JSON을 입력하세요.")
    R = np.asarray(cs["world_to_pipeline_rotation"], dtype=float)
    assert R.shape == (3, 3)
    return R


def to_world(pts, R):
    return np.asarray(pts, dtype=float) @ R  # p_world = p_pipeline @ R


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="도메인 DFN JSON (파이프라인 좌표)")
    ap.add_argument("--out-json", help="world 좌표 JSON 출력 경로")
    ap.add_argument("--vtk", help="vtp/vti 등 VTK 파일 (파이프라인 좌표)")
    ap.add_argument("--out-vtk", help="world 좌표 VTK 출력 경로")
    ap.add_argument("--diagnostics", default=None,
                    help="conversion_diagnostics.json 경로 (회전 R 출처)")
    args = ap.parse_args()

    if args.json:
        with open(args.json, encoding="utf-8") as fh:
            doc = json.load(fh)
        R = load_rotation(args.diagnostics, doc)
        n = 0
        for f in doc.get("fractures", []):
            f["center_xyz_m"] = [round(float(v), 6) for v in
                                 to_world(f["center_xyz_m"], R)]
            f["normal_xyz"] = [round(float(v), 6) for v in
                               to_world(f["normal_xyz"], R)]
            n += 1
        meta = doc.setdefault("meta", {})
        cs = meta.setdefault("coordinate_system", {})
        cs["frame"] = "world(원자료 측량좌표)"
        cs["note"] = ("pipeline_to_world.py 로 회전 복원 적용됨. "
                      "meta.domain 의 x_range 등 스칼라 필드는 파이프라인 좌표 기준 그대로임 — "
                      "world 도메인 모서리는 domain.box_corners_world 참조.")
        dom = meta.get("domain") or {}
        yz = dom.get("yz_bounds_m")
        if yz and dom.get("x_range_m"):
            x0, x1 = dom["x_range_m"]
            corners = [[x, y, z] for x in (x0, x1)
                       for y in (yz["y_min"], yz["y_max"])
                       for z in (yz["z_min"], yz["z_max"])]
            dom["box_corners_world"] = [[round(float(v), 4) for v in to_world(c, R)]
                                        for c in corners]
        out = args.out_json or args.json.replace(".json", "_world.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"[json] 균열 {n:,}개 world 변환 → {out}")

    if args.vtk:
        import pyvista as pv
        R = load_rotation(args.diagnostics)
        m = pv.read(args.vtk)
        ext = args.vtk.rsplit(".", 1)[1]
        if isinstance(m, pv.ImageData):
            m = m.cast_to_structured_grid()  # 회전하면 축정렬 격자가 아니게 됨
            ext = "vts"
        m.points = to_world(m.points, R)
        out = args.out_vtk or args.vtk.rsplit(".", 1)[0] + "_world." + ext
        m.save(out)
        print(f"[vtk] 점 {m.n_points:,}개 world 변환 → {out}")

    if not args.json and not args.vtk:
        sys.exit("--json 또는 --vtk 중 하나는 필요합니다.")


if __name__ == "__main__":
    main()
