# -*- coding: utf-8 -*-
"""블록뷰 장면 기하 vtp 생성: tunnel_behind / face_cap / domain_box.

사용: python make_scene_geometry.py <forward_json> <x_back0> <out_dir>
  - tunnel_behind.vtp : 터널 단면을 x_back0 ~ 막장면(x0)까지 연장한 벽면(옆면만)
  - face_cap.vtp      : 막장면 x=x0 의 터널 단면 패치
  - domain_box.vtp    : 전방 생성 도메인 외곽 박스 (wireframe용)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

json_path, x_back0, out_dir = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

with open(json_path, encoding="utf-8") as f:
    dom = json.load(f)["meta"]["domain"]
poly_yz = np.asarray(dom["tunnel_polygon_yz_m"], dtype=float)
x0, x1 = [float(v) for v in dom["x_range_m"]]
yz = dom["yz_bounds_m"]
n = len(poly_yz)

# --- tunnel_behind: 옆면 쿼드 스트립 (x_back0 → x0) ---
ring0 = np.column_stack([np.full(n, x_back0), poly_yz])
ring1 = np.column_stack([np.full(n, x0), poly_yz])
pts = np.vstack([ring0, ring1])
quads = []
for i in range(n):
    j = (i + 1) % n
    quads.append([4, i, j, n + j, n + i])
tunnel = pv.PolyData(pts, faces=np.array(quads).ravel())
tunnel.save(out_dir / "tunnel_behind.vtp")

# --- face_cap: 막장면 폴리곤 패치 ---
cap = pv.PolyData(ring1, faces=np.concatenate([[n], np.arange(n)]))
cap = cap.triangulate()
cap.save(out_dir / "face_cap.vtp")

# --- domain_box: 전방 도메인 외곽 ---
box = pv.Box(bounds=(x0, x1, yz["y_min"], yz["y_max"], yz["z_min"], yz["z_max"]))
box.extract_all_edges().save(out_dir / "domain_box.vtp")

print(f"saved 3 vtp → {out_dir}  (tunnel x [{x_back0:.3f},{x0:.3f}], "
      f"domain x [{x0:.3f},{x1:.3f}])")
