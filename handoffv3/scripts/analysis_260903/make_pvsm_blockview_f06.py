# -*- coding: utf-8 -*-
"""pvpython용: 1~6면 시나리오 블록뷰 상태 생성 (막장면 x=11.368).

구성 (좌표 정합, 굴착영역과 생성 도메인 비중첩):
  x < 11.368 (기굴착)  : tunnel_behind + excavated_wall 블록(블록별 rgb)
  x = 11.368 (막장면)   : face_cap + 관측기반 막장면 쐐기(빨강)
  x > 11.368 (생성 도메인): 미굴착 암반 내부 닫힌 블록(청회색) + 도메인 박스

사용: pvpython make_pvsm_blockview_f06.py <handoffv2_dir> <out_pvsm> [screenshot]
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, GetActiveViewOrCreate,
    SaveState, Render, ResetCamera,
)

h2 = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]
EIO = f"{h2}/example_io"
GEO = f"{EIO}/paraview/f06"
BLK = f"{EIO}/blocks_edgecut/f06"

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

tunnel = XMLPolyDataReader(FileName=[f"{GEO}/tunnel_behind.vtp"],
                           registrationName="tunnel_excavated")
tn = Show(tunnel, view)
tn.Representation = "Surface"
tn.AmbientColor = tn.DiffuseColor = [0.55, 0.55, 0.55]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.35

wall = XMLPolyDataReader(
    FileName=[f"{BLK}/excavated_wall_polyhedra_colored.vtp"],
    registrationName="excavated_wall_blocks")
w = Show(wall, view)
w.ColorArrayName = ["CELLS", "rgb"]
w.MapScalars = 0  # rgb 셀 배열 그대로 (블록별 고정색)
w.Opacity = 1.0

cap = XMLPolyDataReader(FileName=[f"{GEO}/face_cap.vtp"],
                        registrationName="face_cap")
c = Show(cap, view)
c.Representation = "Surface"
c.AmbientColor = c.DiffuseColor = [0.76, 0.70, 0.60]
c.ColorArrayName = ["CELLS", ""]
c.Opacity = 0.45

wedge = XMLPolyDataReader(FileName=[f"{BLK}/face_obs_polyhedra.vtp"],
                          registrationName="face_wedges_observed")
fw = Show(wedge, view)
fw.AmbientColor = fw.DiffuseColor = [0.79, 0.10, 0.11]
fw.ColorArrayName = ["CELLS", ""]
fw.Opacity = 1.0

inner = XMLPolyDataReader(
    FileName=[f"{BLK}/unexcavated_interior_polyhedra.vtp"],
    registrationName="domain_interior_blocks")
ib = Show(inner, view)
ib.AmbientColor = ib.DiffuseColor = [0.42, 0.53, 0.62]
ib.ColorArrayName = ["CELLS", ""]
ib.Opacity = 1.0

box = XMLPolyDataReader(FileName=[f"{GEO}/domain_box.vtp"],
                        registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]
b.ColorArrayName = ["CELLS", ""]
b.LineWidth = 2

ResetCamera(view)
# 굴착 터널 뒤쪽에서 막장면(x=11.368)·생성 도메인을 바라보는 시점
view.CameraPosition = [-30.0, -24.0, 19.0]
view.CameraFocalPoint = [15.7, 1.35, 1.5]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
if len(sys.argv) > 3:
    from paraview.simple import SaveScreenshot
    SaveScreenshot(sys.argv[3], view, ImageResolution=[1400, 900])
SaveState(out_pvsm)
print("saved:", out_pvsm)
