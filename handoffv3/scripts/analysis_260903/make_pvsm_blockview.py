# -*- coding: utf-8 -*-
"""pvpython용: 정합 블록뷰 상태 생성 — 굴착영역과 DFN 생성 도메인 비중첩.

구성 (좌표 정합):
  x < 22.204 (기굴착)  : tunnel_behind(굴착 터널) + excavated_wall 블록(block_id 색)
  x = 22.204 (막장면)   : face_cap + 관측기반 막장면 쐐기(face_obs, 빨강)
  x > 22.204 (생성 도메인): 미굴착 암반 — 내부 닫힌 블록(unexcavated, 청회색 단일색)
                            터널 없음(굴착 전이므로)

사용: pvpython make_pvsm_blockview.py <handoffv2_dir> <out_pvsm>
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, GetActiveViewOrCreate, ColorBy,
    GetColorTransferFunction, SaveState, Render, ResetCamera,
)

h2 = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]
EIO = f"{h2}/example_io"

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

# --- 기굴착 터널 (막장면 뒤, x<=22.204) ---
tunnel = XMLPolyDataReader(FileName=[f"{EIO}/paraview/full/tunnel_behind.vtp"],
                           registrationName="tunnel_excavated")
tn = Show(tunnel, view)
tn.Representation = "Surface"
tn.AmbientColor = tn.DiffuseColor = [0.55, 0.55, 0.55]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.35

# --- 기굴착 구간 터널 벽 접촉 블록 (block_id 색) ---
wall = XMLPolyDataReader(
    FileName=[f"{EIO}/blocks_edgecut/excavated_wall_polyhedra_colored.vtp"],
    registrationName="excavated_wall_blocks")
w = Show(wall, view)
w.ColorArrayName = ["CELLS", "rgb"]
w.MapScalars = 0  # rgb 셀 배열을 그대로 사용 (블록별 고정색)
w.Opacity = 1.0

# --- 막장면 (자유면) ---
cap = XMLPolyDataReader(FileName=[f"{EIO}/paraview/full/face_cap.vtp"],
                        registrationName="face_cap")
c = Show(cap, view)
c.Representation = "Surface"
c.AmbientColor = c.DiffuseColor = [0.76, 0.70, 0.60]
c.ColorArrayName = ["CELLS", ""]
c.Opacity = 0.45

# --- 막장면 쐐기 (관측 기반 7개, 빨강) ---
wedge = XMLPolyDataReader(FileName=[f"{EIO}/blocks_edgecut/face_obs_polyhedra.vtp"],
                          registrationName="face_wedges_observed")
fw = Show(wedge, view)
fw.AmbientColor = fw.DiffuseColor = [0.79, 0.10, 0.11]
fw.ColorArrayName = ["CELLS", ""]
fw.Opacity = 1.0

# --- 생성 도메인 내부 닫힌 블록 (미굴착 기준, 청회색 단일색) ---
inner = XMLPolyDataReader(
    FileName=[f"{EIO}/blocks_edgecut/unexcavated_interior_polyhedra.vtp"],
    registrationName="domain_interior_blocks")
ib = Show(inner, view)
ib.AmbientColor = ib.DiffuseColor = [0.42, 0.53, 0.62]
ib.ColorArrayName = ["CELLS", ""]
ib.Opacity = 1.0

# --- 생성 도메인 박스 ---
box = XMLPolyDataReader(FileName=[f"{EIO}/paraview/full/domain_box.vtp"],
                        registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]
b.ColorArrayName = ["CELLS", ""]
b.LineWidth = 2

ResetCamera(view)
# 굴착 터널 안쪽에서 막장면·생성 도메인을 바라보는 준축방향 시점
view.CameraPosition = [-24.0, -22.0, 18.0]
view.CameraFocalPoint = [26.5, 1.35, 1.5]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
if len(sys.argv) > 3:  # 빌드 시점 렌더 확인용
    from paraview.simple import SaveScreenshot
    SaveScreenshot(sys.argv[3], view, ImageResolution=[1400, 900])
SaveState(out_pvsm)
print("saved:", out_pvsm)
