# -*- coding: utf-8 -*-
"""pvpython용: '연결 모암 + 완전 둘러싸인 블록' 뷰 상태(.pvsm) 생성.

사용: pvpython make_pvsm_matrix.py <vti> <discs_vtp> <tunnel_vtp> <box_vtp> <out_pvsm>
구성:
  - matrix  : region==1 Threshold -> Clip(x 평면, 드래그 가능) -> 회색 반투명
  - blocks  : region==2 Threshold -> block_id 색 (완전 둘러싸인 블록 168개)
  - discs   : 균열 원판 (아주 흐리게)
  - tunnel/box
"""
import sys

from paraview.simple import (
    XMLImageDataReader, XMLPolyDataReader, Show, GetActiveViewOrCreate,
    ColorBy, GetColorTransferFunction, Threshold, Clip, SaveState, Render,
    ResetCamera,
)

vti, discs_vtp, tunnel_vtp, box_vtp, out_pvsm = [a.replace("\\", "/") for a in sys.argv[1:6]]
poly_vtp = sys.argv[6].replace("\\", "/") if len(sys.argv) > 6 else None
face_wedges_vtp = sys.argv[7].replace("\\", "/") if len(sys.argv) > 7 else None
face_cap_vtp = sys.argv[8].replace("\\", "/") if len(sys.argv) > 8 else None

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

vol = XMLImageDataReader(FileName=[vti], registrationName="edgecut_volume")

# --- 연결 모암 (region==1) + 클립 ---
mat = Threshold(Input=vol, registrationName="matrix_rock")
mat.Scalars = ["CELLS", "region"]
mat.LowerThreshold = 1
mat.UpperThreshold = 1
clip = Clip(Input=mat, registrationName="matrix_clip")
clip.ClipType = "Plane"
clip.ClipType.Origin = [27.2, 1.35, 1.3]
clip.ClipType.Normal = [1.0, 0.0, 0.0]
mshow = Show(clip, view)
mshow.ColorArrayName = ["CELLS", ""]
mshow.AmbientColor = mshow.DiffuseColor = [0.72, 0.70, 0.66]
mshow.Opacity = 0.5

# --- 터널 접촉 블록 (region==2) ---
blk = Threshold(Input=vol, registrationName="blocks_tunnel_contact")
blk.Scalars = ["CELLS", "region"]
blk.LowerThreshold = 2
blk.UpperThreshold = 2
bshow = Show(blk, view)
ColorBy(bshow, ("CELLS", "block_id"))
lut = GetColorTransferFunction("block_id")
lut.ApplyPreset("Turbo", True)
bshow.Opacity = 1.0

# --- 터널 비접촉 내부 블록 (region==3) ---
ib = Threshold(Input=vol, registrationName="blocks_interior")
ib.Scalars = ["CELLS", "region"]
ib.LowerThreshold = 3
ib.UpperThreshold = 3
ishow = Show(ib, view)
ishow.ColorArrayName = ["CELLS", ""]
ishow.AmbientColor = ishow.DiffuseColor = [0.42, 0.55, 0.64]  # 단일색 (청회색)
ishow.Opacity = 0.85

# --- 하이브리드 재구성 다면체 (정확한 쐐기 기하) ---
if poly_vtp:
    ph = XMLPolyDataReader(FileName=[poly_vtp], registrationName="blocks_polyhedra_src")
    # 터널 접촉 다면체: block_id 색
    pt = Threshold(Input=ph, registrationName="poly_tunnel_contact")
    pt.Scalars = ["CELLS", "tunnel_contact"]
    pt.LowerThreshold = 1
    pt.UpperThreshold = 1
    pshow = Show(pt, view)
    ColorBy(pshow, ("CELLS", "block_id"))
    plut = GetColorTransferFunction("block_id")
    plut.ApplyPreset("Turbo", True)
    pshow.Opacity = 1.0
    # 내부 다면체: 단일색 (청회색)
    pi = Threshold(Input=ph, registrationName="poly_interior")
    pi.Scalars = ["CELLS", "tunnel_contact"]
    pi.LowerThreshold = 0
    pi.UpperThreshold = 0
    pishow = Show(pi, view)
    pishow.ColorArrayName = ["CELLS", ""]
    pishow.AmbientColor = pishow.DiffuseColor = [0.42, 0.55, 0.64]
    pishow.Opacity = 0.85
    # 같은 블록의 복셀 버전은 기본 숨김
    bshow.Visibility = 0
    ishow.Visibility = 0

# --- 균열 원판: 확률 생성(unobserved) + 관측 복원(observed) 중첩 ---
discs = XMLPolyDataReader(FileName=[discs_vtp], registrationName="dfn_discs_src")
slut = GetColorTransferFunction("set_id")
slut.InterpretValuesAsCategories = 1
slut.Annotations = ["1", "set 1", "2", "set 2", "3", "set 3", "4", "set 4"]
slut.IndexedColors = [0.122, 0.467, 0.706, 0.173, 0.627, 0.173,
                      1.000, 0.498, 0.055, 0.580, 0.404, 0.741]
# 확률 생성 균열망 (19,434개) — 도메인 전체, 흐리게
uno = Threshold(Input=discs, registrationName="stochastic_discs")
uno.Scalars = ["CELLS", "observed"]
uno.LowerThreshold = 0
uno.UpperThreshold = 0
ud = Show(uno, view)
ColorBy(ud, ("CELLS", "set_id"))
ud.Opacity = 0.06
# 막장면에서 복원된 관측 균열 (183개) — 뚜렷하게
obs = Threshold(Input=discs, registrationName="observed_discs")
obs.Scalars = ["CELLS", "observed"]
obs.LowerThreshold = 1
obs.UpperThreshold = 1
od = Show(obs, view)
ColorBy(od, ("CELLS", "set_id"))
od.Opacity = 0.9

# --- 막장면: 벽면 캡 + 막장면 쐐기 ---
if face_cap_vtp:
    fc = XMLPolyDataReader(FileName=[face_cap_vtp], registrationName="face_surface")
    fcs = Show(fc, view)
    fcs.ColorArrayName = ["CELLS", ""]
    fcs.AmbientColor = fcs.DiffuseColor = [0.62, 0.57, 0.50]  # 암갈색 벽면
    fcs.Opacity = 0.45
if face_wedges_vtp:
    fw = XMLPolyDataReader(FileName=[face_wedges_vtp], registrationName="face_wedges")
    fws = Show(fw, view)
    fws.ColorArrayName = ["CELLS", ""]
    fws.AmbientColor = fws.DiffuseColor = [0.86, 0.16, 0.30]  # 진홍 — 막장면 쐐기 강조
    fws.Opacity = 1.0

# --- 터널 / 도메인 박스 ---
tun = XMLPolyDataReader(FileName=[tunnel_vtp], registrationName="tunnel")
t = Show(tun, view)
t.ColorArrayName = ["CELLS", ""]
t.AmbientColor = t.DiffuseColor = [0.45, 0.45, 0.5]
t.Opacity = 0.3
box = XMLPolyDataReader(FileName=[box_vtp], registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.ColorArrayName = ["CELLS", ""]
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]

ResetCamera(view)
view.CameraPosition = [62.0, -38.0, 33.0]
view.CameraFocalPoint = [27.2, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveState(out_pvsm)
print("saved:", out_pvsm)
