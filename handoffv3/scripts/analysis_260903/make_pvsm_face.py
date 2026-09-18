# -*- coding: utf-8 -*-
"""pvpython용: 막장면 쐐기 블록 뷰 상태(.pvsm) 생성.

사용: pvpython make_pvsm_face.py <wedges_vtp> <discs_vtp> <tunnel_vtp> <out_pvsm>
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, GetActiveViewOrCreate, ColorBy,
    GetColorTransferFunction, Threshold, Clip, SaveState, Render,
)

wedges_vtp, discs_vtp, tunnel_vtp, out_pvsm = [a.replace("\\", "/") for a in sys.argv[1:5]]

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

# 막장면 쐐기 (관측 균열 기반)
w = XMLPolyDataReader(FileName=[wedges_vtp], registrationName="face_wedges")
ws = Show(w, view)
ColorBy(ws, ("CELLS", "block_id"))
lut = GetColorTransferFunction("block_id")
lut.ApplyPreset("Turbo", True)
ws.Opacity = 1.0

# 관측 균열만 (Threshold observed==1)
discs = XMLPolyDataReader(FileName=[discs_vtp], registrationName="dfn_discs_src")
obs = Threshold(Input=discs, registrationName="observed_discs")
obs.Scalars = ["CELLS", "observed"]
obs.LowerThreshold = 1
obs.UpperThreshold = 1
os_ = Show(obs, view)
os_.ColorArrayName = ["CELLS", ""]
os_.AmbientColor = os_.DiffuseColor = [0.62, 0.70, 0.78]
os_.Opacity = 0.25

# 확률 생성(unobserved) 균열 — 막장면 +2 m 슬랩만 기본 표시
uno = Threshold(Input=discs, registrationName="unobserved_discs_all")
uno.Scalars = ["CELLS", "observed"]
uno.LowerThreshold = 0
uno.UpperThreshold = 0
us_all = Show(uno, view)
ColorBy(us_all, ("CELLS", "set_id"))
slut = GetColorTransferFunction("set_id")
slut.InterpretValuesAsCategories = 1
slut.Annotations = ["1", "set 1", "2", "set 2", "3", "set 3", "4", "set 4"]
slut.IndexedColors = [0.122, 0.467, 0.706, 0.173, 0.627, 0.173,
                      1.000, 0.498, 0.055, 0.580, 0.404, 0.741]
us_all.Opacity = 0.04
us_all.Visibility = 0  # 전체(19,434개)는 기본 숨김 — 눈 아이콘으로 켜기

near = Clip(Input=uno, registrationName="unobserved_near_face")
near.ClipType = "Plane"
near.ClipType.Origin = [24.2, 0.0, 0.0]   # 막장면 +2 m
near.ClipType.Normal = [1.0, 0.0, 0.0]    # x <= 24.2 유지
un = Show(near, view)
ColorBy(un, ("CELLS", "set_id"))
un.Opacity = 0.15

# 터널 (기굴착 구간 참고용)
tun = XMLPolyDataReader(FileName=[tunnel_vtp], registrationName="tunnel")
t = Show(tun, view)
t.ColorArrayName = ["CELLS", ""]
t.AmbientColor = t.DiffuseColor = [0.5, 0.5, 0.55]
t.Opacity = 0.2

# 카메라: 터널 안쪽에서 막장면을 정면으로
view.CameraPosition = [8.2, 1.35, 1.3]
view.CameraFocalPoint = [22.2, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveState(out_pvsm)
print("saved:", out_pvsm)
