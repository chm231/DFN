# -*- coding: utf-8 -*-
"""pvpython용: '현재 상태' 뷰 — 기굴착 터널(막장면 뒤) + 미굴착 전방 도메인.

사용: pvpython make_pvsm_current.py <unexc_vti> <face_wedges_vtp> <interior_poly_vtp>
     <discs_vtp> <face_cap_vtp> <tunnel_behind_vtp> <box_vtp> <out_pvsm>

전방 도메인(막장면 +10 m)은 아직 굴착되지 않은 꽉 찬 암반이며, 그 안에
확률 균열망 + 관측 복원 균열 + 닫힌 블록이 들어 있다. 굴착된 터널은 막장면
뒤(-x)에만 존재한다.
"""
import sys

from paraview.simple import (
    XMLImageDataReader, XMLPolyDataReader, Show, GetActiveViewOrCreate,
    ColorBy, GetColorTransferFunction, Threshold, Clip, SaveState, Render,
)

(unexc_vti, wedges_vtp, interior_vtp, discs_vtp, cap_vtp, behind_vtp,
 box_vtp, out_pvsm) = [a.replace("\\", "/") for a in sys.argv[1:9]]
poly_all_vtp = sys.argv[9].replace("\\", "/") if len(sys.argv) > 9 else None
excavated_vtp = sys.argv[10].replace("\\", "/") if len(sys.argv) > 10 else None

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

# 미굴착 모암 (region==1) + 클립 — 터널 구멍 없음
vol = XMLImageDataReader(FileName=[unexc_vti], registrationName="unexcavated_volume")
mat = Threshold(Input=vol, registrationName="matrix_rock")
mat.Scalars = ["CELLS", "region"]
mat.LowerThreshold = 1
mat.UpperThreshold = 1
clip = Clip(Input=mat, registrationName="matrix_clip")
clip.ClipType = "Plane"
clip.ClipType.Origin = [27.2, 1.35, 1.3]
clip.ClipType.Normal = [1.0, 0.0, 0.0]
ms = Show(clip, view)
ms.ColorArrayName = ["CELLS", ""]
ms.AmbientColor = ms.DiffuseColor = [0.74, 0.72, 0.68]
ms.Opacity = 0.4

# 막장면 쐐기 (관측 기반, 진홍)
fw = XMLPolyDataReader(FileName=[wedges_vtp], registrationName="face_wedges")
f = Show(fw, view)
f.ColorArrayName = ["CELLS", ""]
f.AmbientColor = f.DiffuseColor = [0.86, 0.16, 0.30]
f.Opacity = 1.0

# 내부 닫힌 블록 다면체 (청회색 단일색)
ib = XMLPolyDataReader(FileName=[interior_vtp], registrationName="interior_blocks")
i = Show(ib, view)
i.ColorArrayName = ["CELLS", ""]
i.AmbientColor = i.DiffuseColor = [0.42, 0.55, 0.64]
i.Opacity = 0.85

# 굴착 시 터널면에 노출될 쐐기 블록 (굴착-후 판정의 tunnel_contact==1)
if poly_all_vtp:
    pa = XMLPolyDataReader(FileName=[poly_all_vtp], registrationName="tunnel_blocks_src")
    tb = Threshold(Input=pa, registrationName="tunnel_touching_blocks")
    tb.Scalars = ["CELLS", "tunnel_contact"]
    tb.LowerThreshold = 1
    tb.UpperThreshold = 1
    tbs = Show(tb, view)
    ColorBy(tbs, ("CELLS", "block_id"))
    tlut = GetColorTransferFunction("block_id")
    tlut.ApplyPreset("Turbo", True)
    tbs.Opacity = 0.9

# 기굴착 구간(막장면 뒤)의 터널 벽 접촉 블록
if excavated_vtp:
    ex = XMLPolyDataReader(FileName=[excavated_vtp], registrationName="excavated_wall_src")
    ew = Threshold(Input=ex, registrationName="excavated_wall_blocks")
    ew.Scalars = ["CELLS", "tunnel_contact"]
    ew.LowerThreshold = 1
    ew.UpperThreshold = 1
    ews = Show(ew, view)
    ColorBy(ews, ("CELLS", "block_id"))
    elut = GetColorTransferFunction("block_id")
    elut.ApplyPreset("Turbo", True)
    ews.Opacity = 0.9

# 균열: 관측(뚜렷) + 확률(안개)
discs = XMLPolyDataReader(FileName=[discs_vtp], registrationName="dfn_discs_src")
slut = GetColorTransferFunction("set_id")
slut.InterpretValuesAsCategories = 1
slut.Annotations = ["1", "set 1", "2", "set 2", "3", "set 3", "4", "set 4"]
slut.IndexedColors = [0.122, 0.467, 0.706, 0.173, 0.627, 0.173,
                      1.000, 0.498, 0.055, 0.580, 0.404, 0.741]
uno = Threshold(Input=discs, registrationName="stochastic_discs")
uno.Scalars = ["CELLS", "observed"]
uno.LowerThreshold = 0
uno.UpperThreshold = 0
u = Show(uno, view)
ColorBy(u, ("CELLS", "set_id"))
u.Opacity = 0.06
obs = Threshold(Input=discs, registrationName="observed_discs")
obs.Scalars = ["CELLS", "observed"]
obs.LowerThreshold = 1
obs.UpperThreshold = 1
o = Show(obs, view)
ColorBy(o, ("CELLS", "set_id"))
o.Opacity = 0.9

# 막장면 벽 + 기굴착 터널(막장면 뒤)
cap = XMLPolyDataReader(FileName=[cap_vtp], registrationName="face_surface")
c = Show(cap, view)
c.ColorArrayName = ["CELLS", ""]
c.AmbientColor = c.DiffuseColor = [0.62, 0.57, 0.50]
c.Opacity = 0.5
beh = XMLPolyDataReader(FileName=[behind_vtp], registrationName="tunnel_excavated")
bh = Show(beh, view)
bh.ColorArrayName = ["CELLS", ""]
bh.AmbientColor = bh.DiffuseColor = [0.5, 0.5, 0.55]
bh.Opacity = 0.3

box = XMLPolyDataReader(FileName=[box_vtp], registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.ColorArrayName = ["CELLS", ""]
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]

# 카메라: 굴착된 터널 안쪽 위에서 막장면과 전방 도메인을 함께 보는 사선
view.CameraPosition = [8.0, -22.0, 16.0]
view.CameraFocalPoint = [26.0, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveState(out_pvsm)
print("saved:", out_pvsm)
