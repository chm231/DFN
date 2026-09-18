# -*- coding: utf-8 -*-
"""pvpython용: 막장면 쐐기 시드 비교 뷰 (.pvsm).

사용: pvpython make_pvsm_face_mc.py <base_obs_vtp> <seed808_vtp> <seed123_vtp>
                                     <discs_vtp> <tunnel_vtp> <out_pvsm>
레이어:
  wedges_observed_base : 관측 균열만의 쐐기 7개 (청회색, 모든 시드 공통 하한)
  wedges_seed808       : 최악 실현 — 37 L 쐐기 포함 (주황, 반투명)
  wedges_seed123       : 최다 실현 (+5개) (녹색, 기본 숨김)
  observed_discs / tunnel
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, GetActiveViewOrCreate, Threshold, SaveState, Render,
    ColorBy, GetColorTransferFunction,
)

base_vtp, s808_vtp, s123_vtp, discs_vtp, tunnel_vtp, out_pvsm = \
    [a.replace("\\", "/") for a in sys.argv[1:7]]
domain_poly_vtp = sys.argv[7].replace("\\", "/") if len(sys.argv) > 7 else None

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

base = XMLPolyDataReader(FileName=[base_vtp], registrationName="wedges_observed_base")
b = Show(base, view)
b.ColorArrayName = ["CELLS", ""]
b.AmbientColor = b.DiffuseColor = [0.35, 0.47, 0.58]   # 청회색
b.Opacity = 1.0

s808 = XMLPolyDataReader(FileName=[s808_vtp], registrationName="wedges_seed808_worst")
s8 = Show(s808, view)
s8.ColorArrayName = ["CELLS", ""]
s8.AmbientColor = s8.DiffuseColor = [0.90, 0.45, 0.10]  # 주황
s8.Opacity = 0.55

s123 = XMLPolyDataReader(FileName=[s123_vtp], registrationName="wedges_seed123_most")
s1 = Show(s123, view)
s1.ColorArrayName = ["CELLS", ""]
s1.AmbientColor = s1.DiffuseColor = [0.20, 0.60, 0.30]  # 녹색
s1.Opacity = 0.55
s1.Visibility = 0

discs = XMLPolyDataReader(FileName=[discs_vtp], registrationName="dfn_discs_src")
obs = Threshold(Input=discs, registrationName="observed_discs")
obs.Scalars = ["CELLS", "observed"]
obs.LowerThreshold = 1
obs.UpperThreshold = 1
o = Show(obs, view)
o.ColorArrayName = ["CELLS", ""]
o.AmbientColor = o.DiffuseColor = [0.72, 0.78, 0.84]
o.Opacity = 0.15

tun = XMLPolyDataReader(FileName=[tunnel_vtp], registrationName="tunnel")
t = Show(tun, view)
t.ColorArrayName = ["CELLS", ""]
t.AmbientColor = t.DiffuseColor = [0.5, 0.5, 0.55]
t.Opacity = 0.15

# --- 전방 도메인의 닫힌 블록 (하이브리드 다면체) ---
if domain_poly_vtp:
    dp = XMLPolyDataReader(FileName=[domain_poly_vtp], registrationName="domain_blocks_src")
    dt = Threshold(Input=dp, registrationName="domain_blocks_tunnel")
    dt.Scalars = ["CELLS", "tunnel_contact"]
    dt.LowerThreshold = 1
    dt.UpperThreshold = 1
    dts = Show(dt, view)
    ColorBy(dts, ("CELLS", "block_id"))
    dlut = GetColorTransferFunction("block_id")
    dlut.ApplyPreset("Turbo", True)
    dts.Opacity = 0.6
    di = Threshold(Input=dp, registrationName="domain_blocks_interior")
    di.Scalars = ["CELLS", "tunnel_contact"]
    di.LowerThreshold = 0
    di.UpperThreshold = 0
    dis = Show(di, view)
    dis.ColorArrayName = ["CELLS", ""]
    dis.AmbientColor = dis.DiffuseColor = [0.55, 0.62, 0.68]
    dis.Opacity = 0.35
    dis.Visibility = 0  # 기본 숨김 (1,700여 개라 켜면 빽빽함)

view.CameraPosition = [12.0, 1.35, 1.3]
view.CameraFocalPoint = [22.3, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveState(out_pvsm)
print("saved:", out_pvsm)
