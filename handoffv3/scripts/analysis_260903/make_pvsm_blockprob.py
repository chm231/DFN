# -*- coding: utf-8 -*-
"""pvpython용: MC 블록 존재확률 3D 컨투어 상태 생성.

사용: pvpython make_pvsm_blockprob.py <handoffv2_dir> <out_pvsm> [screenshot]
"""
import sys

from paraview.simple import (
    XMLImageDataReader, XMLPolyDataReader, Contour, Show, GetActiveViewOrCreate,
    ColorBy, GetColorTransferFunction, GetOpacityTransferFunction,
    SaveState, Render, ResetCamera,
)

h2 = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]
EIO = f"{h2}/example_io"

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

prob = XMLImageDataReader(FileName=[f"{EIO}/blocks_edgecut/block_prob_mc50.vti"],
                          registrationName="block_prob_mc50")

# 확률 등치면: 0.1 / 0.2 / 0.3 (평활장 기준)
con = Contour(Input=prob, registrationName="prob_contours")
con.ContourBy = ["POINTS", "p_smooth"]
con.Isosurfaces = [0.01, 0.03, 0.06]
c = Show(con, view)
ColorBy(c, ("POINTS", "p_smooth"))
lut = GetColorTransferFunction("p_smooth")
lut.ApplyPreset("Yellow - Gray - Blue", True)
lut.RescaleTransferFunction(0.0, 0.08)
lut.InvertTransferFunction()
c.Opacity = 0.35

# 고확률(관측 기반 결정론) 셀: 원자료 p_block >= 0.5 등치면
hot = Contour(Input=prob, registrationName="prob_high_deterministic")
hot.ContourBy = ["POINTS", "p_block"]
hot.Isosurfaces = [0.5]
hshow = Show(hot, view)
hshow.AmbientColor = hshow.DiffuseColor = [0.79, 0.10, 0.11]
hshow.ColorArrayName = ["POINTS", ""]
hshow.Opacity = 1.0

tunnel = XMLPolyDataReader(FileName=[f"{EIO}/paraview/full/tunnel.vtp"],
                           registrationName="tunnel")
tn = Show(tunnel, view)
tn.AmbientColor = tn.DiffuseColor = [0.55, 0.55, 0.55]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.3

box = XMLPolyDataReader(FileName=[f"{EIO}/paraview/full/domain_box.vtp"],
                        registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]
b.ColorArrayName = ["CELLS", ""]
b.LineWidth = 2

ResetCamera(view)
view.CameraPosition = [-20.0, -24.0, 18.0]
view.CameraFocalPoint = [27.2, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
if len(sys.argv) > 3:
    from paraview.simple import SaveScreenshot
    SaveScreenshot(sys.argv[3], view, ImageResolution=[1400, 900])
SaveState(out_pvsm)
print("saved:", out_pvsm)
