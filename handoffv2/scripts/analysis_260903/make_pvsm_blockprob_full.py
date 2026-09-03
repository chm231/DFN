# -*- coding: utf-8 -*-
"""pvpython용: 전체 도메인 MC 블록 존재확률 3D 컨투어 상태 생성 (1~6면 시나리오).

사용: pvpython make_pvsm_blockprob_full.py <handoffv2_dir> <out_pvsm> [screenshot]
"""
import sys

from paraview.simple import (
    XMLImageDataReader, XMLPolyDataReader, Contour, Show, GetActiveViewOrCreate,
    ColorBy, GetColorTransferFunction, GetScalarBar, SaveState, Render,
    ResetCamera,
)

h2 = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]
EIO = f"{h2}/example_io"
GEO = f"{EIO}/paraview/f06"

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

prob = XMLImageDataReader(
    FileName=[f"{EIO}/blocks_edgecut/f06/block_prob_full_mc50.vti"],
    registrationName="block_prob_full_mc50")

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
c.SetScalarBarVisibility(view, True)
sb = GetScalarBar(lut, view)
sb.Title = "P(block)"
sb.ComponentTitle = ""
sb.TitleColor = [0.0, 0.0, 0.0]
sb.LabelColor = [0.0, 0.0, 0.0]
sb.AutomaticLabelFormat = 0
sb.LabelFormat = '{:.2f}'
sb.RangeLabelFormat = '{:.2f}'
sb.UseCustomLabels = 1
sb.CustomLabels = [0.0, 0.02, 0.04, 0.06, 0.08]

hot = Contour(Input=prob, registrationName="prob_high_p50")
hot.ContourBy = ["POINTS", "p_block"]
hot.Isosurfaces = [0.5]
hshow = Show(hot, view)
hshow.AmbientColor = hshow.DiffuseColor = [0.79, 0.10, 0.11]
hshow.ColorArrayName = ["POINTS", ""]
hshow.Opacity = 1.0

tunnel = XMLPolyDataReader(FileName=[f"{GEO}/tunnel_full.vtp"],
                           registrationName="tunnel_assumed_full")
tn = Show(tunnel, view)
tn.AmbientColor = tn.DiffuseColor = [0.55, 0.55, 0.55]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.3

cap = XMLPolyDataReader(FileName=[f"{GEO}/face_cap.vtp"],
                        registrationName="face_marker")
cp = Show(cap, view)
cp.AmbientColor = cp.DiffuseColor = [0.2, 0.3, 0.8]
cp.ColorArrayName = ["CELLS", ""]
cp.Opacity = 0.25

box = XMLPolyDataReader(FileName=[f"{GEO}/domain_box_full.vtp"],
                        registrationName="domain_box_full")
b = Show(box, view)
b.Representation = "Wireframe"
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]
b.ColorArrayName = ["CELLS", ""]
b.LineWidth = 2

ResetCamera(view)
view.CameraPosition = [-32.0, -30.0, 22.0]
view.CameraFocalPoint = [10.6, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
if len(sys.argv) > 3:
    from paraview.simple import SaveScreenshot
    SaveScreenshot(sys.argv[3], view, ImageResolution=[1400, 900])
SaveState(out_pvsm)
print("saved:", out_pvsm)
