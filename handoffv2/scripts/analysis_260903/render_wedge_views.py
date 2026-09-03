# -*- coding: utf-8 -*-
"""pvpython용: 쐐기 케이스 다각도 PNG 렌더 (평면 제거, 쐐기+터널만).

사용: pvpython render_wedge_views.py <case_dir>
출력: <case_dir>/wedge_view_iso.png, wedge_view_axis.png, wedge_view_crown.png
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, XMLImageDataReader, Threshold, Show,
    GetActiveViewOrCreate, Render, SaveScreenshot,
)

case = sys.argv[1].replace("\\", "/")
view = GetActiveViewOrCreate("RenderView")
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]

tunnel = XMLPolyDataReader(FileName=[f"{case}/tunnel.vtp"],
                           registrationName="tunnel")
tn = Show(tunnel, view)
tn.AmbientColor = tn.DiffuseColor = [0.60, 0.60, 0.60]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.30

wedge = XMLPolyDataReader(FileName=[f"{case}/wedges_polyhedra.vtp"],
                          registrationName="wedges")
w = Show(wedge, view)
w.ColorArrayName = ["CELLS", "rgb"]
w.MapScalars = 0
w.Opacity = 1.0

FOC = [17.7, 1.35, 1.3]
views = [
    ("iso",   [-16.0, -26.0, 20.0], [0.0, 0.0, 1.0]),
    ("axis",  [-14.0, 1.35, 3.0],   [0.0, 0.0, 1.0]),   # 터널축 후방에서 전방
    ("crown", [17.7, 1.35, 26.0],   [1.0, 0.0, 0.0]),   # 천장 위에서 내려다봄
]
for tag, pos, up in views:
    view.CameraPosition = pos
    view.CameraFocalPoint = FOC
    view.CameraViewUp = up
    Render(view)
    out = f"{case}/wedge_view_{tag}.png"
    SaveScreenshot(out, view, ImageResolution=[1600, 1000])
    print("saved:", out)
