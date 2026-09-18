# -*- coding: utf-8 -*-
"""pvpython용: 상위20 무한평면 쐐기 케이스 뷰.

사용: pvpython make_pvsm_wedgecase.py <case_dir> <out_pvsm> [screenshot]
"""
import glob
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, GetActiveViewOrCreate, SaveState, Render,
    ResetCamera,
)

case = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

tunnel = XMLPolyDataReader(FileName=[f"{case}/tunnel.vtp"],
                           registrationName="tunnel")
tn = Show(tunnel, view)
tn.AmbientColor = tn.DiffuseColor = [0.55, 0.55, 0.55]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.35

wedge = XMLPolyDataReader(FileName=[f"{case}/wedges_polyhedra.vtp"],
                          registrationName="wedges")
w = Show(wedge, view)
w.ColorArrayName = ["CELLS", "rgb"]
w.MapScalars = 0
w.Opacity = 1.0

planes = XMLPolyDataReader(FileName=[glob.glob(f"{case}/top*_planes.vtp")[0]],
                           registrationName="top20_planes")
p = Show(planes, view)
p.AmbientColor = p.DiffuseColor = [0.45, 0.60, 0.75]
p.ColorArrayName = ["CELLS", ""]
p.Opacity = 0.06

ResetCamera(view)
view.CameraPosition = [-18.0, -28.0, 20.0]
view.CameraFocalPoint = [17.7, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
if len(sys.argv) > 3:
    from paraview.simple import SaveScreenshot
    SaveScreenshot(sys.argv[3], view, ImageResolution=[1400, 900])
SaveState(out_pvsm)
print("saved:", out_pvsm)
