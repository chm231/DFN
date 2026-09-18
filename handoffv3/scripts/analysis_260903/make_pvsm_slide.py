# -*- coding: utf-8 -*-
"""pvpython용: 슬라이딩 윈도우 3세트 단일 상태 (레이어 그룹 토글식).

그룹: w14_* (기본 표시) / w25_* / w36_* (숨김) — Pipeline Browser 눈 아이콘으로 전환.
사용: pvpython make_pvsm_slide.py <handoffv2_dir> <out_pvsm> [screenshot]
"""
import glob
import sys

from paraview.simple import (
    XMLPolyDataReader, XMLImageDataReader, Contour, Show, Hide,
    GetActiveViewOrCreate, ColorBy, GetColorTransferFunction, GetScalarBar,
    SaveState, Render, ResetCamera,
)

h2 = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]

view = GetActiveViewOrCreate("RenderView")
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]

# 3D 배경 그리드 (축 눈금·라벨)
ag = view.AxesGrid
ag.Visibility = 1
ag.ShowGrid = 1
ag.XTitle = "x [m]"
ag.YTitle = "y [m]"
ag.ZTitle = "z [m]"
for prop in ("GridColor",):
    setattr(ag, prop, [0.55, 0.55, 0.55])
for prop in ("XTitleColor", "YTitleColor", "ZTitleColor",
             "XLabelColor", "YLabelColor", "ZLabelColor"):
    try:
        setattr(ag, prop, [0.1, 0.1, 0.1])
    except Exception:
        pass

lut = GetColorTransferFunction("p_smooth")
lut.ApplyPreset("Yellow - Gray - Blue", True)
lut.RescaleTransferFunction(0.0, 0.6)
lut.InvertTransferFunction()

WINDOWS = ["w14", "w25", "w36"]
first_contour_show = None
for tag in WINDOWS:
    d = f"{h2}/example_io/f_slide/{tag}"
    vis = tag == "w14"
    vti = glob.glob(f"{d}/block_prob_mc*.vti")[0]
    prob = XMLImageDataReader(FileName=[vti], registrationName=f"{tag}_prob")

    con = Contour(Input=prob, registrationName=f"{tag}_prob_contours")
    con.ContourBy = ["POINTS", "p_smooth"]
    con.Isosurfaces = [0.02, 0.05, 0.15, 0.30]
    c = Show(con, view)
    ColorBy(c, ("POINTS", "p_smooth"))
    c.Opacity = 0.35
    if vis and first_contour_show is None:
        first_contour_show = c
    if not vis:
        Hide(con, view)

    hot = Contour(Input=prob, registrationName=f"{tag}_prob_p50")
    hot.ContourBy = ["POINTS", "p_block"]
    hot.Isosurfaces = [0.5]
    hs = Show(hot, view)
    hs.AmbientColor = hs.DiffuseColor = [0.79, 0.10, 0.11]
    hs.ColorArrayName = ["POINTS", ""]
    if not vis:
        Hide(hot, view)

    tt = XMLPolyDataReader(FileName=[f"{d}/target_face_traces.vtp"],
                           registrationName=f"{tag}_target_face_traces")
    t = Show(tt, view)
    t.AmbientColor = t.DiffuseColor = [0.0, 0.25, 0.85]
    t.ColorArrayName = ["CELLS", ""]
    t.LineWidth = 2
    if not vis:
        Hide(tt, view)

    for name, col, op, rep in (("tunnel_behind", [0.55, 0.55, 0.55], 0.3, "Surface"),
                               ("face_cap", [0.76, 0.70, 0.60], 0.4, "Surface"),
                               ("domain_box", [0.0, 0.0, 0.0], 1.0, "Wireframe")):
        r = XMLPolyDataReader(FileName=[f"{d}/{name}.vtp"],
                              registrationName=f"{tag}_{name}")
        s = Show(r, view)
        s.Representation = rep
        s.AmbientColor = s.DiffuseColor = col
        s.ColorArrayName = ["CELLS", ""]
        s.Opacity = op
        if rep == "Wireframe":
            s.LineWidth = 2
        if not vis:
            Hide(r, view)

lut.RescaleTransferFunction(0.0, 0.6)
if first_contour_show is not None:
    first_contour_show.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title = "P(block)"
    sb.ComponentTitle = ""
    sb.TitleColor = [0.0, 0.0, 0.0]
    sb.LabelColor = [0.0, 0.0, 0.0]
    sb.AutomaticLabelFormat = 0
    sb.LabelFormat = '{:.2f}'
    sb.RangeLabelFormat = '{:.2f}'
    sb.UseCustomLabels = 1
    sb.CustomLabels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

ResetCamera(view)
view.CameraPosition = [-26.0, -26.0, 20.0]
view.CameraFocalPoint = [11.0, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
if len(sys.argv) > 3:
    from paraview.simple import SaveScreenshot
    SaveScreenshot(sys.argv[3], view, ImageResolution=[1400, 900])
SaveState(out_pvsm)
print("saved:", out_pvsm)
