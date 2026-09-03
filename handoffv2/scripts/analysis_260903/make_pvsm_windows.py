# -*- coding: utf-8 -*-
"""pvpython용: 누적 윈도우 4종(f01-07~f04-07, seed101, 세트 통일본) 비교 상태 생성.

사용: pvpython make_pvsm_windows.py <windows_dir> <out_pvsm>
  <windows_dir>/f01-07 ... f04-07 아래에 export_domain_dfn_vtk 출력이 있어야 함.
  기본 표시는 f01-07만 켜짐 — 나머지 윈도우는 Pipeline Browser의 눈 아이콘으로 토글.
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, Hide, GetActiveViewOrCreate, ColorBy,
    GetColorTransferFunction, Threshold, SaveState, Render, ResetCamera,
)

base = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]
TAGS = ["f01-07", "f02-07", "f03-07", "f04-07"]

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

lut = GetColorTransferFunction("set_id")
lut.InterpretValuesAsCategories = 1
lut.Annotations = ["1", "set 1", "2", "set 2", "3", "set 3"]
lut.IndexedColors = [
    0.122, 0.467, 0.706,   # set1 파랑
    0.173, 0.627, 0.173,   # set2 초록
    1.000, 0.498, 0.055,   # set3 주황
]

for i, tag in enumerate(TAGS):
    d_dir = f"{base}/{tag}"
    discs = XMLPolyDataReader(FileName=[f"{d_dir}/dfn_discs.vtp"],
                              registrationName=f"{tag}_discs")
    d = Show(discs, view)
    d.Representation = "Surface"
    ColorBy(d, ("CELLS", "set_id"))
    d.Opacity = 0.12

    thr = Threshold(Input=discs, registrationName=f"{tag}_observed")
    thr.Scalars = ["CELLS", "observed"]
    thr.LowerThreshold = 1
    thr.UpperThreshold = 1
    t = Show(thr, view)
    ColorBy(t, ("CELLS", "set_id"))
    t.Opacity = 1.0

    traces = XMLPolyDataReader(FileName=[f"{d_dir}/wall_traces.vtp"],
                               registrationName=f"{tag}_wall_traces")
    tr = Show(traces, view)
    tr.AmbientColor = tr.DiffuseColor = [0.84, 0.15, 0.16]
    tr.ColorArrayName = ["CELLS", ""]
    tr.LineWidth = 3

    if i > 0:  # 기본은 f01-07만 표시
        Hide(discs, view)
        Hide(thr, view)
        Hide(traces, view)

# --- 터널·도메인 박스 (윈도우 간 차이는 cm 수준이라 f01-07 것만 표시) ---
tunnel = XMLPolyDataReader(FileName=[f"{base}/f01-07/tunnel.vtp"],
                           registrationName="tunnel")
tn = Show(tunnel, view)
tn.Representation = "Surface"
tn.AmbientColor = tn.DiffuseColor = [0.6, 0.6, 0.6]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.3

box = XMLPolyDataReader(FileName=[f"{base}/f01-07/domain_box.vtp"],
                        registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]
b.ColorArrayName = ["CELLS", ""]
b.LineWidth = 2

ResetCamera(view)
view.CameraPosition = [50.0, -38.0, 35.0]
view.CameraFocalPoint = [17.7, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveState(out_pvsm)
print("saved:", out_pvsm)
