# -*- coding: utf-8 -*-
"""pvpython용: DFN VTK 세트를 로드·스타일링한 ParaView 상태(.pvsm) 생성.

사용: pvpython make_pvsm.py <vtk_dir> <out_pvsm> [--full]
  --full 이면 전체 19,617개 세트로 간주해 discs를 반투명 + observed Threshold 추가.
"""
import sys

from paraview.simple import (
    XMLPolyDataReader, Show, GetActiveViewOrCreate, ColorBy,
    GetColorTransferFunction, Threshold, SaveState, Render, ResetCamera,
)

vtk_dir = sys.argv[1].replace("\\", "/")
out_pvsm = sys.argv[2]
is_full = "--full" in sys.argv
overlay_vtp = None  # --overlay-observed <full discs vtp>: observed 균열 중첩
if "--overlay-observed" in sys.argv:
    overlay_vtp = sys.argv[sys.argv.index("--overlay-observed") + 1].replace("\\", "/")
blocks_vtp = None  # --blocks-vtp <blocks vtp>: 3DEC식 절단 블록 표시
if "--blocks-vtp" in sys.argv:
    blocks_vtp = sys.argv[sys.argv.index("--blocks-vtp") + 1].replace("\\", "/")

view = GetActiveViewOrCreate("RenderView")
view.Background = [1.0, 1.0, 1.0]

# --- 균열 원판 ---
discs = XMLPolyDataReader(FileName=[f"{vtk_dir}/dfn_discs.vtp"],
                          registrationName="dfn_discs")
d = Show(discs, view)
d.Representation = "Surface"
ColorBy(d, ("CELLS", "set_id"))
lut = GetColorTransferFunction("set_id")
lut.InterpretValuesAsCategories = 1
lut.Annotations = ["1", "set 1", "2", "set 2", "3", "set 3", "4", "set 4"]
lut.IndexedColors = [
    0.122, 0.467, 0.706,   # set1 파랑
    0.173, 0.627, 0.173,   # set2 초록
    1.000, 0.498, 0.055,   # set3 주황
    0.580, 0.404, 0.741,   # set4 보라
]
d.Opacity = 0.15 if is_full else (0.4 if overlay_vtp else 0.8)

# --- 관측 복원 균열 중첩 (top30 등 부분 세트 위에) ---
if overlay_vtp:
    full_discs = XMLPolyDataReader(FileName=[overlay_vtp],
                                   registrationName="observed_discs_src")
    obs_thr = Threshold(Input=full_discs, registrationName="observed_discs")
    obs_thr.Scalars = ["CELLS", "observed"]
    obs_thr.LowerThreshold = 1
    obs_thr.UpperThreshold = 1
    o = Show(obs_thr, view)
    ColorBy(o, ("CELLS", "set_id"))
    o.Opacity = 1.0

# --- full: observed만 보이는 Threshold 추가 ---
if is_full:
    thr = Threshold(Input=discs, registrationName="observed_only")
    thr.Scalars = ["CELLS", "observed"]
    thr.LowerThreshold = 1
    thr.UpperThreshold = 1
    t = Show(thr, view)
    ColorBy(t, ("CELLS", "set_id"))
    t.Opacity = 1.0

# --- 터널 ---
tunnel = XMLPolyDataReader(FileName=[f"{vtk_dir}/tunnel.vtp"],
                           registrationName="tunnel")
tn = Show(tunnel, view)
tn.Representation = "Surface"
tn.AmbientColor = tn.DiffuseColor = [0.6, 0.6, 0.6]
tn.ColorArrayName = ["CELLS", ""]
tn.Opacity = 0.3

# --- 도메인 박스 ---
box = XMLPolyDataReader(FileName=[f"{vtk_dir}/domain_box.vtp"],
                        registrationName="domain_box")
b = Show(box, view)
b.Representation = "Wireframe"
b.AmbientColor = b.DiffuseColor = [0.0, 0.0, 0.0]
b.ColorArrayName = ["CELLS", ""]
b.LineWidth = 2

# --- 터널 벽면 절리선 ---
traces = XMLPolyDataReader(FileName=[f"{vtk_dir}/wall_traces.vtp"],
                           registrationName="wall_traces")
tr = Show(traces, view)
tr.AmbientColor = tr.DiffuseColor = [0.84, 0.15, 0.16]
tr.ColorArrayName = ["CELLS", ""]
tr.LineWidth = 3

# --- 3DEC식 절단 블록 ---
if blocks_vtp:
    blk = XMLPolyDataReader(FileName=[blocks_vtp], registrationName="blocks_3dec")
    bl = Show(blk, view)
    ColorBy(bl, ("CELLS", "block_id"))
    blut = GetColorTransferFunction("block_id")
    blut.ApplyPreset("Turbo", True)
    bl.Opacity = 1.0
    d.Opacity = 0.15  # 블록이 주인공이므로 discs는 더 흐리게

ResetCamera(view)
view.CameraPosition = [60.0, -35.0, 35.0]
view.CameraFocalPoint = [27.2, 1.35, 1.3]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveState(out_pvsm)
print("saved:", out_pvsm)
