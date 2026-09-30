import sys, os
from paraview.simple import *
pvsm, prefix, dolly = sys.argv[1], sys.argv[2], float(sys.argv[3])
LoadState(pvsm)
v = GetActiveViewOrCreate("RenderView")
v.UseColorPaletteForBackground = 0; v.Background = [1, 1, 1]; v.OrientationAxesVisibility = 1; v.ViewSize = [1600, 900]
keep = ("block", "wedge", "tunnel", "cap", "domain", "box", "polyhedra", "wall", "face_plane", "faceplane")
names = []
for (name, _id), src in GetSources().items():
    n = name.lower(); names.append(name)
    if not any(k in n for k in keep):
        Hide(src, v)
print("sources:", names, flush=True)
Render(v); v.ResetCamera(False); GetActiveCamera().Dolly(dolly); Render(v)
SaveScreenshot(prefix + "_blocksonly.png", v, ImageResolution=[1600, 900])
# second: view from +x looking at the face (front)
cam = GetActiveCamera(); fp = list(cam.GetFocalPoint()); d = cam.GetDistance()
cam.SetPosition(fp[0] + d, fp[1], fp[2]); cam.SetViewUp(0, 0, 1); Render(v)
SaveScreenshot(prefix + "_blocksonly_front.png", v, ImageResolution=[1600, 900])
print("done", flush=True)
