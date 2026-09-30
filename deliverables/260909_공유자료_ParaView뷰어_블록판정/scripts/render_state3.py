import sys, os
from paraview.simple import *
pvsm, prefix, dolly = sys.argv[1], sys.argv[2], float(sys.argv[3])
LoadState(pvsm)
v = GetActiveViewOrCreate("RenderView")
v.UseColorPaletteForBackground = 0; v.Background = [1, 1, 1]; v.OrientationAxesVisibility = 1; v.ViewSize = [1600, 900]
hidden = []
for (name, _id), src in GetSources().items():
    n = name.lower()
    if "disc" in n or "dfn" in n or "fracture" in n or "trace" in n:
        Hide(src, v); hidden.append(name)
print("hidden:", hidden, flush=True)
Render(v); v.ResetCamera(False); GetActiveCamera().Dolly(dolly); Render(v)
SaveScreenshot(prefix + "_nodisc.png", v, ImageResolution=[1600, 900])
print("done", os.path.basename(pvsm), flush=True)
