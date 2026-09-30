import sys, os, time
from paraview.simple import *
pvsm, prefix = sys.argv[1], sys.argv[2]
t0 = time.time()
LoadState(pvsm)
v = GetActiveViewOrCreate("RenderView")
v.UseColorPaletteForBackground = 0
v.Background = [1.0, 1.0, 1.0]
v.OrientationAxesVisibility = 1
v.ViewSize = [1600, 900]
Render(v)
# 1) saved camera, white bg
SaveScreenshot(prefix + "_saved.png", v, ImageResolution=[1600, 900])
# 2) fit camera to data along saved direction, then zoom in
v.ResetCamera(False)
cam = GetActiveCamera(); cam.Dolly(1.45)
Render(v)
SaveScreenshot(prefix + "_fit.png", v, ImageResolution=[1600, 900])
# 3) closer view
cam.Dolly(1.6); Render(v)
SaveScreenshot(prefix + "_close.png", v, ImageResolution=[1600, 900])
print("done", os.path.basename(pvsm), f"{time.time()-t0:.0f}s", flush=True)
