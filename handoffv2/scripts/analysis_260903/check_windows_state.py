# -*- coding: utf-8 -*-
import sys
from paraview.simple import LoadState, GetActiveView, SaveScreenshot, Render
LoadState(sys.argv[1])
v = GetActiveView()
Render(v)
SaveScreenshot(sys.argv[2], v, ImageResolution=[1400, 900])
print("screenshot ok")
