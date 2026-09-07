@echo off
REM 6-face blockview with fixed minimum block volume (0.05 m3), voxel 0.1 m.
REM Compare side by side with 열기_블록뷰_6면.bat (min 8 voxels = 0.008 m3).
start "" "C:\Program Files\ParaView 6.1.1\bin\paraview.exe" --state="%~dp0dfn_blockview_f06_minvol05.pvsm"
