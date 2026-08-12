@echo off
rem ============================================================
rem  DFN demo installer (Windows)
rem  - Requires Python 3.10+ on PATH.
rem  - Creates .venv and installs required libraries.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

where python >/dev/null 2>nul
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.10+ from https://www.python.org
    echo         and check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (.venv) ...
python -m venv .venv
if errorlevel 1 ( echo [ERROR] failed to create .venv & pause & exit /b 1 )

echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/3] Installing libraries (numpy, scipy, h5py, matplotlib, pyvista) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some libraries failed to install. Check the internet connection
    echo        and run install_demo.bat again. If only pyvista failed, the demo
    echo        still runs; only the 3D visualization step is skipped.
    pause
    exit /b 1
)

echo.
echo Install complete. Run the demo with:
echo     run_demo.bat --seed 2026
pause
