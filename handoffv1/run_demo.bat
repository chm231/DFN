@echo off
rem Demo run wrapper - uses the .venv Python created by install_demo.bat.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run install_demo.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_demo.py %*
pause
