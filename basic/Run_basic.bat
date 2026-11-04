@echo off
rem Basic limb tracker. Double-click for camera 0, or drag a video file onto this file.
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo The Python environment is missing. See ..\README.md, section "Setup".
  pause
  exit /b 1
)
if "%~1"=="" (
  ..\.venv\Scripts\python.exe basic_tracker.py 0
) else (
  ..\.venv\Scripts\python.exe basic_tracker.py "%~1"
)
if errorlevel 1 pause
