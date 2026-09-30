@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The Python environment is missing. See README.md, section "Setup".
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m limbtrack
