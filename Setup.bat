@echo off
rem Sets up the limb tracker on a new Windows computer. Double-click this file.
rem Safe to run again: it only adds what is missing.
cd /d "%~dp0"
echo.
echo === Limb tracker setup ===
echo.

rem 1. Python 3.12
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.12 was not found. Installing it with winget...
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  echo.
  echo Python was installed. Close this window and double-click Setup.bat again
  echo so Windows can find it.
  pause
  exit /b 0
)

rem 2. Python environment and packages
if not exist ".venv\Scripts\python.exe" (
  echo Making the Python environment in .venv ...
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
echo Installing the Python packages (a few minutes the first time)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

rem 3. Check the protocol file
".venv\Scripts\python.exe" -m limbtrack.protocol check

echo.
echo === Done ===
echo Start the program with Start.bat.
echo First try it without a camera: Live tab, "Open video file (practice)",
echo then demo\demo_leg_white_dots.mp4.
echo.
echo Only needed if you will measure force: install the NI-DAQmx driver from
echo https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html
echo Only needed if you use a phone as camera: install DroidCam, Iriun or Camo on
echo the phone and on this computer.
echo.
pause
exit /b 0

:failed
echo.
echo Something went wrong. Copy the text above and send it to whoever helps you.
pause
exit /b 1
