@echo off
rem Checks the phone-as-camera connection (DroidCam etc.) and says in plain English what is wrong. Only reads; changes nothing.
cd /d "%~dp0"
.venv\Scripts\python.exe -m limbtrack.diagnose
echo.
echo Press any key to close.
pause > nul
