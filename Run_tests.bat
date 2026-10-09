@echo off
rem Runs the whole automatic test suite (about 15 minutes) and writes the result to tests_last_run.txt
cd /d "%~dp0"
echo Running the tests. Do not close this window. The result is saved in tests_last_run.txt
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider > tests_last_run.txt 2>&1
echo.
findstr /C:"passed" /C:"failed" tests_last_run.txt
echo Done. Press any key to close.
pause > nul
