@echo off
REM CrossMind Demo Pipeline Launcher
REM Uses Python 3.12 which has all required packages installed
echo [CrossMind] Running Demo Pipeline using Python 3.12...
"C:\Users\charu\AppData\Local\Programs\Python\Python312\python.exe" "%~dp0run_demo.py" %*
if %ERRORLEVEL% NEQ 0 (
    echo [CrossMind] ERROR: Demo pipeline failed.
    pause
)

