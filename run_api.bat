@echo off
REM CrossMind FastAPI Server Launcher
REM Uses Python 3.12 which has all required packages installed
echo [CrossMind] Starting FastAPI Backend Server using Python 3.12...
"C:\Users\charu\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    echo [CrossMind] ERROR: FastAPI server failed to start.
    pause
)

