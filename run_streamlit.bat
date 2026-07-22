@echo off
REM CrossMind Streamlit Launcher
REM Uses Python 3.12 which has all required packages installed
echo [CrossMind] Starting Streamlit Dashboard using Python 3.12...
"C:\Users\charu\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run "%~dp0dashboard\app.py" %*
if %ERRORLEVEL% NEQ 0 (
    echo [CrossMind] ERROR: Streamlit failed to start. Make sure Python 3.12 has all dependencies installed.
    echo [CrossMind] Run: pip install -r requirements.txt
    pause
)

