@echo off
echo ========================================
echo Starting CrossMind FastAPI Backend Server
echo ========================================
echo.
echo Using Python 3.12...
C:\Users\charu\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn app.main:app --reload --port 8000
pause
