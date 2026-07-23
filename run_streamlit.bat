@echo off
echo ========================================
echo Starting CrossMind Streamlit Dashboard
echo ========================================
echo.
echo Make sure FastAPI server is running on port 8000
echo (run run_api.bat in a separate terminal)
echo.
echo Using Python 3.12...
C:\Users\charu\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run dashboard/app.py --server.port 8501
pause
