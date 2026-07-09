@echo off
echo ==========================================
echo  DDIPredict - Full Stack Startup Script
echo ==========================================

echo.
echo [1/3] Checking Docker / PostgreSQL...
docker-compose up -d
timeout /t 3 /nobreak > nul

echo.
echo [2/3] Starting FastAPI backend...
start cmd /k "cd /d s:\Santhosh\Projects\DDI\backend && C:\Users\Santhos\.local\bin\uv.exe run uvicorn app.main:app --reload --port 8000"

timeout /t 4 /nobreak > nul

echo.
echo [3/3] Starting React frontend...
set NODE=C:\Users\Santhos\AppData\Local\Programs\nodejs\node-v22.16.0-win-x64
start cmd /k "set PATH=%NODE%;%PATH% && cd /d s:\Santhosh\Projects\DDI\frontend && %NODE%\npm.cmd run dev"

echo.
echo ==========================================
echo  App running at: http://localhost:5173
echo  API docs at:    http://localhost:8000/docs
echo ==========================================
pause
