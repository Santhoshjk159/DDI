# DDIPredict - Full Dev Environment Startup
# Run after setup_postgres.ps1 has been run once

$PG_BIN = "C:\Users\$env:USERNAME\AppData\Local\Programs\postgresql16\pgsql\bin"
$PG_DATA = "C:\Users\$env:USERNAME\AppData\Local\Programs\postgresql16\data"
$UV = "C:\Users\$env:USERNAME\.local\bin\uv.exe"
$NODE = "C:\Users\$env:USERNAME\AppData\Local\Programs\nodejs\node-v22.16.0-win-x64"
$BACKEND_DIR = "S:\Santhosh\Projects\DDI\backend"
$FRONTEND_DIR = "S:\Santhosh\Projects\DDI\frontend"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " DDIPredict - Development Environment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Start PostgreSQL
Write-Host "`n[1/4] Starting PostgreSQL..." -ForegroundColor Yellow
$pgProcess = Get-Process -Name "postgres" -ErrorAction SilentlyContinue
if (-not $pgProcess) {
    & "$PG_BIN\pg_ctl.exe" start -D $PG_DATA -l "$PG_DATA\..\pg.log" -w 2>&1
} else {
    Write-Host "  Already running." -ForegroundColor Green
}
Start-Sleep -Seconds 2

# 2. Train model if not done
if (-not (Test-Path "$BACKEND_DIR\model_artifacts\rf_model.pkl")) {
    Write-Host "`n[2/4] Training ML model (first time only)..." -ForegroundColor Yellow
    Set-Location $BACKEND_DIR
    & $UV run python app/ml/train.py
} else {
    Write-Host "`n[2/4] ML model already trained." -ForegroundColor Green
}

# 3. Seed database if not done
Write-Host "`n[3/4] Checking database seed..." -ForegroundColor Yellow
Set-Location $BACKEND_DIR
& $UV run python scripts/seed_db.py

# 4. Start FastAPI + React in separate windows
Write-Host "`n[4/4] Starting services..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$BACKEND_DIR'; Write-Host 'FastAPI Backend' -ForegroundColor Cyan; & '$UV' run uvicorn app.main:app --reload --port 8000"

Start-Sleep -Seconds 3

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "`$env:PATH = '$NODE;' + `$env:PATH; Set-Location '$FRONTEND_DIR'; Write-Host 'React Frontend' -ForegroundColor Cyan; & '$NODE\npm.cmd' run dev"

Write-Host "`n============================================" -ForegroundColor Green
Write-Host " All services starting!" -ForegroundColor Green
Write-Host " Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host " Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host " API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Green

Start-Sleep -Seconds 5
Start-Process "http://localhost:5173"
