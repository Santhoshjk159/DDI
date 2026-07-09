# DDIPredict - PostgreSQL Portable Setup Script
# Run this once to initialize PostgreSQL

$PG_ZIP = "C:\Users\$env:USERNAME\Downloads\postgres16.zip"
$PG_DIR = "C:\Users\$env:USERNAME\AppData\Local\Programs\postgresql16"
$PG_DATA = "C:\Users\$env:USERNAME\AppData\Local\Programs\postgresql16\data"
$PG_BIN = "C:\Users\$env:USERNAME\AppData\Local\Programs\postgresql16\pgsql\bin"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " DDIPredict - PostgreSQL Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check if already extracted
if (-not (Test-Path $PG_BIN)) {
    Write-Host "`n[1/4] Extracting PostgreSQL..." -ForegroundColor Yellow
    if (-not (Test-Path $PG_ZIP)) {
        Write-Host "ERROR: postgres16.zip not found at $PG_ZIP" -ForegroundColor Red
        Write-Host "Download from: https://www.enterprisedb.com/download-postgresql-binaries" -ForegroundColor Yellow
        exit 1
    }
    Expand-Archive -Path $PG_ZIP -DestinationPath $PG_DIR -Force
    Write-Host "  Extracted to: $PG_DIR" -ForegroundColor Green
} else {
    Write-Host "`n[1/4] PostgreSQL already extracted. Skipping." -ForegroundColor Green
}

# Initialize database cluster
if (-not (Test-Path "$PG_DATA\PG_VERSION")) {
    Write-Host "`n[2/4] Initializing database cluster..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $PG_DATA | Out-Null
    & "$PG_BIN\initdb.exe" -D $PG_DATA -U postgres --encoding=UTF8 2>&1
    Write-Host "  Database cluster initialized." -ForegroundColor Green
} else {
    Write-Host "`n[2/4] Database cluster already exists. Skipping." -ForegroundColor Green
}

# Start PostgreSQL server
Write-Host "`n[3/4] Starting PostgreSQL server on port 5432..." -ForegroundColor Yellow
$pgProcess = Get-Process -Name "postgres" -ErrorAction SilentlyContinue
if (-not $pgProcess) {
    Start-Process -FilePath "$PG_BIN\pg_ctl.exe" -ArgumentList "start", "-D", $PG_DATA, "-l", "$PG_DIR\pg.log" -NoNewWindow
    Start-Sleep -Seconds 4
    Write-Host "  PostgreSQL started." -ForegroundColor Green
} else {
    Write-Host "  PostgreSQL already running." -ForegroundColor Green
}

# Create database and user
Write-Host "`n[4/4] Creating DDI database and user..." -ForegroundColor Yellow
$env:PGPASSWORD = ""
& "$PG_BIN\psql.exe" -U postgres -c "CREATE USER ddi_user WITH PASSWORD 'ddi_password';" 2>$null
& "$PG_BIN\psql.exe" -U postgres -c "CREATE DATABASE ddi_db OWNER ddi_user;" 2>$null
& "$PG_BIN\psql.exe" -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE ddi_db TO ddi_user;" 2>$null
Write-Host "  Database 'ddi_db' ready." -ForegroundColor Green

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host " PostgreSQL setup complete!" -ForegroundColor Green
Write-Host " Connection: postgresql://ddi_user:ddi_password@localhost:5432/ddi_db" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "`nNext step: Run start_dev.ps1 to start all services"
