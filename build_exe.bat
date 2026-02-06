@echo off
setlocal
echo ===========================================
echo      AI Upscaler - Build Executable
echo ===========================================

REM 1. Check prerequisites
echo [1/3] Checking prerequisites...
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed.
    pause
    exit /b 1
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed.
    pause
    exit /b 1
)

REM 2. Build Frontend
echo.
echo [2/3] Building Frontend...
call npm install
call npm run build
if %errorlevel% neq 0 (
    echo Error: Frontend build failed.
    pause
    exit /b 1
)

REM 3. Build Backend
echo.
echo [3/3] Building Backend (Exe)...
cd backend

REM Check if venv is active otherwise rely on global python
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Check for PyInstaller
where pyinstaller >nul 2>nul
if %errorlevel% neq 0 (
    echo Warning: PyInstaller not found. Installing...
    pip install -r requirements.txt
    pip install pyinstaller
)

echo.
echo [!] Clossing running processes to release file locks...
:: Kill any existing processes (Aggressive)
taskkill /F /IM UpscalerAI.exe /T 2>nul
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul
taskkill /F /IM UpscalerAI.exe /T 2>nul
taskkill /F /IM python.exe /T 2>nul

echo Fixing numpy issues...
REM Try to install directly, force reinstall if needed
pip install "numpy<2.0.0" --force-reinstall --no-warn-script-location
if %errorlevel% neq 0 (
    echo Error: Failed to install numpy. Please manually close any programs using Python.
    pause
    exit /b 1
)
pip install --upgrade pyinstaller


echo Cleaning previous builds...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo Running PyInstaller...
pyinstaller --clean upscaler.spec

if %errorlevel% neq 0 (
    echo Error: PyInstaller build failed.
    cd ..
    pause
    exit /b 1
)

echo.
echo ===========================================
echo           BUILD SUCCESSFUL!
echo ===========================================
echo.
echo Your app is ready at: backend\dist\UpscalerAI\UpscalerAI.exe
echo.
echo IMPORTANT: FFmpeg is required.
echo Please ensure ffmpeg.exe is in the same folder as the exe
echo or added to the system PATH.
echo.
cd ..
pause
