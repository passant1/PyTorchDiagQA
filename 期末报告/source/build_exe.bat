@echo off
setlocal
title PyTorchDiag build

echo ============================================
echo   PyTorchDiagQA - PyInstaller build
echo ============================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found. Please activate your environment first.
    pause
    exit /b 1
)

set "PYINSTALLER=pyinstaller"
if exist "build_venv\Scripts\pyinstaller.exe" set "PYINSTALLER=build_venv\Scripts\pyinstaller.exe"

echo [1/3] Cleaning old build files...
if exist "dist\PyTorchDiag" rmdir /s /q "dist\PyTorchDiag"
if exist "dist\PyTorchDiag.exe" del /q "dist\PyTorchDiag.exe"
if exist "build" rmdir /s /q "build"

echo [2/3] Running PyInstaller...
%PYINSTALLER% PyDiag.spec --noconfirm
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

echo [3/3] Copying runtime data...
if exist "dist\PyTorchDiag\" (
    if not exist "dist\PyTorchDiag\data" mkdir "dist\PyTorchDiag\data"
    xcopy /E /I /Y "data" "dist\PyTorchDiag\data" >nul
    copy /Y "config.yaml" "dist\PyTorchDiag\" >nul
    if exist ".env" copy /Y ".env" "dist\PyTorchDiag\" >nul
)

echo.
echo ============================================
echo   Build complete
echo   EXE: dist\PyTorchDiag\PyTorchDiag.exe
echo   RAG data: dist\PyTorchDiag\data\doc_chunks.json
echo ============================================
echo.
pause
