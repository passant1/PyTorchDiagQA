@echo off
title PyTorchDiagQA

echo ============================================
echo   PyTorch Diagnostic QA System
echo ============================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version

REM Test if vendor packages work with this Python
python -c "import sys; sys.path.insert(0,'vendor'); import sklearn, jieba, yaml, fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [*] Reinstalling packages for current Python version...
    echo     This takes 1-2 minutes, only once.
    if exist "vendor" rmdir /s /q vendor 2>nul
    python -m pip install --target=vendor --quiet jieba scikit-learn numpy pyyaml openai fastapi uvicorn pydantic
    REM Check if imports work (ignore pip warnings)
    python -c "import sys; sys.path.insert(0,'vendor'); import sklearn, jieba, yaml, fastapi" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Install failed. Check internet, then try again.
        pause
        exit /b 1
    )
    echo [OK] Dependencies ready.
)

echo.
echo ============================================
echo   Server: http://127.0.0.1:18888
echo   Press Ctrl+C to stop
echo ============================================
echo.

start "" "http://127.0.0.1:18888"
set "PYTHONPATH=vendor;%PYTHONPATH%"
python main.py --server

pause
