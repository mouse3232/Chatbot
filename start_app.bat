@echo off
TITLE Astrology ChatBot Launcher
echo [1/3] Terminating existing Python/Chainlit processes...
taskkill /F /IM chainlit.exe /T >nul 2>&1
taskkill /F /IM python.exe /T >nul 2>&1

echo [2/3] Activating Virtual Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Error: .venv not found. Please ensure the virtual environment exists in the root directory.
    pause
    exit /b
)

echo [3/3] Starting Astrology Backend and Frontend...
echo Starting FastAPI Backend Services...
start "Astrology Backend API" cmd /k "call .venv\Scripts\activate.bat && uvicorn backend.app:app --host 0.0.0.0 --port 8001 --reload"

echo Waiting 3 seconds for Backend to initialize...
timeout /t 3 /nobreak >nul

echo Starting Chainlit Frontend UI...
chainlit run frontend\ui.py

pause
