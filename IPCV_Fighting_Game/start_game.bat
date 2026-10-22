@echo off
REM Webcam Warriors - one-step start for Windows. Double-click, or run:
REM   start_game.bat --camera 1 --fullscreen
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the Python environment ^(takes a few minutes^)...
  py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" ( echo Python 3.10+ not found. Install it from python.org & pause & exit /b 1 )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || ( echo Installing packages failed & pause & exit /b 1 )
)
set CAM=
if "%1"=="--camera" set CAM=--camera %2
".venv\Scripts\python.exe" scripts\check_setup.py %CAM% || ( echo. & echo Fix the [FAIL] line above, then run this again. & pause & exit /b 1 )
echo Starting the game in WEBCAM mode (Esc = pause/menu)...
".venv\Scripts\python.exe" main.py --start webcam %*
pause
