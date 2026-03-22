@echo off
REM ============================================================
REM  Face Tracker — Windows Setup Script
REM  Run once before your first launch.
REM ============================================================

echo [1/4] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [3/4] Downloading YOLOv8 face weights...
REM Download yolov8n-face.pt from the releases page
REM (will auto-download on first run if using plain yolov8n.pt)
if not exist models mkdir models
curl -L "https://github.com/derronqi/yolov8-face/releases/download/v1/yolov8n-face.pt" -o models\yolov8n-face.pt

echo [4/4] Setup complete!
echo.
echo ============================================================
echo  BEFORE RUNNING:
echo  1. Install PostgreSQL from https://www.postgresql.org/download/windows/
echo  2. Create the database:
echo       psql -U postgres -c "CREATE DATABASE face_tracker;"
echo  3. Edit config.json -- update the "dsn" field with your password
echo  4. Place your sample video as sample_video.mp4
echo ============================================================
echo.
echo Run the tracker with:
echo   venv\Scripts\activate.bat
echo   python main.py
pause
