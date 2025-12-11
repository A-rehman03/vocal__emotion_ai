@echo off
cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found. Creating one...
    py -3.12 -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

echo Starting Emotion Service...
python app.py
pause
