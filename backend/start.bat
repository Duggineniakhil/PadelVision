@echo off
echo Starting PadelVision Backend...

cd /d "%~dp0"
call ..\.venv311\Scripts\activate.bat

echo Virtual environment activated. Starting FastAPI...
python -m uvicorn app.main:app --reload --port 8000
