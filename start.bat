@echo off
setlocal
cd /d "%~dp0"
title KanjiOCRVibe OCR Service - 127.0.0.1:8765

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Virtual environment not found.
    echo Create it and install the requirements first:
    echo.
    echo   py -3.11 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
echo.
echo Starting KanjiOCRVibe OCR on http://127.0.0.1:8765
echo Close this window to stop the service. Press Ctrl+C to stop it without closing.
echo.

".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8765
set "EXITCODE=%ERRORLEVEL%"

echo.
echo KanjiOCRVibe OCR stopped ^(exit code %EXITCODE%^).
pause
endlocal & exit /b %EXITCODE%
