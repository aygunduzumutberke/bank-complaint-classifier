@echo off
REM Tries one Gemini call per model and logs the result or the exact error (never the key).
cd /d "%~dp0"
set LOG=gemini_check_log.txt
call .venv\Scripts\activate.bat
echo === gemini check === > %LOG%
for %%M in (gemini-flash-latest gemini-flash-lite-latest gemini-2.5-flash-lite) do (
  echo --- %%M --- >> %LOG%
  set GEMINI_MODEL=%%M
  python -u scripts\check_gemini.py >> %LOG% 2>&1
)
echo CHECK_DONE >> %LOG%
