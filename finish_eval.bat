@echo off
REM Second pass: re-asks Gemini only for complaints that failed (successes come from the cache), then rebuilds the report.
cd /d "%~dp0"
set LOG=finish_log.txt
call .venv\Scripts\activate.bat
set GEMINI_MODEL=gemini-flash-lite-latest
echo === finish evaluation === > %LOG%
python -u -m src.evaluate_llm --per-class 30 >> %LOG% 2>&1 || (echo STEP_FAILED evaluate >> %LOG% & exit /b 1)
python -u -m src.report >> %LOG% 2>&1 || (echo STEP_FAILED report >> %LOG% & exit /b 1)
echo FINISH_DONE >> %LOG%
