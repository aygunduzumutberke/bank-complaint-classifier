@echo off
REM Full run on real CFPB data: check Gemini, download data, train, evaluate, report, open the app.
cd /d "%~dp0"
set LOG=real_run_log.txt
echo === real run === > %LOG%
if not exist .venv (
  echo VENV_MISSING: run run_demo.bat first >> %LOG%
  exit /b 1
)
call .venv\Scripts\activate.bat

echo [1/6] Checking the Gemini key...
python -u scripts\check_gemini.py >> %LOG% 2>&1 || (echo STEP_FAILED check_gemini >> %LOG% & exit /b 1)

echo [2/6] Downloading CFPB complaints...
python -u scripts\download_cfpb.py --shards 4 >> %LOG% 2>&1 || (echo STEP_FAILED download >> %LOG% & exit /b 1)

echo [3/6] Preparing the data...
python -u -m src.prepare_data --input data\raw\cfpb_sample.csv --per-class 1500 >> %LOG% 2>&1 || (echo STEP_FAILED prepare >> %LOG% & exit /b 1)

echo [4/6] Training the baseline...
python -u -m src.baseline >> %LOG% 2>&1 || (echo STEP_FAILED baseline >> %LOG% & exit /b 1)

echo [5/6] Evaluating the LLM, about 30 minutes...
python -u -m src.evaluate_llm --per-class 30 >> %LOG% 2>&1 || (echo STEP_FAILED evaluate >> %LOG% & exit /b 1)

echo [6/6] Writing the report...
python -u -m src.report >> %LOG% 2>&1 || (echo STEP_FAILED report >> %LOG% & exit /b 1)
echo REAL_RUN_DONE >> %LOG%

start "" cmd /c "timeout /t 10 >nul & start http://localhost:8501"
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
