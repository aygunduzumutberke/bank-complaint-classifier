@echo off
REM Windows one-click setup: creates .venv, installs requirements, runs the offline demo, opens the app.
cd /d "%~dp0"
echo === Bank Complaint Classifier: setup + demo === > demo_log.txt

set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY python --version >nul 2>&1 && set PY=python
if not defined PY (
  echo PYTHON_NOT_FOUND >> demo_log.txt
  echo Python not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
%PY% --version >> demo_log.txt 2>&1

if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv >> demo_log.txt 2>&1
)
call .venv\Scripts\activate.bat

echo Installing packages, this can take a few minutes...
python -m pip install --upgrade pip >> demo_log.txt 2>&1
python -m pip install -r requirements.txt >> demo_log.txt 2>&1
if errorlevel 1 (
  echo PIP_FAILED >> demo_log.txt
  echo Package installation failed, see demo_log.txt
  pause
  exit /b 1
)

echo Running the offline demo...
python scripts\make_demo_data.py >> demo_log.txt 2>&1
python -m src.prepare_data --demo >> demo_log.txt 2>&1
python -m src.baseline >> demo_log.txt 2>&1
python -m src.evaluate_llm --per-class 5 --provider mock >> demo_log.txt 2>&1
python -m src.report >> demo_log.txt 2>&1
python -m pytest -q >> demo_log.txt 2>&1
echo DEMO_DONE >> demo_log.txt

echo Starting the app at http://localhost:8501 (close this window to stop it)
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8501"
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
