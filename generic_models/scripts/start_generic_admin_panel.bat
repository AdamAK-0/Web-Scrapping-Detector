@echo off
setlocal

cd /d "%~dp0\..\.."

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo Starting generic WSD lab...
echo Admin: http://127.0.0.1:8050/
echo Sites: http://127.0.0.1:8061/  http://127.0.0.1:8062/  http://127.0.0.1:8063/  http://127.0.0.1:8064/

start "" powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8050/'"

"%PYTHON%" -m generic_models.generic_lab --host 127.0.0.1 --admin-port 8050 --model-dir generic_models\artifacts\models --log-dir generic_models\live_logs

endlocal
