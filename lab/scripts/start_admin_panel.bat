@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
cd /d "%PROJECT_ROOT%"

set PYTHON_EXE=python
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe

echo Starting local website on http://127.0.0.1:8039 ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\lab\scripts\start_nginx_windows.ps1"
if errorlevel 1 exit /b 1

echo Starting WSD admin panel on http://127.0.0.1:8040 ...
start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:8040/"
"%PYTHON_EXE%" -m wsd.admin_panel ^
  --model-dir data\prepared_live\models ^
  --graph-dir data\prepared_live ^
  --log-path data\live_logs\access.log ^
  --labels-path data\live_labels\manual_labels.csv ^
  --base-url http://127.0.0.1:8039 ^
  --host 127.0.0.1 ^
  --port 8040

endlocal
