@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..
cd /d "%PROJECT_ROOT%"
if not exist data\live_logs\access.log (
  echo Missing data\live_logs\access.log. Start Nginx on 8039 and generate manual or scripted traffic first.
  exit /b 1
)
python -m wsd.prepare_dataset --input-path data\live_logs\access.log --format nginx_combined --manual-labels data\live_labels\manual_labels.csv --output-dir data\prepared_live
endlocal
