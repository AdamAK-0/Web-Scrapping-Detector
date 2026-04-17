@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
cd /d "%PROJECT_ROOT%"

set SOURCE_LOG=%~1
if "%SOURCE_LOG%"=="" (
  if exist "logs.txt" (
    set SOURCE_LOG=logs.txt
  ) else (
    set SOURCE_LOG=data\live_logs\access.log
  )
)

if not exist "%SOURCE_LOG%" (
  echo Missing source human log: %SOURCE_LOG%
  echo Usage: lab\scripts\import_human_sessions.bat [logs.txt]
  exit /b 1
)

python -m wsd.import_human_sessions ^
  --input-log "%SOURCE_LOG%" ^
  --format auto ^
  --archive-dir data\human_session_archives ^
  --live-log data\live_logs\access.log ^
  --labels-path data\live_labels\manual_labels.csv

endlocal
