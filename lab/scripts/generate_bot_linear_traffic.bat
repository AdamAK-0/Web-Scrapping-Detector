@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..
cd /d "%PROJECT_ROOT%"
python -m wsd.lab_traffic --mode linear --sessions 12 --base-url http://127.0.0.1:8039 --real-sleep --labels-path data\live_labels\manual_labels.csv
endlocal
