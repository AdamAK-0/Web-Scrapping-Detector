@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
if not exist "%PROJECT_ROOT%\data\live_logs" mkdir "%PROJECT_ROOT%\data\live_logs"
if exist "%PROJECT_ROOT%\data\live_logs\access.log" del /q "%PROJECT_ROOT%\data\live_logs\access.log"
if exist "%PROJECT_ROOT%\data\live_logs\error.log" del /q "%PROJECT_ROOT%\data\live_logs\error.log"
if exist "%PROJECT_ROOT%\data\live_labels\manual_labels.csv" del /q "%PROJECT_ROOT%\data\live_labels\manual_labels.csv"
echo Live logs and label file reset.
endlocal
