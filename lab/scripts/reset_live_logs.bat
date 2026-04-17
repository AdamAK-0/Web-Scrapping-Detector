@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
if not exist "%PROJECT_ROOT%\data\live_logs" mkdir "%PROJECT_ROOT%\data\live_logs"
if exist "%PROJECT_ROOT%\data\live_logs\access.log" del /q "%PROJECT_ROOT%\data\live_logs\access.log"
if exist "%PROJECT_ROOT%\data\live_logs\error.log" del /q "%PROJECT_ROOT%\data\live_logs\error.log"
type nul > "%PROJECT_ROOT%\data\live_logs\access.log"
type nul > "%PROJECT_ROOT%\data\live_logs\error.log"
if exist "%PROJECT_ROOT%\data\live_labels\manual_labels.csv" del /q "%PROJECT_ROOT%\data\live_labels\manual_labels.csv"
if exist "%PROJECT_ROOT%\data\admin_bot_runs" rmdir /s /q "%PROJECT_ROOT%\data\admin_bot_runs"
echo Live logs, label file, and admin bot-run logs reset.
echo Archived real-human sessions under data\human_session_archives were kept.
endlocal
