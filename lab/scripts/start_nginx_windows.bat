@echo off
setlocal
set PORT=8039
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\lab\scripts\start_nginx_windows.ps1" -Port %PORT%
endlocal
