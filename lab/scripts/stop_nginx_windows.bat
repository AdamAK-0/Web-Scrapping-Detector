@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\lab\scripts\stop_nginx_windows.ps1" -Port 8039
endlocal
