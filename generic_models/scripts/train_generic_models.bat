@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
cd /d "%PROJECT_ROOT%"

set PYTHON_EXE=python
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe

"%PYTHON_EXE%" generic_models\train_generic_models.py ^
  --artifacts-dir generic_models\artifacts ^
  --external-dir generic_models\data\external\zenodo ^
  --download-public ^
  --num-sites 72 ^
  --sessions-per-site 90

endlocal
