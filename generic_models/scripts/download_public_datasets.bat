@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
cd /d "%PROJECT_ROOT%"

set PYTHON_EXE=python
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe

"%PYTHON_EXE%" generic_models\train_generic_models.py ^
  --artifacts-dir generic_models\artifacts_download_check ^
  --external-dir generic_models\data\external\zenodo ^
  --download-public ^
  --skip-public ^
  --num-sites 1 ^
  --sessions-per-site 2 ^
  --model-set logistic_regression

if exist generic_models\artifacts_download_check rmdir /s /q generic_models\artifacts_download_check

endlocal
