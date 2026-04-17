@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
cd /d "%PROJECT_ROOT%"

set PYTHON_EXE=python
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe

"%PYTHON_EXE%" generic_models\train_generic_models.py ^
  --artifacts-dir generic_models\artifacts_fast ^
  --external-dir generic_models\data\external\zenodo ^
  --download-public ^
  --num-sites 28 ^
  --sessions-per-site 50 ^
  --model-set logistic_regression random_forest extra_trees hist_gradient_boosting lightgbm

endlocal
