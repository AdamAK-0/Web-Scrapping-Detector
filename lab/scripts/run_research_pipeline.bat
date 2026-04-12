@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..
cd /d "%PROJECT_ROOT%"
python -m wsd.train --data-dir data\prepared_live --prefixes 3 5 10 15 20
python -m wsd.experiment --data-dir data\prepared_live --prefixes 3 5 10 15 20 --n-bootstrap 300 --save-models
endlocal
