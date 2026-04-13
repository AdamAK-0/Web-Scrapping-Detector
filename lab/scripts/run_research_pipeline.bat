@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..
cd /d "%PROJECT_ROOT%"
python -m wsd.train --data-dir data\prepared_live --prefixes 3 5 10 15 20
python -m wsd.experiment --data-dir data\prepared_live --prefixes 3 5 10 15 20 --hard-prefixes 3 5 10 --protocols session_split hard_prefix_session_split leave_one_bot_family_out time_split leave_one_human_user_out --group-key path_signature --n-bootstrap 300 --save-models
endlocal
