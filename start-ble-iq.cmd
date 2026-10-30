@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0ble_iq_explorer.py" %*
) else (
    python "%~dp0ble_iq_explorer.py" %*
)
if errorlevel 1 pause
endlocal
