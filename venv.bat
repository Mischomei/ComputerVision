@echo off

:: Set working directory to the location of the script
cd /d "%~dp0"

set VENV_DIR=venv

:: Check if venv directory exists
if exist %VENV_DIR%\Scripts\activate.bat (
    echo Activating existing virtual environment...
) else (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)
call %VENV_DIR%\Scripts\activate.bat
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Virtual environment is ready.
cmd /k %VENV_DIR%\Scripts\activate.bat