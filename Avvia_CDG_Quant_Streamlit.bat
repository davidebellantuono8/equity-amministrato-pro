@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CDG Quant Suite - Streamlit v3.2

echo ============================================================
echo   CDG Quant Suite - Streamlit v3.2
echo   Sector Rotation + Stock Score + Factor Leadership
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo Ambiente Python locale trovato. Verifico le librerie...
    ".venv\Scripts\python.exe" -c "import streamlit,pandas,numpy,openpyxl,sklearn,scipy" >nul 2>&1
    if not errorlevel 1 goto RUN
    echo Ambiente presente ma incompleto: installo/riparo le librerie...
    goto INSTALL
)

where py >nul 2>&1
if not errorlevel 1 goto CREATE_PY
where python >nul 2>&1
if not errorlevel 1 goto CREATE_PYTHON

echo ERRORE: Python non risulta installato o non e' nel PATH.
pause
exit /b 1

:CREATE_PY
echo Primo avvio: creo l'ambiente Python locale...
py -3 -m venv .venv
if errorlevel 1 goto ERROR
goto INSTALL

:CREATE_PYTHON
echo Primo avvio: creo l'ambiente Python locale...
python -m venv .venv
if errorlevel 1 goto ERROR
goto INSTALL

:INSTALL
echo.
echo Aggiorno pip e installo le librerie...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto ERROR
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto ERROR

:RUN
echo.
echo Avvio CDG Quant Suite v3.2 nel browser...
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address localhost
if errorlevel 1 goto ERROR
exit /b 0

:ERROR
echo.
echo ERRORE durante la preparazione o l'avvio.
echo Se necessario elimina SOLO la cartella .venv e riavvia il BAT.
pause
exit /b 1
