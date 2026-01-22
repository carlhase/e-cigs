@echo off
REM Ensure we are running from the project root (directory of this .bat file)
cd /d "%~dp0"

REM Fail fast if the virtual environment does not exist
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\
    echo Create it with: python -m venv .venv
    echo Then install deps with: .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate the environment
call .venv\Scripts\activate

REM Your existing environment variables (if any)
set STORE_ROOT=D:\convenience_store\data\processed\LS_Otter

REM Run
python -m scripts.run_vape_price_index ^
    --store-path "%STORE_ROOT%\da_store_id_monthly_ag_feather" ^
    --outpath data\processed\store_vape_qty_indexes_calendar ^
    --panel-output-path data\processed\index_panels\vape_qty_indexes_calendar.feather ^
    --weight-basis calendar

pause
