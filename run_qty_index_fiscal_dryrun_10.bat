cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\
    echo Create it with: python -m venv .venv
    echo Then install deps with: .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate

set STORE_ROOT=D:\convenience_store\data\processed\LS_Otter

python -m scripts.run_vape_price_index ^
    --store-path "%STORE_ROOT%\da_store_id_monthly_ag_feather" ^
    --weight-basis fiscal ^
    --index-kind qty ^
    --limit 10000

pause
