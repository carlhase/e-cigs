@echo off
python -m scripts.run_vape_price_index ^
    --store-path data/interim/da_store_id_monthly_ag/ ^
    --outpath data/processed/store_vape_price_indexes/ ^
    --panel-output-path data/processed/index_panels/vape_price_indexes.feather ^
    --limit 10
pause

