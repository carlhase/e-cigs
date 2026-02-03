"""
Runner: assigns treatment to violative stores based on matched FDA warning letters

- Inputs:  
    data/processed/warning_letters_matched.xlsx
    data/processed/merged_index_panels.feather
- Outputs: 
    data/processed/violators_insp_issue.feather
"""

# Imports and paths
from pathlib import Path

from src.assign_treatment_to_violative_stores import (
    load_and_prep_matched_warning_letters_to_stores,
    load_indexes_df,
    assign_treatment_to_violative_stores
) 

def main() -> None:
    # Project root = parent of scripts/ (robust no matter where its run from)
    project_root = Path(__file__).resolve().parents[1]

    matched_letters_and_stores_path = project_root / "data" / "processed" / "warning_letters_matched.xlsx"
    indexes_panel_path = project_root / "data" / "processed" / "index_panels" / "vape_qty_indexes_fiscal.feather" # yet to be created
    output_file_path = project_root / "data" / "processed" / "violators_insp_issue.feather"

    # load datasets
    matched_df = load_and_prep_matched_warning_letters_to_stores(file_path=str(matched_letters_and_stores_path))
    indexes_panel = load_indexes_df(file_path=str(indexes_panel_path))

    # merge dataframes and assign treatment
    treated_df = assign_treatment_to_violative_stores(
        indexes_df=indexes_panel,
        matched_df=matched_df
    )

    # export as feather
    treated_df.to_feather(output_file_path)

    print(f"Exported treated violative stores data to {output_file_path} with {len(treated_df):,} rows.")
    print(f"Insp treatments: {len(treated_df.loc[treated_df['insp'] == 1]):,}")
    print(f"Issue treatments: {len(treated_df.loc[treated_df['issue'] == 1]):,}")

if __name__ == "__main__":
    main()