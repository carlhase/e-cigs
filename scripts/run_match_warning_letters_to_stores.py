"""
Runner: build FDA warning letter panel, load PDI Technologies store info, and merge two two on zip code

- Inputs:  
    data/raw/warning_letters
    data/raw/pdi/STORE_STATUS_NEW-0.csv
- Outputs: 
    data/processed/warning_letters/warning_letters_panel.feather
    store_letter_match_by_zip.xlsx
"""

# Imports and paths
from pathlib import Path

from src.match_warning_letters_to_stores import (
    build_warning_letter_panel, 
    prep_store_info_df,
    merge_warning_letters_to_stores_by_zip,
    apply_address_cleaning,
    similarity_score
)

    
def main() -> None:
    # Project root = parent of scripts/ (robust no matter where its run from)
    project_root = Path(__file__).resolve().parents[1]

    letter_path = project_root / "data" / "raw" / "warning_letters"
    letter_panel_output_file_path = project_root / "data" / "processed" / "warning_letters" / "warning_letters_panel.feather"

    # build the warning letter panel
    warning_letters_df = build_warning_letter_panel(
        letter_path=str(letter_path),
        letter_panel_output_file_path=str(letter_panel_output_file_path)
    )   

    warning_letters_df.head()
    warning_letters_df.columns
    print(f"Built warning letter panel with {len(warning_letters_df):,} rows.")
    
    # load and prep PDI store info
    store_info_file_path = project_root / "data" / "raw" / "pdi" / "STORE_STATUS_NEW-0.csv"  # specify the correct path here
    
    store_info_df = prep_store_info_df(store_info_file_path=str(store_info_file_path))
    
    store_info_df.head()
    store_info_df.columns
    print(f"Loaded and prepared store info with {len(store_info_df):,} rows.")

    # merge warning letters to stores by zip code
    merged_df = merge_warning_letters_to_stores_by_zip(
        store_info_df=store_info_df,
        warning_letters_df=warning_letters_df
    )

    merged_df.head()
    merged_df.columns
    merged_df.shape

    # Clean addresses
    merged_df_clean = apply_address_cleaning(
        merged_df, 
        address_columns = ['address_pdi', 'address_fda']
        )
    
    merged_df_clean[["address_pdi", "address_fda"]].head(10)

    # compute similarity scores
    merged_df_clean['similarity_score'] = (
        merged_df_clean
        .apply(lambda x: similarity_score(x['address_pdi'], x['address_fda']), axis=1)
    )

    merged_df_clean.sort_values(by="similarity_score", ascending=False, inplace=True)
    merged_df_clean["similarity_score"].describe()
    merged_df_clean.head(10)

    # save to excel
    output_excel_path = project_root / "data" / "processed" / "warning_letters_matched.xlsx"
    merged_df_clean.to_excel(output_excel_path, index=False)

    print(f"Merged warning letters to stores, df has {len(merged_df_clean):,} rows.")

if __name__ == "__main__":
    main()