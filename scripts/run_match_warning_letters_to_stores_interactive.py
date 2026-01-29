# -*- coding: utf-8 -*-
"""
Interactive Window DEV-mode runner for matching FDA warning letters to PDI Technology convenience stores.

DEV (Interactive Window):
    Run the # %% cells one by one in VS Code.
"""
# %% Interactive setup (required for VS Code Interactive Window)
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT

# Import pipeline functions
from src.match_warning_letters_to_stores import (
    build_warning_letter_panel,
    prep_store_info_df,
    merge_warning_letters_to_stores_by_zip,
    apply_address_cleaning,
    similarity_score,
)

# -----------------------------------------------------------------------------
# Shared path config (single source of truth)
# -----------------------------------------------------------------------------

def get_paths() -> dict[str, Path]:
    """
    Centralize all file/folder paths used by both dev cells and prod main().
    Returns:
        dict[str, Path]: Dictionary of relevant paths.
    """
    project_root = Path(__file__).resolve().parents[1]

    letter_path = project_root / "data" / "raw" / "warning_letters"
    warning_letters_panel_path = (
        project_root / "data" / "processed" / "warning_letters" / "warning_letters_panel.feather"
    )

    store_info_file_path = (
        project_root / "data" / "raw" / "pdi" / "STORE_STATUS_NEW-0.csv"
    )

    output_excel_path = (
        project_root / "data" / "processed" / "warning_letters_matched.xlsx"
    )

    debug_dir = project_root / "data" / "processed" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "letter_path": letter_path,
        "warning_letters_panel_path": warning_letters_panel_path,
        "store_info_file_path": store_info_file_path,
        "output_excel_path": output_excel_path,
        "debug_dir": debug_dir,
    }


def peek(df, name: str, n: int = 5) -> None:
    """Lightweight inspection helper for both dev and prod."""
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print("cols:", list(df.columns))
    print(df.head(n))


# =============================================================================
# DEV MODE (VS Code Interactive Window)
# =============================================================================

# %%
# Dev: setup paths
paths = get_paths()
paths

# %%
# Dev: build/load warning letters panel
warning_letters_df = build_warning_letter_panel(
    letter_path=str(paths["letter_path"]),
    letter_panel_output_file_path=str(paths["warning_letters_panel_path"]),
)
peek(warning_letters_df, "warning_letters_df")

# %%
# Dev: load + prep store info
store_info_df = prep_store_info_df(store_info_file_path=str(paths["store_info_file_path"]))
peek(store_info_df, "store_info_df")

# %%
# Dev: merge by ZIP (inspect columns!)
merged_df = merge_warning_letters_to_stores_by_zip(
    store_info_df=store_info_df,
    warning_letters_df=warning_letters_df,
)
peek(merged_df, "merged_df")

# Optional: save a debug snapshot you can open later
merged_df.to_feather(paths["debug_dir"] / "merged_by_zip.csv")

# %%
# Dev: clean addresses
merged_df_clean = apply_address_cleaning(
    merged_df,
    address_columns=["address_pdi", "address_fda"],
)
peek(merged_df_clean, "merged_df_clean")
merged_df_clean[["address_pdi", "address_fda"]].head(10)

# %%
# Dev: similarity scoring (can be slow; start small if needed)
# For dev: run on a subset first
tmp = merged_df_clean.head(4000).copy()
tmp["similarity_score"] = tmp.apply(
    lambda x: similarity_score(x["address_pdi"], x["address_fda"]),
    axis=1,
)
tmp["similarity_score"].describe()

# %%
# Dev: full similarity scoring + export
merged_df_clean["similarity_score"] = merged_df_clean.apply(
    lambda x: similarity_score(x["address_pdi"], x["address_fda"]),
    axis=1,
)

merged_df_clean.sort_values(by="similarity_score", ascending=False, inplace=True)
merged_df_clean.to_excel(paths["output_excel_path"], index=False)
print(f"Saved: {paths['output_excel_path']}")

# =============================================================================
# PROD MODE (python -m scripts.run_match_warning_letters_to_stores)