"""
This module defines functions to match warning letters to stores based on street address
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import glob
import re
import unicodedata
from fuzzywuzzy import fuzz


#%% Load and prep warning letters dataframes

def list_warning_letter_files(letter_path: str) -> list[str]:
    """
    List all warning letter files in the specified directory.

    Args:
        letter_path (str): Path to the directory containing warning letter files.
    Returns:
        list[str]: List of file paths to warning letter files.
    """
    file_pattern = os.path.join(letter_path, "*announcement.xlsx")
    file_list = glob.glob(file_pattern)
    file_list = [os.path.normpath(p) for p in file_list]
    return file_list


def load_and_prep_warning_letter_file(file_path: str) -> pd.DataFrame:
    """
    Load a warning letter Excel file into a DataFrame.

    Args:
        file_path (str): Path to the warning letter Excel file.
    Returns:
        pd.DataFrame: DataFrame containing the warning letter data.
    """
    df = pd.read_excel(file_path)
    # lowercase columns
    df.columns = df.columns.str.lower()
    # rename columns
    cols_names = ['store_type', 'insp_date', 'issue_date', 'store_name', 'address', 'city', 'state', 'zip_code', 'products']
    df.columns = cols_names
    
    # standardize dtypes
    str_cols = ['store_type', 'store_name', 'address', 'city', 'state', 'zip_code', 'products']
    df[str_cols] = df[str_cols].astype('str')
    
    df['zip_code'] = df['zip_code'].str.replace('\"','')

    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

    # drop online stores
    df = df.loc[~df['store_type'].isin(['Online', 'online'])].copy()

    return df


def build_warning_letter_panel(
        letter_path: str,
        letter_panel_output_file_path: str,
        ) -> pd.DataFrame:
    """
    Build a combined DataFrame of all warning letters in the specified directory.

    Args:
        letter_path (str): Path to the directory containing warning letter files.
        output_path (str): File path for the combined warning letter feather file.
    Returns:
        pd.DataFrame: Concatinated warning letters DataFrame.
    """
    
    letter_panel_output_file_path = os.path.normpath(letter_panel_output_file_path)
    os.makedirs(os.path.dirname(letter_panel_output_file_path), exist_ok=True)

    file_list = list_warning_letter_files(letter_path)
    
    df_list = []
    for file_path in file_list:
        df = load_and_prep_warning_letter_file(file_path)
        df_list.append(df)
    
    if not df_list:
        raise ValueError(f"No warning letter files found in directory: {letter_path}")

    combined_df = pd.concat(df_list, ignore_index=True)

    # save panel
    combined_df.to_feather(letter_panel_output_file_path)

    return combined_df

#%% Load and prep PDI Technologies Store Info DataFrame

def prep_store_info_df(store_info_file_path: str) -> pd.DataFrame:
    """
    Load PDI Technologies convenience store info csv file and prepare
    the store info DataFrame for matching.

    Args:
        store_info_file_path (str): Path to the store info csv file.
    Returns:
        pd.DataFrame: Prepared store info DataFrame.
    """
    df = pd.read_csv(store_info_file_path, dtype = {'ZIP_CODE': 'str'})
    
    # lowercase columns
    df.columns = df.columns.str.lower()

    # rename columns
    df = df.rename(
        columns = {
            'store_chain_name': 'store_chain_name_pdi',
            'street_address': 'address'
            }
        )

    # standardize dtypes
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df[str_cols] = df[str_cols].astype('str')

    # clean up columns
    df['zip_code'] = df['zip_code'].str.replace(r'-.*', '')

    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

    return df


def load_warning_letters_panel(letter_panel_path: str) -> pd.DataFrame:
    """
    Load the warning letter panel DataFrame from a feather file.

    Args:
        letter_panel_path (str): Path to the warning letter panel feather file.
    """
    return pd.read_feather(letter_panel_path)

# as a first step, match letters to stores by zip code
def merge_warning_letters_to_stores_by_zip(
        store_info_df: pd.DataFrame,
        warning_letters_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Merge warning letters to store info based on address matching.

    Args:
        store_info_df (pd.DataFrame): DataFrame containing store info.
        warning_letters_df (pd.DataFrame): DataFrame containing warning letters.
    Returns:
        pd.DataFrame: Merged DataFrame with warning letters matched to stores.
    """
    merged_df = pd.merge(
        store_info_df,
        warning_letters_df,
        on=['zip_code'],
        how='left',
        suffixes=('_pdi', '_fda')
        )

    # subset columns
    stores_matched_on_zips = merged_df[
        ['address_pdi', 
         'address_fda', 
         'store_id', 
         'store_name_pdi', 
         'store_chain_name_pdi', 
         'store_name_fda', 
         'insp_date', 
         'issue_date', 
         'zip_code']
         ]

    return stores_matched_on_zips

#%% Clean and standardize addresses

# Basic cleaning: remove accents, special characters, extra spaces
def basic_clean_address(text):
    if pd.isnull(text):
        return None
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s,]', '', text)  # keep only words, spaces, commas
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()

# Expand abbreviations (only for address/street name)
def expand_street_abbreviations(text):
    if pd.isnull(text):
        return None
    abbrev_mapping = {
        r'\bStreet\b': 'St',
        r'\bSt.\b': 'St',
        r'\bAvenue\b': 'Ave',
        r'\bAve.\b': 'Ave',
        r'\bBoulevard\b': 'Blvd',
        r'\bBlvd.\b': 'Blvd',
        r'\bRoad\b': 'Rd',
        r'\bRd.\b': 'Rd',
        r'\bDrive\b': 'Dr',
        r'\bDr.\b': 'Dr',
        r'\bLn\b': 'Lane',
        r'\bLn.\b': 'Lane',
        r'\bPl\b': 'Place',
        r'\bPl.\b': 'Place',
        r'\bCt\b': 'Court',
        r'\bCt.\b': 'Court',
        r'\bHighway\b': 'Hwy',
        r'\bsuite\b.*': '',
        r'\bSuite\b.*': '',
        r'\bste\b.*': '',
        r'\bSTE\b.*': '',
        r'\,': '',
        r'\bNortheast\b': 'NE',
        r'\bNorthwest\b': 'NW',
        r'\bSoutheast\b': 'SE',
        r'\bSouthwest\b': 'SW',
        r'\bNorth east\b': 'NE',
        r'\bNorth west\b': 'NW',
        r'\bSouth east\b': 'SE',
        r'\bSouth west\b': 'SW',
        r'\bNorth\b': 'N',
        r'\bSouth\b': 'S',
        r'\bEast\b': 'E',
        r'\bWest\b': 'W',
        r'\bN(?:\.)?\b': 'N',
        r'\bS(?:\.)?\b': 'S',
        r'\bE(?:\.)?\b': 'E',
        r'\bW(?:\.)?\b': 'W',
        r'\bNE(?:\.)?\b': 'NE',
        r'\bNW(?:\.)?\b': 'NW',
        r'\bSE(?:\.)?\b': 'SE',
        r'\bSW(?:\.)?\b': 'SW',
        r'\bN E(?:\.)?\b': 'NE',
        r'\bN W(?:\.)?\b': 'NW',
        r'\bS E(?:\.)?\b': 'SE',
        r'\bS W(?:\.)?\b': 'SW',

    }
    
    for abbrev, full in abbrev_mapping.items():
        text = re.sub(abbrev, full, text, flags=re.IGNORECASE)
    
    return text

# apply cleaning functions to address columns
def apply_address_cleaning(
        df: pd.DataFrame,
        address_columns: list[str]
    ) -> pd.DataFrame:
    """
    Apply cleaning functions to specified address columns in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing address columns.
        address_columns (list[str]): List of address column names to clean.
    Returns:
        pd.DataFrame: DataFrame with cleaned address columns.
    """
    df_cleaned = df.copy()
    for col in address_columns:
        df_cleaned[col] = df_cleaned[col].apply(basic_clean_address)
        df_cleaned[col] = df_cleaned[col].apply(expand_street_abbreviations)
    return df_cleaned

#%% Fuzzy matching

# function to calculate similarity score between two strings
def similarity_score(col1, col2):
    if pd.isna(col1) or pd.isna(col2):
        return np.nan
    # standardize case and strip whitespace
    col1 = col1.lower().strip()
    col2 = col2.lower().strip()

    return fuzz.token_sort_ratio(col1, col2)