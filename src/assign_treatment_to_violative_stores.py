import numpy as np
import pandas as pd


def load_and_prep_matched_warning_letters_to_stores(file_path: str) -> pd.DataFrame:
    """
    Load the matched warning letters to stores Excel file into a DataFrame.

    Args:
        file_path (str): Path to the matched warning letters to stores Excel file.
    Returns:
        pd.DataFrame: DataFrame containing the matched warning letters to stores data.
    """
    df = pd.read_excel(file_path)

    # subset matched stores
    df = df.copy()
    matched_stores = df.loc[df["matching"] == 1].copy()
    
    return matched_stores


def load_indexes_df(file_path: str) -> pd.DataFrame:
    """
    Load the baseline dataset (merged index panes) into a DataFrame.

    Args:
        file_path (str): Path to the baseline store information Excel file.
    Returns:
        pd.DataFrame: DataFrame containing the baseline store information.
    """
    baseline_df = pd.read_feather(file_path)

    return baseline_df


def assign_treatment_to_violative_stores(
    indexes_df: pd.DataFrame,
    matched_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Assign treatment groups to violative stores based on matched warning letters.

    Args:
        matched_df (pd.DataFrame): DataFrame containing matched warning letters to stores.
        treatment_mapping (dict[str, str]): Mapping of warning letter types to treatment groups.
    Returns:
        pd.DataFrame: DataFrame with assigned treatment groups.
    """
    # convert dates to year-month format
    matched_df['insp_date'] = pd.to_datetime(matched_df['insp_date']).dt.to_period('M')
    matched_df['issue_date'] = pd.to_datetime(matched_df['issue_date']).dt.to_period('M')

    indexes_df['date'] = indexes_df['date'].dt.to_period('M')

    # unique treatments per store-date
    insp_issue_date_df = matched_df[['store_id', 'insp_date', 'issue_date']].drop_duplicates()

    # merge to main df
    matched_df = pd.merge(
        indexes_df,
        insp_issue_date_df,
        left_on=['store_id', 'date'],
        right_on=['store_id', 'insp_date'],
        how='left',
        )

    # add treatment indicators
    matched_df['insp'] = np.where(
        ~matched_df['insp_date'].isna(), 1, 0
        )
    matched_df['issue'] = np.where(
        ~matched_df['issue_date'].isna(), 1, 0
        ) 
    
    return matched_df