from __future__ import annotations

import pandas as pd
from typing import Sequence, Tuple


def make_xy(
        df: pd.Dataframe,
        x_cols: Sequence[str],
        y_col: str
        ) -> Tuple[pd.DataFrame, pd.Series]:
    
    # check for missing columns
    missing = set(x_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing X columns in df: {sorted(missing)}")
    if y_col not in df.columns:
        raise KeyError(f"Missing target column '{y_col}' in df.")
    
    X = df[list(x_cols)].copy()
    y = df[y_col].copy()

    return X, y