from __future__ import annotations

from typing import Sequence, Tuple, Any
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_num_cat_columns(
        X: pd.DataFrame,
        force_categorical: Sequence[str] = (),
) -> Tuple[list[str], list[str]]:
    """
    Split dataframe columns into numerical and categorical lists.
    This function will be called in the orchestration layer (not in thus module).

    Args:
        X: Input dataframe.
        force_categorical: List of column names to force as categorical.
    Returns:
        A tuple of two lists: (numerical_columns, categorical_columns).
    """
    
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = list(X.select_dtypes(exclude=["number"]).columns)

    for c in force_categorical:         # force_categorical is specified in conf/ml/features/*.yaml
        if c in num_cols:
            num_cols.remove(c)
        if c in X.columns and c not in cat_cols:
            cat_cols.append(c)

    return num_cols, cat_cols


def _build_numerical_pipe(cfg_num: Any) -> Pipeline | None:
    """
    Build a preprocessing pipeline for numerical features.

    Args:
        cfg_num: Configuration for numerical preprocessing.
    Returns:
        A sklearn Pipeline for numerical preprocessing, or None if no steps are specified.
    """
    if cfg_num is None:
        return None
    
    steps = []

    # imputer step
    imputer_cfg = cfg_num["imputer"]    # reads info from yaml file into python as data
    if imputer_cfg is not None:
        steps.append((
            "imputer",
            SimpleImputer(
                strategy=imputer_cfg["strategy"],
                fill_value=imputer_cfg["fill_value"],
            )
        ))

    # scaler step
    scaler_cfg = cfg_num["scaler"]  # reads info from yaml file into python as data
    kind = scaler_cfg["kind"]
    
    # add scaler step to list or skip altogether
    if kind == "standard":
        steps.append(("scaler", StandardScaler()))
    elif kind == "none":
        pass  # no scaling
    else:
        raise ValueError(f"Unsupported scaler kind: {kind}")

    # If no steps, return None so we can "passthrough"
    return Pipeline(steps) if steps else None


def _build_categorical_pipe(cfg_cat: Any) -> Pipeline | None:
    """
    Build a preprocessing pipeline for categorical features.

    Args:
        cfg_cat: Configuration for categorical preprocessing.
    Returns:
        A sklearn Pipeline for categorical preprocessing, or None if no steps are specified.
    """
    if cfg_cat is None:
        return None
    
    steps = []

    # imputer step
    imputer_cfg = cfg_cat["imputer"]   
    if imputer_cfg is not None:
        steps.append((
            "imputer",
            SimpleImputer(
                strategy=imputer_cfg["strategy"],
                fill_value=imputer_cfg["fill_value"],
            )
        ))

    # encoder step
    encoder_cfg = cfg_cat["onehot"]
    if encoder_cfg is not None:
        steps.append((
            "onehot",
            OneHotEncoder(
                handle_unknown=encoder_cfg["handle_unknown"],
                sparse_output=encoder_cfg["sparse_output"],
            )
        ))

    return Pipeline(steps) if steps else None

def build_preprocessor_from_cfg(
        num_cols: Sequence[str], 
        cat_cols: Sequence[str], 
        preprocess_cfg: Any
        ) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numerical and categorical features.
    Note: does not call split_num_cat_columns; num_cols and cat_cols must be provided.
    Args:
        num_cols: List of numerical column names.
        cat_cols: List of categorical column names.
        preprocess_cfg: Configuration for preprocessing.
    Returns:
        A sklearn ColumnTransformer for preprocessing.
        """
    num_pipe = _build_numerical_pipe(preprocess_cfg["numerical"])
    cat_pipe = _build_categorical_pipe(preprocess_cfg["categorical"])

    # list of transformers
    transformers = []

    transformers.append(("num", num_pipe if num_pipe is not None else "passthrough", num_cols))
    transformers.append(("cat", cat_pipe if cat_pipe is not None else "passthrough", cat_cols))

    return ColumnTransformer(transformers)



