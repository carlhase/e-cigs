from __future__ import annotations

from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def build_model_pipeline(
        preprocess: ColumnTransformer,
        model_cfg: Any) -> Pipeline:
    """
    Builds the model pipeline
    
    Args:
        preprocess: column transformer pipeline created in src/ml/preprocess.py
        model_cfg: model configuration from YAML file
    Returns:
        A sklearn Pipeline for a random forest classifier
    """
    name = model_cfg["name"]        # get model name from model configuration yaml file
    if name != "random_forest":
        raise ValueError(f"Unsupported model: {name}")
    
    params = model_cfg["params"]    # get params mapping from model configuration yaml file
    model = RandomForestClassifier(**params) # **params unpacks the mapping as keyword arguments

    # construct pipeline
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model),
        ]
    )



