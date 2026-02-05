"""
Unit tests for the build_model_pipeline() function
Tests:
    1: If model name != random_forest, the function raises a ValueError
    2: If params are note defined, the function raises a KeyError
    3. The function returns a pipeline, with two steps in correct order, and 
    the exact objects are passed through the pipeline (no duplication)
    4. The pipeline applies the parameters it is passed
"""

import pytest

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.ml.model import build_model_pipeline


def _dummy_preprocessor() -> ColumnTransformer:
    # Minimal, valid ColumnTransformer. I don't need it to do anything for these tests.
    return ColumnTransformer(transformers=[], remainder="drop")


def test_build_model_pipeline_raises_on_unknown_model():
    preprocess = _dummy_preprocessor()
    
    # sample model config mimicking YAML structure
    cfg = {
        "name": "xgboost", 
        "params": {}
    }

    with pytest.raises(ValueError, match=r"Unsupported model: xgboost"):
        build_model_pipeline(preprocess, cfg)


def test_build_model_pipeline_raises_error_if_missing_params_key():
    preprocess = _dummy_preprocessor()
    
    # config missing params key
    cfg = {"name": "random_forest"}

    # passing a cfg without the params defined should raise an error
    with pytest.raises(KeyError, match=r"params"):
        build_model_pipeline(preprocess, cfg)


def test_build_model_pipeline_returns_pipeline_with_expected_steps():
    preprocess = _dummy_preprocessor
    # sample model config mimicking YAML structure
    cfg = {"name": "random_forest", "params": {}}

    # apply build_model_pipeline function
    pipe = build_model_pipeline(preprocess, cfg)

    assert isinstance(pipe, Pipeline), "build_model_pipeline should return a pipeline"
    assert [name for name, _ in pipe.steps] == ["preprocess", "model"], (
        "pipeline should have two steps in this order: preprocess, model"
    )
    
    # Identity check: exact object passed through, i.e. no unexpected duplication or 
    # transformation of the preprocessor occurred during pipeline construction
    assert pipe.named_steps["preprocess"] is preprocess
    assert isinstance(pipe.named_steps["model"], RandomForestClassifier)


def test_build_model_pipeline_applies_rf_params():
    preprocess = _dummy_preprocessor()
    # # sample model config with various parameters
    cfg = {
        "name": "random_forest",
        "params": {
            "n_estimators": 321,
            "max_depth": 7,
            "random_state": 42,
            "n_jobs": 2,
        },
    }

    pipe = build_model_pipeline(preprocess, cfg)
    model = pipe.named_steps["model"]

    assert model.n_estimators == 321
    assert model.max_depth == 7
    assert model.random_state == 42
    assert model.n_jobs == 2






