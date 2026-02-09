from __future__ import annotations

from omegaconf import OmegaConf
from typing import Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


def _to_plain_py(obj: Any) -> Any:
    """
    Convert OmegaConf DictConfig/ListConfig (possibly nested) into plain Python
    dict/list scalars so sklearn can safely index them.
    """
    return OmegaConf.to_container(obj, resolve=True)


def build_search(
        pipe: Pipeline,
        search_cfg: Any
) -> RandomizedSearchCV | GridSearchCV:
    """
    Args:

    Returns:
        Either a RandomizedSearchCV or a GridSearchCV object
    """
    # get search method from search configuration yaml file
    method = search_cfg.get("method", "")

    # dictionary with parameters common to both methods
    common_kwargs = {
        "estimator": pipe,
        "scoring": str(search_cfg["scoring"]),
        "cv": int(search_cfg["cv"]),
        "verbose": int(search_cfg.get("verbose", 0)),
        "n_jobs": int(search_cfg.get("n_jobs", -1))
    }

    if method == "randomized":
        param_distributions = _to_plain_py(search_cfg["param_distributions"])
        return RandomizedSearchCV(
            **common_kwargs,
            param_distributions = param_distributions,
            # parameters specific to randomized search
            n_iter = int(search_cfg["n_iter"]), # defensive coerce to int type
            random_state=int(search_cfg["random_state"]),
        )
    
    if method == "grid":
        param_grid = _to_plain_py(search_cfg["param_grid"])
        return GridSearchCV(
            **common_kwargs,
            param_grid=dict(search_cfg["param_grid"])
        )
    
    raise ValueError(f"Unsupported search method: {method}. Use 'randomized' or 'grid'.")



    