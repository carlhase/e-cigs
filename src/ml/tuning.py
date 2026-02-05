from __future__ import annotations

from typing import Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


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
        "scoring": search_cfg["scoring"],
        "cv": search_cfg["cv"],
        "verbose": search_cfg.get("verbose", 0),
        "n_jobs": search_cfg.get("n_jobs", -1)
    }

    if method == "randomized":
        return RandomizedSearchCV(
            **common_kwargs,
            param_distributions = dict(search_cfg["param_distributions"]),
            # parameters specific to randomized search
            n_iter = int(search_cfg["n_iter"]), # defensive coerce to int type
            random_state=int(search_cfg["random_state"]),
        )
    
    if method == "grid":
        return GridSearchCV(
            **common_kwargs,
            param_grid=dict(search_cfg["param_grid"])
        )
    
    raise ValueError(f"Unsupported search method: {method}. Use 'randomized' or 'grid'.")



    