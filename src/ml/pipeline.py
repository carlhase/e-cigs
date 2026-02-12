from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import json
from omegaconf import OmegaConf

from src.ml.features import make_xy
from src.ml.preprocess import split_num_cat_columns, build_preprocessor_from_cfg
from src.ml.model import build_model_pipeline
from src.ml.tuning import build_search
from src.ml.metrics import optimal_threshold_by_f1, evaluate_at_threshold
from src.ml.plots import plot_threshold_curves, feature_importances_df, plot_feature_importances
from src.ml.shap_utils import compute_and_plot_shap


def run_training(
        cfg: Any,
        df: pd.DataFrame,
        outdir: Path
) -> dict:
    """
    This is essentially an orchestrator function that calls the collaborators with the 
    right inputs, in the right order, and returns summary objects.
    Args:
        cfg: the DictConfig for Hydra to process (a specific yaml file in the conf directory)
        df: DataFrame with features and target
        outdir: relative path specified in the config.yaml file
    Returns:
        Returns summary objects from running the ML pipeline
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # log the config
    (outdir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    X, y = make_xy(
        df,
        x_cols = cfg.features.x_cols,
        y_col = cfg.data.target_col
        )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = float(cfg.train.test_size),
        random_state = int(cfg.train.random_state),
        stratify = y if bool(cfg.train.stratify) else None,
        )
    

    # preprocess on training data
    num_cols, cat_cols = split_num_cat_columns(
        X_train,
        force_categorical= cfg.features.force_categorical,
    )

    preprocess = build_preprocessor_from_cfg(
        num_cols = num_cols,
        cat_cols = cat_cols,
        preprocess_cfg = cfg.preprocess
    )

    # build model pipeline
    pipe = build_model_pipeline(preprocess = preprocess, model_cfg = cfg.model)

    # hyperparameter tuning
    search = build_search(pipe = pipe, search_cfg = cfg.search)
    search.fit(X_train, y_train)

    best = search.best_estimator_ # object storing best hyperparameter set

    # Build configuration-specific filename (not necessary but makes future comparison easier)
    run_tag = f"{cfg.data.name}_{cfg.search.name}_{cfg.features.name}"
    
    # save trained model
    joblib.dump(best, outdir / f"{run_tag}.joblib")

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    # use my trained classifier to output predicted probabilities (not just class labels) and the optimal threshold
    y_prob = best.predict_proba(X_test)[:, 1]
    thr = optimal_threshold_by_f1(y_test, y_prob)
    eval_opt = evaluate_at_threshold(y_test, y_prob, thresh=thr.best_thresh)

    # Save evaluation summary to file
    training_and_eval_summary = {
        "run_tag": run_tag,
        "model_artifact": f"{run_tag}.joblib",
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "threshold_best": thr.best_thresh,  # already float
        "threshold_best_f1": thr.best_f1,   # already float
        "eval_opt": eval_opt,
    }
    (outdir / "training_and_eval_summary.json").write_text(json.dumps(training_and_eval_summary, indent=2))

    # ------------------------------------------------------------------
    # Plot: threshold curves
    # ------------------------------------------------------------------
    plot_threshold_curves(
        thr.precision, thr.recall, thr.thresholds,
        thr.best_thresh, thr.best_f1,
        outpath=outdir / "classification_thresholds.png", # generic filename because the folder is configuration-specific
    )
    
    # ------------------------------------------------------------------
    # Plot: Feature importances
    # ------------------------------------------------------------------
    pre = best.named_steps["preprocess"]
    model = best.named_steps["model"]
    fi = feature_importances_df(pre, model)
    fi.to_csv(outdir / "feature_importances.csv", index=False)
    plot_feature_importances(fi, outdir / "feature_importances.png", top_k=15)

    # ------------------------------------------------------------------
    # Plot: SHAP
    # ------------------------------------------------------------------
    compute_and_plot_shap(
        best, 
        X_test, 
        outpath=outdir / "shap_summary.png"
        )

    # return summary to print later on during run
    return training_and_eval_summary
    
    # store various objects to print later on during run
"""     return {
        "best_estimator": best,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "threshold_best": thr.best_thresh,
        "threshold_best_f1": thr.best_f1,
        "eval_opt": eval_opt,
        }
 """    
