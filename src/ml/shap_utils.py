from __future__ import annotations
 # safety feature needed for pyplt
import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def compute_and_plot_shap(best, X_test, outpath: Path) -> None:
    """
    Uses SHAP to compute explanations for the positive class and saves a summary plot.
    Args:
        best: defined in src/ml/pipeline.py
        X_test: test set defined in pipeline.py
        oupath: file path to save figure
    """
    import shap  # keep import local (optional dependency)

    pre = best.named_steps["preprocess"]
    model = best.named_steps["model"]

    Xt = pre.transform(X_test) # apply preprocessing to test set
    X_used = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(
        model, 
        data=X_used,                # background dataset
        model_output="probability"  # explain probabilities
        )
    
    # for binary classification, shap_values is typically a list [class0, class1]
    shap_values = explainer.shap_values(X_used, check_additivity=False)

     # Normalize to a 2D array for the positive class
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values
        if vals.ndim == 3:
            vals = vals[:, :, 1]


    if vals.shape != X_used.shape:
        raise ValueError(f"SHAP values shape {vals.shape} != X_used shape {X_used.shape}")

    plt.figure()
    shap.summary_plot(vals, X_used, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(outpath, format="png", dpi=300)
    plt.close()
