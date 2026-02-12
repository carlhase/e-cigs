from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_threshold_curves(precision, recall, thresholds, best_thresh: float, best_f1: float, outpath: Path) -> None:
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1[:-1], label="F1 score", linewidth=2)
    plt.plot(thresholds, precision[:-1], "--", label="Precision")
    plt.plot(thresholds, recall[:-1], "--", label="Recall")

    plt.axvline(best_thresh, linestyle="--", label=f"Best threshold = {best_thresh:.2f}")
    plt.scatter([best_thresh], [best_f1], s=80, zorder=5)

    plt.title("Precision, Recall, and F1 vs. Classification Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, format="png", dpi=300)
    plt.close()


def feature_importances_df(preprocessor, model) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    return fi


def plot_feature_importances(fi: pd.DataFrame, outpath: Path, top_k: int = 15) -> None:
    top = fi.head(top_k).iloc[::-1]  # reverse for barh

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_k} Most Important Predictors")
    plt.tight_layout()
    plt.savefig(outpath, format="png", dpi=300)
    plt.close()
