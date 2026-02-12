from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve,
)


@dataclass(frozen=True)
class ThresholdResult:
    best_thresh: float
    best_f1: float
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


def optimal_threshold_by_f1(y_true, y_prob) -> ThresholdResult:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length n-1 relative to prec/rec
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    ix = int(np.nanargmax(f1[:-1]))  # align with thresholds length
    return ThresholdResult(
        best_thresh=float(thresholds[ix]),
        best_f1=float(f1[ix]),
        precision=prec,
        recall=rec,
        thresholds=thresholds,
    )


def evaluate_at_threshold(y_true, y_prob, thresh: float) -> dict:
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),                        # JSON safe
        "classification_report": classification_report(
            y_true, y_pred, digits=3, output_dict=True
            ),                                                  # JSON safe
    }
