"""Evaluation Module for CreditGuard.

Computes classification metrics, ROC & Precision-Recall curves, confusion matrices,
and threshold-dependent trade-offs.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes standard binary classification metrics at a specific decision threshold.
    
    Args:
        y_true: True binary targets (0: non-default, 1: default).
        y_prob: Predicted default probability scores [0, 1].
        threshold: Decision boundary threshold (default: 0.50).
        
    Returns:
        Dict containing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
    }


def compute_confusion_matrix_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes detailed confusion matrix and lending business cost metrics.
    
    In lending:
    - TN: True Negatives (Reliable applicant correctly predicted reliable) -> Approved smoothly
    - FP: False Positives (Reliable applicant incorrectly flagged as high-risk) -> Extra review cost / friction
    - FN: False Negatives (Defaulter incorrectly approved as reliable) -> Severe loss of principal
    - TP: True Positives (Defaulter correctly flagged as high-risk) -> Loss avoided / mitigated
    
    Args:
        y_true: True binary targets.
        y_prob: Predicted probabilities.
        threshold: Decision boundary.
        
    Returns:
        Dict with confusion matrix counts and lending business interpretations.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    return {
        "matrix": cm.tolist(),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN_pct": float(tn / total * 100),
        "FP_pct": float(fp / total * 100),
        "FN_pct": float(fn / total * 100),
        "TP_pct": float(tp / total * 100),
        "threshold": float(threshold),
    }


def compute_curves_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Computes ROC and Precision-Recall curve coordinates for interactive charts.
    
    Args:
        y_true: True binary targets.
        y_prob: Predicted probabilities.
        
    Returns:
        Dict with fpr, tpr, roc_auc, precisions, recalls, pr_thresholds.
    """
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    precisions, recalls, pr_thresh = precision_recall_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresh.tolist(),
        "roc_auc": float(auc_score),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "pr_thresholds": pr_thresh.tolist(),
    }


def evaluate_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Full evaluation of a scikit-learn pipeline on test data.
    
    Args:
        pipeline: Trained pipeline object.
        X_test: Test features.
        y_test: Test true labels.
        threshold: Decision threshold.
        
    Returns:
        Dictionary of metrics, curves, and confusion matrix.
    """
    y_true = np.asarray(y_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold=threshold)
    cm_data = compute_confusion_matrix_data(y_true, y_prob, threshold=threshold)
    curves_data = compute_curves_data(y_true, y_prob)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_data,
        "curves": curves_data,
        "y_prob": y_prob.tolist(),
        "y_true": y_true.tolist(),
    }
