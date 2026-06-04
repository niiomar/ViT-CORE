import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(y_true, y_scores):
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.0


def compute_tdr(y_true, y_scores, fpr_threshold=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = fpr <= fpr_threshold
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))
