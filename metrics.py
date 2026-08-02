import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc(y_true, y_scores):
    # Modern scikit-learn no longer raises on a single-class y_true — it warns
    # and returns nan instead. Guard both cases so a skewed split degrades to
    # 0.0 rather than propagating nan into logs/checkpoint comparisons.
    try:
        score = roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.0
    return 0.0 if np.isnan(score) else float(score)

def compute_tdr(y_true, y_scores, fpr_threshold=0.1):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
    except ValueError:
        return 0.0
    mask = fpr <= fpr_threshold
    if not np.any(mask):
        return 0.0
    result = np.max(tpr[mask])
    return 0.0 if np.isnan(result) else float(result)
