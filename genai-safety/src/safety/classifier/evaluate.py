"""Classifier metrics. Pure numpy/sklearn so it is easy to unit test."""
from typing import Sequence, Tuple

import numpy as np


def recall_at_fpr(labels: Sequence[int], scores: Sequence[float],
                  max_fpr: float = 0.01) -> Tuple[float, float]:
    """Best recall with false-positive rate <= max_fpr.

    Returns (recall, threshold); predict positive when score > threshold.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], np.sort(scores[labels == 0])[::-1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    allowed_fp = int(np.floor(max_fpr * len(neg)))
    # Threshold sits at the (allowed_fp + 1)-th highest negative score, so at
    # most allowed_fp negatives are strictly above it (ties count against us).
    threshold = neg[allowed_fp] if allowed_fp < len(neg) else -np.inf
    return float((pos > threshold).mean()), float(threshold)


def classifier_report(labels: Sequence[int], scores: Sequence[float],
                      max_fpr: float = 0.01, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    preds = scores >= threshold
    tp = int((preds & (labels == 1)).sum())
    fp = int((preds & (labels == 0)).sum())
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    recall, thr = recall_at_fpr(labels, scores, max_fpr)
    both = n_pos > 0 and n_neg > 0
    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "pr_auc": float(average_precision_score(labels, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else float("nan"),
        "recall_at_max_fpr": recall, "threshold_at_max_fpr": thr,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / n_pos if n_pos else float("nan"),
        "fpr": fp / n_neg if n_neg else float("nan"),
    }
