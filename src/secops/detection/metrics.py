"""Detection metrics. PR-AUC first; accuracy is reported but never a headline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    brier: float
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class ThresholdChoice:
    threshold: float
    fpr: float
    recall: float
    method: str


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_metrics(y_true: ArrayLike, y_prob: ArrayLike, threshold: float) -> BinaryMetrics:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    single_class = len(np.unique(y)) < 2
    return BinaryMetrics(
        pr_auc=float("nan") if single_class else float(average_precision_score(y, p)),
        roc_auc=float("nan") if single_class else float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=_safe_div(fp, fp + tn),
        accuracy=_safe_div(tp + tn, len(y)),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def _strictest(p: np.ndarray) -> ThresholdChoice:
    t = float(np.nextafter(p.max(), np.inf))
    return ThresholdChoice(t, 0.0, 0.0, "fallback_max_threshold")


def select_threshold(
    y_true: ArrayLike, y_prob: ArrayLike, max_fpr: float = 0.01
) -> ThresholdChoice:
    """Highest recall whose FPR <= max_fpr; ties -> higher threshold; fallback if none."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    fpr, tpr, thr = roc_curve(y, p)
    ok = fpr <= max_fpr
    if not ok.any():
        return _strictest(p)
    best_tpr = tpr[ok].max()
    cands = np.where(ok & (tpr == best_tpr))[0]
    i = cands[np.argmax(thr[cands])]
    t = float(min(thr[i], 1.0))  # roc_curve's first threshold is +inf
    m = binary_metrics(y, p, t)
    if m.fpr > max_fpr or m.recall == 0.0:
        return _strictest(p)
    return ThresholdChoice(t, m.fpr, m.recall, "max_recall_at_fpr")


def max_f1_threshold(y_true: ArrayLike, y_prob: ArrayLike) -> ThresholdChoice:
    """Threshold maximising F1, computed along the precision-recall curve (O(n log n))."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, p)
    # The last (precision, recall) point has no threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    t = float(thresholds[int(np.argmax(f1))])
    m = binary_metrics(y, p, t)
    return ThresholdChoice(t, m.fpr, m.recall, "max_f1")


def recall_at_fprs(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    fprs: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05),
) -> dict[str, float]:
    return {f"recall_at_fpr_{f}": select_threshold(y_true, y_prob, f).recall for f in fprs}


def multiclass_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_prob: ArrayLike, classes: list[str]
) -> dict[str, Any]:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    per_class = {
        c: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i, c in enumerate(classes)
    }
    return {
        "macro_f1": float(f1_score(yt, yp, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=classes, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(yt, np.asarray(y_prob), labels=classes)),
        "per_class": per_class,
        "confusion": confusion_matrix(yt, yp, labels=classes).tolist(),
    }


def per_group_recall(
    y_true: ArrayLike, y_prob: ArrayLike, threshold: float, groups: ArrayLike
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    g = np.asarray(groups).astype(str)
    out: dict[str, float] = {}
    for name in np.unique(g):
        mask = (g == name) & (y == 1)
        if mask.any():
            out[str(name)] = float(pred[mask].mean())
    return out
