import numpy as np
import pytest

from secops.detection.metrics import (
    binary_metrics,
    max_f1_threshold,
    multiclass_metrics,
    per_group_recall,
    recall_at_fprs,
    select_threshold,
)

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])


def test_binary_metrics_hand_computed() -> None:
    m = binary_metrics(Y, P, threshold=0.5)
    # >= 0.5: idx 2 (FP), 4, 5, 7 (TP); idx 6 is FN
    assert (m.tp, m.fp, m.tn, m.fn) == (3, 1, 3, 1)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.75)
    assert m.fpr == pytest.approx(0.25)
    assert m.accuracy == pytest.approx(0.75)
    assert 0 < m.pr_auc <= 1 and 0 < m.roc_auc <= 1
    assert set(m.to_dict()) >= {"pr_auc", "roc_auc", "recall", "fpr", "threshold"}


def test_select_threshold_respects_fpr_budget() -> None:
    c = select_threshold(Y, P, max_fpr=0.0)
    m = binary_metrics(Y, P, c.threshold)
    assert m.fpr == 0.0
    assert c.method == "max_recall_at_fpr"
    assert c.recall == pytest.approx(m.recall)


def test_select_threshold_picks_highest_recall_under_budget() -> None:
    c = select_threshold(Y, P, max_fpr=0.25)
    assert binary_metrics(Y, P, c.threshold).fpr <= 0.25
    assert c.recall >= select_threshold(Y, P, max_fpr=0.0).recall


def test_select_threshold_fallback_when_no_threshold_meets_fpr() -> None:
    y = np.array([0, 1])
    p = np.array([0.9, 0.9])  # catching the positive always catches the negative
    c = select_threshold(y, p, max_fpr=0.0)
    assert c.method == "fallback_max_threshold"
    assert binary_metrics(y, p, c.threshold).fp == 0


def test_max_f1_threshold_and_recall_at_fprs() -> None:
    c = max_f1_threshold(Y, P)
    assert 0 < c.threshold <= 1 and c.method == "max_f1"
    r = recall_at_fprs(Y, P, fprs=(0.0, 0.25, 1.0))
    assert r["recall_at_fpr_1.0"] == 1.0
    assert r["recall_at_fpr_0.0"] <= r["recall_at_fpr_0.25"] <= 1.0


def test_multiclass_metrics_shape() -> None:
    classes = ["a", "b", "c"]
    yt = np.array(["a", "b", "c", "a"])
    yp = np.array(["a", "b", "a", "a"])
    prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.7, 0.2, 0.1]])
    m = multiclass_metrics(yt, yp, prob, classes)
    assert set(m["per_class"]) == set(classes)
    assert m["per_class"]["a"]["support"] == 2
    assert len(m["confusion"]) == 3 and len(m["confusion"][0]) == 3
    assert 0 <= m["macro_f1"] <= 1 and m["log_loss"] > 0


def test_per_group_recall() -> None:
    groups = np.array(["x", "x", "y", "y", "x", "x", "y", "y"])
    r = per_group_recall(Y, P, 0.5, groups)
    assert r["x"] == pytest.approx(1.0)  # positives in x: idx 4, 5 -> both >= 0.5
    assert r["y"] == pytest.approx(0.5)  # positives in y: idx 6 (0.4), 7 (0.7)


def test_max_f1_threshold_matches_brute_force_and_scales() -> None:
    import time

    rng = np.random.default_rng(1)
    y_small = rng.integers(0, 2, size=400)
    p_small = np.clip(rng.normal(0.5 + 0.3 * y_small, 0.25), 0, 1)
    grid = np.unique(p_small)
    f1s = [binary_metrics(y_small, p_small, t).f1 for t in grid]
    brute = float(grid[int(np.argmax(f1s))])
    assert max_f1_threshold(y_small, p_small).threshold == pytest.approx(brute)

    y_big = rng.integers(0, 2, size=200_000)
    p_big = np.clip(rng.normal(0.5 + 0.3 * y_big, 0.25), 0, 1)
    start = time.perf_counter()
    c = max_f1_threshold(y_big, p_big)
    assert time.perf_counter() - start < 5.0, "max_f1_threshold must be O(n log n), not O(n^2)"
    assert 0 < c.threshold <= 1
