"""Diagnostic figures logged to MLflow: single-series, one axis, recessive grid, thin marks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

INK = "#1f2933"
MUTED = "#8a94a6"
GRID = "#e5e9f0"
SERIES = "#2f6fed"


def _style(ax: Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def pr_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Figure:
    p, r, t = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, lw=2, color=SERIES)
    i = min(int(np.searchsorted(t, threshold)), len(r) - 1) if len(t) else 0
    ax.scatter([r[i]], [p[i]], s=40, zorder=3, color=SERIES, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"threshold = {threshold:.3f}",
        (r[i], p[i]),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    ax.set_xlabel("Recall", color=INK)
    ax.set_ylabel("Precision", color=INK)
    ax.set_title("Precision-recall (test split)", color=INK, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(cm: list[list[int]], labels: list[str]) -> Figure:
    m = np.asarray(cm)
    size = 1.2 * len(labels) + 2
    fig, ax = plt.subplots(figsize=(size, size - 0.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=INK)
    threshold = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            colour = "white" if m[i, j] > threshold else INK
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=8, color=colour)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Actual", color=INK)
    fig.tight_layout()
    return fig


def shap_bar_figure(importance: pd.DataFrame, top_n: int = 20) -> Figure:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(top) + 1.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=SERIES, height=0.6)
    ax.set_xlabel("mean |SHAP value|", color=INK)
    ax.set_title(f"Top {len(top)} features by mean |SHAP|", color=INK, fontsize=10)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
