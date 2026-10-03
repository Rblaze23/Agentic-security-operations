from pathlib import Path

import numpy as np
import pandas as pd

from secops.detection.plots import (
    confusion_matrix_figure,
    pr_curve_figure,
    save_figure,
    shap_bar_figure,
)


def test_figures_render_and_save(tmp_path: Path) -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    assert save_figure(pr_curve_figure(y, p, 0.5), tmp_path / "pr.png").stat().st_size > 0
    cm = confusion_matrix_figure([[2, 0], [1, 1]], ["benign", "attack"])
    assert save_figure(cm, tmp_path / "cm.png").exists()
    imp = pd.DataFrame({"feature": ["a", "b"], "mean_abs_shap": [0.5, 0.2]})
    assert save_figure(shap_bar_figure(imp), tmp_path / "shap.png").exists()
