from pathlib import Path

import pytest
from pydantic import ValidationError

from secops.detection.train import TrainConfig


def _yaml(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text("experiment: e\nrun_name: r\ntask: binary\nmodel: lightgbm\nweighting: balanced\n")
    return p


def test_unknown_override_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="weihgting"):
        TrainConfig.from_yaml(_yaml(tmp_path), {"weihgting": "none"})


def test_override_applies_and_empty_means_none(tmp_path: Path) -> None:
    cfg = TrainConfig.from_yaml(
        _yaml(tmp_path), {"weighting": "none", "register_as": "", "max_fpr": "0.02"}
    )
    assert cfg.weighting == "none"
    assert cfg.register_as is None
    assert cfg.max_fpr == 0.02
