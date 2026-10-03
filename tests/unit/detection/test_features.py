from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secops.data import schema as s
from secops.detection.features import FEATURE_SPEC_V1, FEATURE_SPEC_V1_WITHPORT, FeatureSpec


def _frame(cols: list[str], n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(np.ones((n, len(cols)), dtype=np.float32), columns=cols)


def test_v1_spec_matches_schema() -> None:
    assert FEATURE_SPEC_V1.names == s.FEATURE_COLS
    assert FEATURE_SPEC_V1.version == "v1-noport"
    assert FEATURE_SPEC_V1_WITHPORT.names[-1] == s.PORT_COL


def test_to_matrix_selects_in_spec_order_from_wider_frame() -> None:
    df = _frame([*reversed(s.FEATURE_COLS), "label", "Dst Port"])
    X = FEATURE_SPEC_V1.to_matrix(df)
    assert X.shape == (3, 82) and X.dtype == np.float32


def test_feature_spec_rejects_reordered_columns() -> None:
    df = _frame(list(reversed(s.FEATURE_COLS)))
    with pytest.raises(ValueError, match="order"):
        FEATURE_SPEC_V1.to_matrix(df, strict_order=True)


def test_to_matrix_rejects_missing_column() -> None:
    df = _frame(s.FEATURE_COLS[:-1])
    with pytest.raises(ValueError, match="missing"):
        FEATURE_SPEC_V1.to_matrix(df)


def test_save_load_round_trip(tmp_path: Path) -> None:
    p = FEATURE_SPEC_V1.save(tmp_path / "fs.json")
    assert FeatureSpec.load(p) == FEATURE_SPEC_V1
