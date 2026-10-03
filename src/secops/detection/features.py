"""Feature specification: the exact ordered list of columns a model consumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from secops.data.schema import FEATURE_COLS, PORT_COL


class FeatureSpec(BaseModel):
    names: list[str]
    version: str

    def to_matrix(self, df: pd.DataFrame, strict_order: bool = False) -> np.ndarray:
        missing = [c for c in self.names if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:10]}")
        if strict_order:
            wanted = set(self.names)
            present = [c for c in df.columns if c in wanted]
            if present != self.names:
                raise ValueError("feature columns are not in spec order")
        return df[self.names].to_numpy(dtype=np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        return cls.model_validate_json(path.read_text())


FEATURE_SPEC_V1 = FeatureSpec(names=list(FEATURE_COLS), version="v1-noport")
FEATURE_SPEC_V1_WITHPORT = FeatureSpec(names=[*FEATURE_COLS, PORT_COL], version="v1-withport")
FEATURE_SPECS: dict[str, FeatureSpec] = {
    fs.version: fs for fs in (FEATURE_SPEC_V1, FEATURE_SPEC_V1_WITHPORT)
}
