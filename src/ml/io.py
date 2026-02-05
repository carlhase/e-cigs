from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Any


def load_df_from_cfg(data_cfg: Any, project_root: Path) -> pd.DataFrame:
    path = project_root / str(data_cfg.input_path)
    fmt = str(data_cfg.format).lower()

    if fmt == "feather":
        return pd.read_feather(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {fmt}")
