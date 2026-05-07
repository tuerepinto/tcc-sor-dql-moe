from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import numpy as np
import pandas as pd

L2_REQUIRED_COLS = (
    [f"ask_{i}" for i in range(1, 6)] +
    [f"vol_ask_{i}" for i in range(1, 6)]
)

def list_available_dates(root: str | Path, venue: str, symbol: str) -> List[str]:
    root = Path(root)
    base = root / f"venue={venue}" / f"symbol={symbol}"
    if not base.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {base}")

    dates = []
    for p in base.glob("date=*"):
        if p.is_dir():
            dates.append(p.name.split("date=")[-1])
    dates.sort()
    return dates

def load_l2_day_df(root: str | Path, venue: str, symbol: str, date: str, columns=None) -> pd.DataFrame:
    root = Path(root)
    path = root / f"venue={venue}" / f"symbol={symbol}" / f"date={date}"
    cols = columns or list(L2_REQUIRED_COLS)
    df = pd.read_parquet(path, columns=cols)
    return df.reset_index(drop=True)

def df_to_l2_numpy(df: pd.DataFrame, levels: int = 5) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for i in range(1, levels + 1):
        out[f"ask_{i}"] = df[f"ask_{i}"].to_numpy(dtype=np.float32, copy=False)
        out[f"vol_ask_{i}"] = df[f"vol_ask_{i}"].to_numpy(dtype=np.float32, copy=False)
    return out

def slice_l2_numpy(l2: Dict[str, np.ndarray], start: int, length: int) -> Dict[str, np.ndarray]:
    end = start + length
    return {k: v[start:end] for k, v in l2.items()}