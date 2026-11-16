"""Utilidades de entrada/salida."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)


def write_csv(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs) -> Path:
    p = Path(path)
    ensure_parent_dir(p)
    df.to_csv(p, index=index, **kwargs)
    return p


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> tuple[list[str], list[str]]:
    cols = list(df.columns)
    req = list(required)
    missing = [c for c in req if c not in cols]
    extra = [c for c in cols if c not in req]
    return missing, extra


def file_exists(path: Path | str) -> bool:
    return Path(path).exists()
