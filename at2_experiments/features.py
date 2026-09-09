"""Generic, dataset-agnostic feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_transforms(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add log1p-transformed versions of the given columns.

    Useful for right-skewed numeric columns. Does not modify the original
    columns; adds new '{col}_log' columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Column names to transform. Columns not present in df raise a
        KeyError, same as standard pandas indexing.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_log"] = np.log1p(df[col])
    return df