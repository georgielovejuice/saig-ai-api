"""Dataset 1 (accidents) loading + cleaning.

Small enough for pandas. Dataset 5 is not - that goes through duckdb.
"""

from __future__ import annotations

import pandas as pd

from .config import ACCIDENTS_CSV, TEST_YEARS, TH_BBOX, TRAIN_YEARS


def load_accidents(path=ACCIDENTS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


def flag_out_of_bbox(df: pd.DataFrame) -> pd.Series:
    # 10 rows in the raw file. flag instead of dropping inline so we can count them
    return ~(
        df["latitude"].between(TH_BBOX["min_lat"], TH_BBOX["max_lat"])
        & df["longitude"].between(TH_BBOX["min_lon"], TH_BBOX["max_lon"])
    )


def clean_accidents(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unusable coords. Returns (df, report)."""
    # not touching missing categoricals here - missingness looks agency-dependent,
    # so it becomes an explicit "unknown" level at encoding time instead
    bad = flag_out_of_bbox(df)
    report = {"rows_in": len(df), "dropped_out_of_bbox": int(bad.sum())}
    out = df.loc[~bad].copy()
    report["rows_out"] = len(out)
    return out, report


def temporal_split(df: pd.DataFrame, time_col: str = "time") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train 2019-2023, test 2024-2025. Never shuffle this."""
    year = df[time_col].dt.year
    return df[year.between(*TRAIN_YEARS)], df[year.between(*TEST_YEARS)]
