"""Builds the two model-ready tables.

Severity: one row per accident, target is fatal/not.
Risk: one row per (cell, weekly slot), target is accident count.

Both split 2019-2023 train / 2024-2025 test. Temporal, never shuffled.
"""

from __future__ import annotations

import pandas as pd

from .config import H3_RESOLUTION, TEST_YEARS, TRAIN_YEARS
from .spatial import assign_cells

# Songkran and new year are the two big road-fatality windows in Thailand.
# Approximate fixed dates - good enough, the real "seven dangerous days" shift a bit
# year to year but not by much.
SONGKRAN = [(4, 13), (4, 14), (4, 15)]
NEWYEAR = [(12, 31), (1, 1), (1, 2)]


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    t = df[time_col]
    out = df.copy()
    out["dow"] = t.dt.dayofweek
    out["hour_block"] = t.dt.hour // 3
    out["month"] = t.dt.month
    md = list(zip(t.dt.month, t.dt.day, strict=True))
    out["is_songkran"] = [int(x in SONGKRAN) for x in md]
    out["is_newyear"] = [int(x in NEWYEAR) for x in md]
    out["is_holiday"] = (out["is_songkran"] | out["is_newyear"]).astype(int)
    return out


def build_severity_table(accidents: pd.DataFrame, resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """One row per accident. Target: was anyone killed."""
    df = add_time_features(accidents)
    df["h3_cell"] = assign_cells(df, resolution)
    df["is_fatal"] = (df["fatalities"] > 0).astype(int)
    return df


def build_risk_table(accidents: pd.DataFrame, resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """(cell, dow, hour_block) grid with accident counts.

    Grid is restricted to cells that ever saw an accident. Covering all of Thailand
    would mean mostly sea and forest, and the model would spend its capacity
    learning that nothing happens in the ocean.

    Counting over the 5 train years as a recurring weekly profile rather than per
    date - per (cell, date) is 99% zeros and the model just predicts 0 everywhere.
    """
    df = add_time_features(accidents)
    df["h3_cell"] = assign_cells(df, resolution)

    counts = (
        df.groupby(["h3_cell", "dow", "hour_block"]).size().rename("accident_count").reset_index()
    )
    # fill the slots that never had one, they're real zeros not missing rows
    cells = counts["h3_cell"].unique()
    full = pd.MultiIndex.from_product(
        [cells, range(7), range(8)], names=["h3_cell", "dow", "hour_block"]
    ).to_frame(index=False)
    grid = full.merge(counts, on=["h3_cell", "dow", "hour_block"], how="left")
    grid["accident_count"] = grid["accident_count"].fillna(0).astype(int)
    return grid


def split_years(df: pd.DataFrame, time_col: str = "time"):
    year = df[time_col].dt.year
    return df[year.between(*TRAIN_YEARS)], df[year.between(*TEST_YEARS)]
