"""H3 binning.

h3 wants (lat, lng). GeoJSON/MapLibre want [lng, lat]. Mixing them up mirrors
the whole country and it is not obvious on screen, so double check conversions.
"""

from __future__ import annotations

import h3
import pandas as pd


def assign_cells(
    df: pd.DataFrame, resolution: int, lat_col="latitude", lon_col="longitude"
) -> pd.Series:
    return pd.Series(
        [
            h3.latlng_to_cell(lat, lon, resolution)
            for lat, lon in zip(df[lat_col], df[lon_col], strict=True)
        ],
        index=df.index,
        name=f"h3_r{resolution}",
    )


def resolution_report(df: pd.DataFrame, resolutions: tuple[int, ...]) -> pd.DataFrame:
    """Stats to pick a resolution: fine = sharp but sparse, coarse = stable but blunt."""
    rows = []
    for res in resolutions:
        counts = assign_cells(df, res).value_counts()
        rows.append(
            {
                "res": res,
                "edge_km": round(h3.average_hexagon_edge_length(res, unit="km"), 2),
                "occupied_cells": len(counts),
                "median_per_cell": int(counts.median()),
                "p90_per_cell": int(counts.quantile(0.90)),
                "max_per_cell": int(counts.max()),
                # sparsity is the thing that kills the model, so track it directly
                "cells_with_1_pct": round((counts == 1).mean() * 100, 1),
                "cells_lt5_pct": round((counts < 5).mean() * 100, 1),
            }
        )
    return pd.DataFrame(rows)
