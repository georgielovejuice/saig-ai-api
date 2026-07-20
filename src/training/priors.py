"""Static per-cell priors from dataset 5.

Static on purpose. Dataset 5 stops in 2023 and we test on 2024-25, so anything
keyed on time would be missing at inference and would leak in training. These
describe what kind of place a cell is, which doesn't expire.

Counts are events (already deduped by eid upstream). Not rows - rows would make
this a roadworks-duration proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import H3_RESOLUTION
from .spatial import assign_cells

# the types worth their own column, rest gets folded into the total
PRIOR_TYPES = {
    "อุบัติเหตุ": "itic_accident_density",
    "รถเสีย/กีดขวาง": "itic_breakdown_density",
    "น้ำท่วม": "itic_flood_density",
}

# below this an empty cell is more likely "nobody was watching" than "nothing happens"
COVERAGE_MIN_EVENTS = 3


def build_priors(events: pd.DataFrame, resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """One row per cell that has any iTIC event."""
    ev = events.copy()
    ev["h3_cell"] = assign_cells(ev, resolution)

    out = pd.DataFrame({"itic_total_density": ev.groupby("h3_cell").size()})
    for thai, col in PRIOR_TYPES.items():
        out[col] = ev[ev["type"] == thai].groupby("h3_cell").size()
    out = out.fillna(0).astype(int)

    # coverage is a density thing, not a bangkok flag. iTIC does reach the provinces,
    # just thinly - a cell with 3 events isn't uncovered, it's badly sampled. zero
    # means unknown either way, never safe.
    out["has_itic_coverage"] = (out["itic_total_density"] >= COVERAGE_MIN_EVENTS).astype(int)
    # log because the bangkok cells are orders of magnitude denser than everything else
    out["itic_density_log"] = np.log1p(out["itic_total_density"])
    return out.reset_index()


def attach_priors(cells: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Left join. Cells with no iTIC events get 0 density and coverage 0."""
    out = cells.merge(priors, on="h3_cell", how="left")
    dens = [c for c in priors.columns if c != "h3_cell"]
    out[dens] = out[dens].fillna(0)
    return out
