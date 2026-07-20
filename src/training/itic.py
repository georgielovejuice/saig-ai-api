"""Dataset 5 (iTIC events) via duckdb.

The three csvs are 6.1M rows but only 78,690 events - they're periodic snapshots
of whatever was active at poll time, so a long-running roadworks shows up
thousands of times. Dedup by eid or every count is wrong.

start_time/stop_time come back as VARCHAR because the column mixes
'2021-01-01 21:21:20' with date-only '2023-01-05', hence the try_cast.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from .config import ITIC_GLOB

# 0.2% of eids ever move or change type, so which snapshot we keep barely matters.
# Taking the first one seen.
DEDUP_SQL = """
WITH raw AS (
    SELECT
        eid, title_th, description, latitude, longitude, type,
        try_cast(start_time AS TIMESTAMP) AS start_time,
        try_cast(stop_time  AS TIMESTAMP) AS stop_time
    FROM read_csv(?, header=true, union_by_name=true, sample_size=-1)
),
ranked AS (
    SELECT *, row_number() OVER (PARTITION BY eid ORDER BY start_time, stop_time) AS rn
    FROM raw
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
      AND latitude BETWEEN 5.6 AND 20.5 AND longitude BETWEEN 97.3 AND 105.7
      AND start_time IS NOT NULL
      -- 18.6k rows have stop before start, and 5.3k are test rows
      AND NOT (stop_time IS NOT NULL AND stop_time < start_time)
      AND lower(coalesce(title_th, '')) NOT LIKE '%test%'
      AND lower(coalesce(description, '')) NOT LIKE '%test%'
)
SELECT eid, title_th, description, latitude, longitude, type, start_time, stop_time
FROM ranked WHERE rn = 1
"""


def load_events(glob: str = ITIC_GLOB) -> pd.DataFrame:
    """One row per eid. Small enough for pandas after this."""
    con = duckdb.connect()
    return con.execute(DEDUP_SQL, [glob]).df()
