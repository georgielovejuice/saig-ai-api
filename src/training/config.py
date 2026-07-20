"""Central paths and constants.

Every path in the project resolves from here. Notebooks import this module;
they never hardcode a path, so moving the data means editing one file.

Raw data lives at the *workspace root* (`../data/`), outside this git repo,
because the CSVs total ~4.8 GB and must never enter version control.
Override with the ROADRISK_DATA_ROOT environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

DATA_ROOT = Path(os.getenv("ROADRISK_DATA_ROOT", WORKSPACE_ROOT / "data"))

# Dataset 1 — nationwide accidents, 2019-2025, 151,778 rows. PRIMARY training data.
ACCIDENTS_CSV = DATA_ROOT / "1_ข้อมูลอุบัติเหตุทางรถยนต์ในประเทศไทย" / "thai_accidental_dataset.csv"

# Dataset 5 — iTIC traffic events, 2021-2023. Bangkok-dominant (87% of events) but
# nationwide in extent, so coverage is a density measure, not a Bangkok boolean.
# 6,148,474 rows / ~4.8 GB across three files. DuckDB only — never pd.read_csv these.
# eid is NOT unique: those rows are only 78,690 events. Always count(DISTINCT eid).
ITIC_DIR = DATA_ROOT / "5_ข้อมูลเหตุการณ์ทางถนนในประเทศไทย"
ITIC_GLOB = str(ITIC_DIR / "traffic_event_*.csv")

# Dataset 2 — Lat Krabang road speeds. OPTIONAL stretch goal only.
SPEEDS_CSV = DATA_ROOT / "2_ข้อมูลความเร็วรถแถวลาดกระบัง" / "traffic_results.csv"

# Outputs (all gitignored)
INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MLRUNS_URI = f"file://{REPO_ROOT / 'mlruns'}"

# ------------------------------------------------------- modelling constants

# THE SPLIT IS TEMPORAL AND FIXED. Never random-split this data — a random split
# leaks future accidents into training and inflates every metric you report.
TRAIN_YEARS = (2019, 2023)  # inclusive
TEST_YEARS = (2024, 2025)  # inclusive

# Thailand bounding box. Dataset 1 has 10 rows with coordinates outside it (drop or fix).
TH_BBOX = {"min_lat": 5.6, "max_lat": 20.5, "min_lon": 97.3, "max_lon": 105.7}

# res 6 - ~3.7km edge, 6412 occupied cells, median 5 accidents per cell over the
# train period. res 5 is denser per cell but the hexes get big enough (~10km) that a
# hotspot stops meaning anything to a driver. res 7+ is too sparse, 66% of cells
# under 5 accidents.
H3_RESOLUTION = 6
H3_CANDIDATE_RESOLUTIONS = (5, 6, 7, 8)

# Dataset 5 coverage window. Priors built from these years are static: they describe
# what kind of place a cell is, not what is happening at time t.
ITIC_YEARS = (2021, 2023)
