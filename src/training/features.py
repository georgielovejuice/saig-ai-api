"""Which columns feed which model.

The two models get different features and it matters. The risk model predicts
(hex cell, weekly slot) for a future slot - at that point nobody knows what the
crash will be, what caused it, or what vehicle was involved. Those columns only
exist because a crash already happened, so feeding them to the risk model means
training on the answer.

Severity is a different question: given a crash happened, how bad was it. There
the crash-descriptive columns are legitimately available at analysis time.
"""

from __future__ import annotations

# Only known after someone investigated the crash. หลับใน (microsleep), เมาสุรา (drunk),
# อุปกรณ์ยานพาหนะบกพร่อง (vehicle defect) - you cannot know any of these in advance,
# and accident_type is literally a description of the crash that happened.
FORENSIC = ("cause", "accident_type")

# Outcome columns. injuries is the co-outcome of fatalities, never a feature.
OUTCOME = ("fatalities", "injuries")

# Risk model: per (cell, weekly slot). Everything here is either a property of the
# place or of the clock, both knowable for a slot that hasn't happened yet.
RISK_FEATURES = (
    "h3_cell",
    "province",
    "dow",
    "hour_block",
    "month",
    "is_holiday",
    "is_songkran",
    "is_newyear",
    # dataset 5 static priors, per cell, count(DISTINCT eid) by type
    "itic_accident_density",
    "itic_breakdown_density",
    "itic_flood_density",
    "itic_total_density",
    "has_itic_coverage",
    # modal road geometry for the cell, derived from history not from the row
    "cell_road_characteristic",
)

# Severity model: given a crash, was it fatal. Conditions at the time are fair game.
SEVERITY_FEATURES = (
    "province",
    "dow",
    "hour_block",
    "month",
    "is_holiday",
    "weather",
    "road_characteristic",
    "first_vehicle",
    "itic_accident_density",
    "has_itic_coverage",
)

# Kept for SHAP/descriptive work only. Not in SEVERITY_FEATURES because a severity
# model that uses cause is explaining, not predicting, and we should be honest
# about which one we are doing.
DESCRIPTIVE_ONLY = FORENSIC


def check_no_leakage(features: tuple[str, ...], model: str) -> None:
    """Raise if a feature list picked up an outcome or forensic column."""
    bad = [f for f in features if f in OUTCOME]
    if bad:
        raise ValueError(f"{model}: outcome column in features: {bad}")
    if model == "risk":
        bad = [f for f in features if f in FORENSIC]
        if bad:
            raise ValueError(f"risk model cannot use post-crash columns: {bad}")
