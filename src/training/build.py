"""Runs the whole phase 1 pipeline and writes the model-ready tables.

    uv run python -m src.training.build
"""

from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR
from .data import clean_accidents, load_accidents
from .datasets import build_risk_table, build_severity_table, split_years
from .itic import load_events
from .normalize import NORM_COLS, apply_frequency_floor, apply_mapping, build_mapping, check_mapping
from .priors import attach_priors, build_priors

REVIEW_CSV = "docs/category_normalization_review.csv"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_accidents()
    acc, clean_report = clean_accidents(raw)
    print("clean:", clean_report)

    review = pd.read_csv(REVIEW_CSV)
    mapping = build_mapping(review, {c: acc[c].value_counts() for c in NORM_COLS})
    problems = check_mapping(review, mapping)
    if problems:
        raise SystemExit("mapping conflicts:\n" + "\n".join(problems))
    acc, floor_report = apply_frequency_floor(apply_mapping(acc, mapping), 50)
    print(floor_report.to_string(index=False))

    priors = build_priors(load_events())
    priors.to_parquet(PROCESSED_DIR / "itic_priors.parquet", index=False)

    train, test = split_years(acc)
    for name, part in (("train", train), ("test", test)):
        sev = build_severity_table(part)
        sev = attach_priors(sev, priors)
        sev.to_parquet(PROCESSED_DIR / f"severity_{name}.parquet", index=False)

        risk = attach_priors(build_risk_table(part), priors)
        risk.to_parquet(PROCESSED_DIR / f"risk_{name}.parquet", index=False)
        print(f"{name}: severity {len(sev):,} rows, risk {len(risk):,} rows")


if __name__ == "__main__":
    main()
