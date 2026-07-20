"""Thai category normalization.

Merges come from the hand-reviewed pair list. Applied as connected components,
not pairwise - values chain (a~b, b~c) and doing it pairwise in dict order
orphans whichever one gets visited last.
"""

from __future__ import annotations

import pandas as pd

NORM_COLS = ("cause", "accident_type", "road_characteristic")
RARE_LABEL = "อื่นๆ (rare)"

# Highest-count wins normally, but for the brake cluster the most common spelling is
# the wrong one (เบรคกระทันหัน, 2 rows vs 1 each for the others). Kim picked the correct
# spelling. Whole cluster is 5 rows so it ends up in the rare bucket anyway - this is
# really just so the label is right if anyone reads it.
CANONICAL_OVERRIDES = {
    "cause": {"เบรกกะทันหัน"},
    "accident_type": {"อื่นๆ (ชนสัตว์)"},
}


def _components(edges: list[tuple[str, str]]) -> list[set[str]]:
    """Union-find, minus the find. Small input so the merge-sets version is fine."""
    comps: list[set[str]] = []
    for a, b in edges:
        hit = [c for c in comps if a in c or b in c]
        merged = {a, b}
        for c in hit:
            merged |= c
            comps.remove(c)
        comps.append(merged)
    return comps


def build_mapping(review: pd.DataFrame, counts: dict[str, pd.Series]) -> dict[str, dict[str, str]]:
    """{column: {raw_value: canonical}}. Canonical is the most frequent member."""
    mapping: dict[str, dict[str, str]] = {}
    for col in NORM_COLS:
        yes = review[(review["column"] == col) & (review["MERGE_yes_no"].str.upper() == "YES")]
        edges = list(zip(yes["value_a"], yes["value_b"], strict=True))
        vc = counts[col]
        overrides = CANONICAL_OVERRIDES.get(col, set())
        col_map = {}
        for comp in _components(edges):
            forced = comp & overrides
            canon = forced.pop() if forced else max(comp, key=lambda v: vc.get(v, 0))
            for v in comp:
                col_map[v] = canon
        mapping[col] = col_map
    return mapping


def check_mapping(review: pd.DataFrame, mapping: dict[str, dict[str, str]]) -> list[str]:
    """Every value must land on exactly one canonical. Returns problems, empty is good."""
    problems = []
    for col in NORM_COLS:
        seen: dict[str, set[str]] = {}
        for raw, canon in mapping[col].items():
            seen.setdefault(raw, set()).add(canon)
        for raw, canons in seen.items():
            if len(canons) > 1:
                problems.append(f"{col}: {raw!r} -> {sorted(canons)}")
        # canonical must not itself be remapped, otherwise the chain didn't close
        for canon in mapping[col].values():
            if canon in mapping[col] and mapping[col][canon] != canon:
                problems.append(
                    f"{col}: canonical {canon!r} is itself mapped to {mapping[col][canon]!r}"
                )
    return sorted(set(problems))


def apply_mapping(df: pd.DataFrame, mapping: dict[str, dict[str, str]]) -> pd.DataFrame:
    out = df.copy()
    for col in NORM_COLS:
        out[col] = out[col].map(lambda v, m=mapping[col]: m.get(v, v))
    return out


def apply_frequency_floor(
    df: pd.DataFrame, floor: int = 50, cols=NORM_COLS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anything under `floor` rows becomes an explicit rare bucket.

    A category in <50 of 151k rows can't teach the model anything, and one-hot
    encoding a long tail of them just inflates the feature space.
    """
    out = df.copy()
    report = []
    for col in cols:
        vc = out[col].value_counts()
        rare = set(vc[vc < floor].index)
        before = out[col].nunique(dropna=True)
        out[col] = out[col].map(lambda v, r=rare: RARE_LABEL if v in r else v)
        moved = out[col].eq(RARE_LABEL).sum()
        report.append(
            {
                "column": col,
                "categories_before": before,
                "categories_after": out[col].nunique(dropna=True),
                "rare_categories_bucketed": len(rare),
                "rows_in_other": int(moved),
                "pct_rows_in_other": round(100 * moved / len(out), 2),
            }
        )
    return out, pd.DataFrame(report)
