"""Shared paths, study settings and state-name normalisation."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
TEMPLATES = ROOT / "templates"
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

YEARS = list(range(2018, 2026))  # study window: calendar years 2018-2025

for _p in (RAW, INTERIM, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def states() -> pd.DataFrame:
    return pd.read_csv(CONFIG / "states.csv", dtype=str).fillna("")


def _norm(name: str) -> str:
    s = str(name).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z ]", " ", s)          # drop digits, dots, hyphens, brackets
    s = re.sub(r"\b(the|state|ut|u t)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for _, r in states().iterrows():
        keys = [r.state_name, r.state_code, r.phonepe_slug.replace("-", " ")]
        keys += [a for a in r.aliases.split("|") if a]
        for k in keys:
            m[_norm(k)] = r.state_code
    return m


def to_state_code(name: str) -> str | None:
    """Map any common spelling of a state/UT name to the project's 2-letter code.

    Returns None for totals ("All India", "Total") and unknown names, so callers
    can drop or report them instead of silently mis-assigning rows.
    """
    n = _norm(name)
    if n in {"all india", "india", "total", "grand total", ""}:
        return None
    return _alias_map().get(n)


def add_state_code(df: pd.DataFrame, col: str = "state") -> tuple[pd.DataFrame, list[str]]:
    """Add a state_code column; return the frame and the list of unmatched names."""
    df = df.copy()
    df["state_code"] = df[col].map(to_state_code)
    unmatched = sorted(set(df.loc[df["state_code"].isna(), col].astype(str)) -
                       {"All India", "India", "Total", "Grand Total", ""})
    return df, unmatched


def skeleton() -> pd.DataFrame:
    """Empty 36 states x 8 years frame that every source is merged onto."""
    s = states()[["state_code", "state_name", "type", "region"]]
    return s.merge(pd.DataFrame({"year": YEARS}), how="cross")
