"""Templates, loading and validation for sources that have no download API.

Most Indian government dashboards (e-Taal, DBT Bharat, MGNREGA MIS, Annavitran,
ABDM, CPGRAMS, NeSDA, TRAI, RBI, PMJDY, NFHS) publish state tables as web pages,
PDFs or Excel files. For each of them the workflow is:

  1. python -m src.collect.manual_sources templates
        writes templates/<source>.csv : one row per state x year, blank value columns
  2. Copy the template to data/raw/<source>/<source>.csv and fill it in from the
     official report (instructions: config/manual_sources.csv, column where_to_find).
     Put the report itself (PDF/XLSX/screenshot) in the same folder as evidence and
     write its file name in the source_document column.
  3. python -m src.collect.manual_sources load
        validates every filled file and writes data/interim/<source>.csv
        plus reports/validation/<source>.txt

Missing-value codes in the filled files:
  (blank) = not collected yet      NR = source does not publish the value for that year
  NS      = scheme/indicator does not operate in that unit (structural; excluded, not imputed)
Each value's status is written to data/interim/_status_<source>.csv.

Validation checks: unknown state names, years outside the study window, values
outside the plausible range in config/manual_sources.csv, duplicate rows, and
coverage. Pre-2020 rows for "Dadra and Nagar Haveli" and "Daman and Diu" are
combined into the merged UT (sum for counts, mean for rates).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import CONFIG, INTERIM, RAW, REPORTS, TEMPLATES, YEARS, add_state_code, states  # noqa: E402

VALDIR = REPORTS / "validation"


def spec() -> pd.DataFrame:
    return pd.read_csv(CONFIG / "manual_sources.csv")


def _years(text: str) -> list[int]:
    """'2018-2025' -> [2018..2025]; '2019, 2021, 2023' -> [2019, 2021, 2023]."""
    out: list[int] = []
    for part in str(text).split(","):
        m = re.match(r"\s*(\d{4})\s*-\s*(\d{4})\s*$", part)
        out += list(range(int(m[1]), int(m[2]) + 1)) if m else [int(part)]
    return [y for y in out if y in YEARS]


def make_templates() -> None:
    TEMPLATES.mkdir(exist_ok=True)
    st = states()[["state_name"]]
    for source, g in spec().groupby("source", sort=False):
        years = sorted(set(sum((_years(y) for y in g.years), [])))
        t = st.merge(pd.DataFrame({"year": years}), how="cross")
        for v in g.variable:
            t[v] = ""
        t["source_document"] = ""
        t["notes"] = ""
        t.to_csv(TEMPLATES / f"{source}.csv", index=False)
        (RAW / source).mkdir(parents=True, exist_ok=True)
    print(f"Templates written to {TEMPLATES} for {spec().source.nunique()} sources")


def _read(source: str) -> pd.DataFrame | None:
    folder = RAW / source
    for name in (f"{source}.csv", f"{source}.xlsx"):
        p = folder / name
        if p.exists():
            return pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
    return None


def load_one(source: str, g: pd.DataFrame) -> dict:
    df = _read(source)
    status = {"source": source, "variables": ", ".join(g.variable), "status": "not filled yet",
              "rows_with_data": 0, "issues": 0}
    if df is None:
        return status
    log: list[str] = []
    vars_ = [v for v in g.variable if v in df.columns]
    missing_cols = sorted(set(g.variable) - set(vars_))
    if missing_cols:
        log.append(f"Missing columns (left empty): {missing_cols}")

    df, unmatched = add_state_code(df, "state_name")
    if unmatched:
        log.append(f"Unrecognised state names dropped: {unmatched} - add them as aliases in config/states.csv")
    df = df[df.state_code.notna()]

    bad_years = sorted(set(df.year) - set(YEARS))
    if bad_years:
        log.append(f"Rows outside 2018-2025 dropped: years {bad_years}")
    df = df[df.year.isin(YEARS)]

    status_rows = []
    for _, s in g[g.variable.isin(vars_)].iterrows():
        raw = df[s.variable].astype(str).str.strip()
        code = raw.str.upper()
        col = pd.to_numeric(raw.str.replace(",", "").replace(
            {"": None, "nan": None, "NaN": None, "None": None}), errors="coerce")
        st = pd.Series("blank", index=df.index)
        st[code.eq("NR")] = "not_reported"        # source does not publish it for that year
        st[code.eq("NS")] = "not_applicable"      # scheme/indicator does not exist in that unit
        st[col.notna()] = "ok"
        invalid = ~code.isin(["", "NAN", "NONE", "NR", "NS"]) & col.isna()
        st[invalid] = "invalid"
        if invalid.any():
            log.append(f"{s.variable}: {int(invalid.sum())} non-numeric value(s) set to missing "
                       f"(allowed codes: blank, NR, NS): {sorted(set(raw[invalid]))[:5]}")
        out = col.notna() & ((col < s["min"]) | (col > s["max"]))
        if out.any():
            rows = df.loc[out, ["state_code", "year"]].assign(value=col[out]).to_string(index=False)
            log.append(f"{s.variable}: {int(out.sum())} value(s) outside [{s['min']}, {s['max']}] {s.unit} were set to missing - check them in the source report:\n{rows}")
            col = col.mask(out)
            st[out] = "out_of_range"
        df[s.variable] = col
        status_rows.append(pd.DataFrame({"state_code": df.state_code, "year": df.year,
                                         "variable": s.variable, "status": st}))

    # combine duplicates (e.g. DNH + Daman & Diu before 2020)
    dup = df.duplicated(["state_code", "year"], keep=False)
    if dup.any():
        pairs = df.loc[dup, ["state_code", "year"]].drop_duplicates()
        log.append(f"{len(pairs)} state-year(s) had several rows and were combined: "
                   + ", ".join(f"{a}-{b}" for a, b in pairs.itertuples(index=False)))
    aggs = {v: ("sum" if a == "sum" else "mean") for v, a in zip(g["variable"], g["agg"]) if v in vars_}
    out = (df.groupby(["state_code", "year"])
             .agg({**aggs})
             .reset_index())
    if aggs:
        # sum() turns all-missing groups into 0 - restore missing
        cnt = df.groupby(["state_code", "year"])[vars_].count().reset_index(drop=True)
        for v in vars_:
            out.loc[cnt[v].values == 0, v] = pd.NA

    out.to_csv(INTERIM / f"{source}.csv", index=False)
    if status_rows:
        rank = {"ok": 0, "not_applicable": 1, "not_reported": 2, "out_of_range": 3, "invalid": 4, "blank": 5}
        sr = pd.concat(status_rows)
        sr["r"] = sr.status.map(rank)
        sr = (sr.sort_values("r").drop_duplicates(["state_code", "year", "variable"])
                .drop(columns="r").sort_values(["variable", "state_code", "year"]))
        sr.to_csv(INTERIM / f"_status_{source}.csv", index=False)
    filled = int(out[vars_].notna().any(axis=1).sum()) if vars_ else 0
    VALDIR.mkdir(parents=True, exist_ok=True)
    (VALDIR / f"{source}.txt").write_text("\n\n".join(log) or "No issues found.\n")
    status.update(status="loaded", rows_with_data=filled, issues=len(log))
    return status


def load_all() -> pd.DataFrame:
    res = [load_one(s, g) for s, g in spec().groupby("source", sort=False)]
    res = pd.DataFrame(res)
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["templates", "load"])
    a = ap.parse_args()
    make_templates() if a.action == "templates" else load_all()
