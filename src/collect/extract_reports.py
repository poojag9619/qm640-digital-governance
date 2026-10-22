"""Extract state-year values from the official source files stored in data/raw/.

    python -m src.collect.extract_reports

Writes the filled entry sheets that src/collect/manual_sources.py validates:
  data/raw/population/population.csv   <- NCP (2020) projections, Table 11 (1 July)
  data/raw/rbi/rbi.csv                 <- RBI HSIS Table 10 (NSDP per capita) + Table 6 (literacy)
  data/raw/trai/trai.csv               <- TRAI Oct-Dec reports, service-area internet per 100
  data/raw/nrega/nrega.csv             <- MGNREGA "Scheme At a Glance" state dashboards
                                          (record in nrega_at_a_glance_scrape_2026-10-20.txt)

Every rule that changes or maps a value is applied here and printed, so the
extraction can be audited against the PDFs/Excel files in the same folders.
Requires `pdftotext` (poppler-utils).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import CONFIG, RAW, YEARS, states, to_state_code  # noqa: E402

POP_PDF = RAW / "population" / "NCP_2020_population_projections_2011_2036.pdf"
NSDP_XLSX = RAW / "rbi" / "RBI_HSIS_T10_per_capita_NSDP_constant.xlsx"
LIT_XLSX = RAW / "rbi" / "RBI_HSIS_T6_literacy.xlsx"
TRAI_PDF = RAW / "trai" / "TRAI_PIR_Oct-Dec_{y}.pdf"


def _pdftext(pdf: Path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                          text=True, check=True).stdout


def _template(source: str) -> pd.DataFrame:
    return pd.read_csv(Path(__file__).resolve().parents[2] / "templates" / f"{source}.csv")


# --------------------------------------------------------------------------- population
# Printed page number -> units on that page (left to right), Table 11 "as on 1st July"
POP_PAGES = {83: ["INDIA", "JK", "HP"], 84: ["PB", "HR", "DL"], 85: ["RJ", "UP", "BR"],
             86: ["AS", "WB", "JH"], 87: ["OD", "CG", "MP"], 88: ["GJ", "MH", "AP"],
             89: ["KA", "KL", "TN"], 90: ["CH", "UK", "SK"], 91: ["AR", "NL", "MN"],
             92: ["MZ", "TR", "ML"], 93: ["DD", "DNH", "GA"], 94: ["LD", "PY", "AN"],
             95: ["TS", "LA"]}


def population() -> pd.DataFrame:
    rows, seen = [], set()
    for page in _pdftext(POP_PDF).split("\f"):
        if "as on 1st July" not in page or "Projected Total Population" not in page:
            continue
        lines = [l for l in page.strip().splitlines() if l.strip()]
        m = re.match(r"^\s*(\d{2,3})\s*$", lines[-1])
        if not m or int(m[1]) not in POP_PAGES or int(m[1]) in seen:
            continue
        pg = int(m[1]); seen.add(pg); units = POP_PAGES[pg]; k = len(units)
        toks = [int(t.replace(",", "")) for t in re.findall(r"\d[\d,]*", "\n".join(lines[:-1]))]
        i = 0
        while not (toks[i] == 2011 and toks[i + 1 + 3 * k] == 2012):
            i += 1
        years = []
        while i < len(toks) and 2011 <= toks[i] <= 2036:
            vals = toks[i + 1:i + 1 + 3 * k]
            for j, u in enumerate(units):
                p, mle, fem = vals[3 * j:3 * j + 3]
                assert abs(p - mle - fem) <= 2, (pg, toks[i], u)   # persons = male + female
                rows.append((u, toks[i], p * 1000))                # report is in thousands
            years.append(toks[i]); i += 1 + 3 * k
        assert years == list(range(2011, 2037)), pg
    assert seen == set(POP_PAGES), sorted(set(POP_PAGES) - seen)
    df = pd.DataFrame(rows, columns=["unit", "year", "population"])
    w = df.pivot(index="unit", columns="year", values="population")
    check = (w.drop("INDIA").sum() / w.loc["INDIA"]).round(4)
    assert (check == 1).all(), "states do not add up to India"
    print(f"population: {df.unit.nunique() - 1} units x 26 years; states sum to India in every year")
    name = {"DD": "Daman and Diu", "DNH": "Dadra and Nagar Haveli"}
    code2name = dict(zip(states().state_code, states().state_name))
    df = df[(df.unit != "INDIA") & df.year.isin(YEARS)].copy()
    df["state_name"] = df.unit.map(lambda u: name.get(u, code2name.get(u)))
    df["source_document"] = POP_PDF.name + ", Table 11 (1 July)"
    df["notes"] = ""
    return df[["state_name", "year", "population", "source_document", "notes"]]


# --------------------------------------------------------------------------- RBI
def rbi() -> pd.DataFrame:
    recs = []
    for sh, d in pd.read_excel(NSDP_XLSX, header=None, sheet_name=None).items():
        names = list(d.iloc[4, 2:])
        for _, r in d.iterrows():
            m = re.match(r"^(\d{4})-\d{2}$", str(r.iloc[1]).strip())
            if not m or int(m[1]) not in YEARS:
                continue
            for name, v in zip(names, r.iloc[2:]):
                if "All-India" in str(name):
                    continue
                recs.append({"state_name": name, "year": int(m[1]),
                             "nsdp_pc_real": "NR" if str(v).strip() in (".", "", "nan") else v})
    nsdp = pd.DataFrame(recs)
    # keep the 2011-12 base series only (the last occurrence of each year)
    nsdp = nsdp.drop_duplicates(["state_name", "year"], keep="last")
    lit = pd.read_excel(LIT_XLSX, header=None)
    lit = lit.iloc[4:, [1, 8]].dropna()
    lit.columns = ["state_name", "literacy_pct"]
    lit = lit[~lit.state_name.astype(str).str.contains("ALL INDIA|Not Available|Notes|Source")]
    lit_map = {to_state_code(n): v for n, v in zip(lit.state_name, lit.literacy_pct)}
    # Census 2011 predates Telangana and Ladakh: use the undivided-state value (noted)
    lit_map.setdefault("TS", lit_map["AP"]); lit_map.setdefault("LA", lit_map["JK"])
    # DNH and Daman & Diu listed separately; both map to DH -> keep both rows (loader averages)
    t = _template("rbi")[["state_name", "year"]]
    t["code"] = t.state_name.map(to_state_code)
    nsdp["code"] = nsdp.state_name.map(to_state_code)
    t = t.merge(nsdp[["code", "year", "nsdp_pc_real"]], on=["code", "year"], how="left")
    t["nsdp_pc_real"] = t.nsdp_pc_real.fillna("NR")    # not published (e.g. Ladakh, DH, LD, 2024-25)
    lit_dh = [v for n, v in zip(lit.state_name, lit.literacy_pct) if to_state_code(n) == "DH"]
    t["literacy_pct"] = t.code.map(lit_map)
    t.loc[t.code == "DH", "literacy_pct"] = sum(lit_dh) / len(lit_dh)
    t["source_document"] = f"{NSDP_XLSX.name}; {LIT_XLSX.name}"
    t["notes"] = ""
    t.loc[t.code.isin(["TS", "LA"]), "notes"] = "literacy: Census 2011 value of undivided AP / J&K"
    t.loc[t.code == "DH", "notes"] = "literacy: mean of DNH and Daman & Diu (Census 2011)"
    t.loc[(t.code == "JK") & (t.year == 2018), "notes"] += " NSDP 2018-19 includes Ladakh"
    n_nr = (t.nsdp_pc_real == "NR").sum()
    print(f"rbi: NSDP for {nsdp.code.nunique()} units, {n_nr} state-years NR; literacy for {t.literacy_pct.notna().sum() // len(YEARS)} units")
    return t.drop(columns="code")


# --------------------------------------------------------------------------- TRAI
AREAS = ["Andhra Pradesh", "Assam", "Bihar", "Delhi", "Gujarat", "Haryana", "Himachal Pradesh",
         "Jammu & Kashmir", "Jammu and Kashmir", "Karnataka", "Kerala", "Madhya Pradesh",
         "Maharashtra", "Mumbai", "North East", "Odisha", "Orissa", "Punjab", "Rajasthan",
         "Tamil Nadu", "UP (East)", "UP (West)", "Kolkata", "West Bengal"]
PAIRS = [("Maharashtra", "Mumbai"), ("UP (East)", "UP (West)"), ("Kolkata", "West Bengal")]


def trai_area_table(year: int) -> dict[str, float]:
    t = _pdftext(Path(str(TRAI_PDF).format(y=year))).split("\n")
    start = [i for i, l in enumerate(t)
             if re.search(r"Service Area[- ]wise number of Internet Subscribers", l, re.I)
             and "...." not in l and not re.search(r"population\s+\d+\s*$", l)][0]
    vals, combos = {}, []
    for l in t[start:start + 90]:
        s = l.strip()
        nums = [float(x) for x in re.findall(r"\d+\.\d+", s)]
        name = next((a for a in AREAS if s.startswith(a)), "Total" if s.startswith("Total") else None)
        if name and len(nums) >= 6:
            vals[name] = nums[5]                      # last column: total per 100 population
        elif name and len(nums) in (4, 5):
            vals[name] = nums[-1]                     # rural/urban shown as '-' (e.g. Delhi)
        elif not name and len(nums) == 3:
            combos.append(nums[2])                    # value printed between a combined pair
        if name == "Total":
            break
    for a, b in PAIRS:
        v = vals.get(a, vals.get(b))
        if v is None:
            v = combos.pop(0)
        vals[a] = vals[b] = v
    vals = {{"Orissa": "Odisha", "Jammu and Kashmir": "Jammu & Kashmir"}.get(k, k): v for k, v in vals.items()}
    return vals


def trai() -> pd.DataFrame:
    amap = pd.read_csv(CONFIG / "trai_service_areas.csv")
    area_of = {}
    for _, r in amap.iterrows():
        for c in r.state_codes.split("|"):
            area_of.setdefault(c, r.service_area)
    rows = []
    for y in YEARS:
        v = trai_area_table(y)
        print(f"trai {y}: national total {v['Total']} per 100")
        for c, area in area_of.items():
            key = {"UP (East)": "UP (East)"}.get(area, area)
            rows.append({"code": c, "year": y, "internet_per100": v[key],
                         "source_document": Path(str(TRAI_PDF).format(y=y)).name,
                         "notes": f"service area value: {area}"})
    df = pd.DataFrame(rows)
    code2name = dict(zip(states().state_code, states().state_name))
    df["state_name"] = df.code.map(code2name)
    return df[["state_name", "year", "internet_per100", "source_document", "notes"]]


# --------------------------------------------------------------------------- MGNREGA
NREGA_TXT = RAW / "nrega" / "nrega_at_a_glance_scrape_2026-10-20.txt"
NREGA_FY = [2026, 2025, 2024, 2023, 2022, 2021]     # column order on the portal (FY t-(t+1) -> t)
MIN_VOLUME_LAKH = 0.1                               # < 10,000 person-days: too small to be reliable
SUSPENDED_SHARE = 0.01                              # < 1% of the state's peak volume: scheme suspended


def r149_timely() -> pd.DataFrame:
    """MIS report R14.9 'Timely Payment Status (in %)', Stage 1, state-wise; one HTML-as-XLS per FY."""
    import io
    recs = []
    for f in sorted((RAW / "nrega").glob("R14-9_Timely_payment_stage_*.xls")):
        fy = int(re.search(r"_(\d{4})_\d{4}\.xls$", f.name)[1])
        tab = pd.read_html(io.StringIO(f.read_text(encoding="utf-8", errors="replace")))[0]
        tab.columns = ["sno", "state", "stage1"] + list(tab.columns[3:])
        tab = tab[pd.to_numeric(tab.sno, errors="coerce").notna() & (tab.state.astype(str) != "2")]
        for _, r in tab.iterrows():
            name = str(r.state).replace("DN HAVELI AND DD", "dnhdd").replace("ANDAMAN AND NICOBAR", "andaman and nicobar islands")
            recs.append({"state_code": to_state_code(name), "year": fy, "r149": float(r.stage1)})
    df = pd.DataFrame(recs)
    assert df.state_code.notna().all()
    print(f"R14.9: {df.year.nunique()} financial years x {df.state_code.nunique()} units")
    return df


def nrega() -> pd.DataFrame:
    recs = []
    for line in NREGA_TXT.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, *parts = line.split(";")
        vals = {k: [float(x) for x in v.split(",")] for k, v in (p.split(":") for p in parts)}
        for i, fy in enumerate(NREGA_FY):
            recs.append({"name": name, "year": fy, **{k: vals[k][i] for k in vals}})
    df = pd.DataFrame(recs)
    df = df[df.year.isin(YEARS)]
    df["code"] = df.name.map(lambda n: to_state_code(n.replace("DN HAVELI AND DD", "dnhdd")
                                                       .replace("ANDAMAN AND NICOBAR", "andaman and nicobar islands")))
    assert df.code.notna().all(), df[df.code.isna()].name.unique()
    peak = df.groupby("code").P.transform("max")
    df["status"] = "ok"
    df.loc[df.P == 0, "status"] = "NS"                                  # no work generated
    df.loc[(df.P > 0) & (df.P < SUSPENDED_SHARE * peak), "status"] = "NS"  # suspended (West Bengal 2023-)
    df.loc[(df.status == "ok") & (df.P < MIN_VOLUME_LAKH), "status"] = "LOWVOL"
    t = states()[["state_code", "state_name"]].merge(pd.DataFrame({"year": YEARS}), how="cross")
    t = t.merge(df.rename(columns={"code": "state_code"}), on=["state_code", "year"], how="left")
    out = pd.DataFrame({"state_name": t.state_name, "year": t.year})
    note = pd.Series("", index=t.index)
    for col, var in (("T", "nrega_timely_dash_pct"), ("W", "nrega_women_pd_pct"),
                     ("P", "nrega_persondays_lakh"), ("E", "nrega_efms_pct")):
        v = t[col].astype(object)
        v[t.status == "NS"] = "NS"
        v[t.status == "LOWVOL"] = "NR" if col in ("T", "W", "E") else t.loc[t.status == "LOWVOL", col]
        out[var] = v
    no_scheme = t.state_code.isin(["CH", "DL"])            # fully urban UTs: no rural MGNREGS
    portal_gap = t.state_code.isin(["KL", "MH"])           # hidden on the portal during upgrade
    before = t.year < 2021                                  # older years: CAPTCHA-protected reports
    for var in ("nrega_timely_dash_pct", "nrega_women_pd_pct", "nrega_persondays_lakh", "nrega_efms_pct"):
        out.loc[no_scheme, var] = "NS"
        out.loc[portal_gap & ~no_scheme, var] = "NR"
        out.loc[before & ~no_scheme, var] = "NR"
    note[no_scheme] = "MGNREGS not implemented (urban UT)"
    note[portal_gap] = "not shown on portal (technical upgradation notice)"
    note[before & ~no_scheme & ~portal_gap] = "FY before 2021-22 not on At a Glance dashboard"
    note[t.status == "NS"] = "no/suspended activity (person-days 0 or <1% of peak)"
    note[t.status == "LOWVOL"] = "low volume (<10,000 person-days): rates set to NR"
    # primary outcome: MIS report R14.9 (Stage 1), one consistent series for FY 2018-19 to 2025-26
    r149 = r149_timely()
    t2 = t.merge(r149, on=["state_code", "year"], how="left")
    v = t2["r149"].astype(object)
    v[t2.r149 == 0] = "NS"                                   # no payments processed in the year
    v[t2.status == "NS"] = "NS"                              # suspended/no activity (dashboard volumes)
    v[t2.state_code == "WB"] = v[t2.state_code == "WB"].where(t2.year[t2.state_code == "WB"] < 2023, "NS")
    v[t2.state_code == "LD"] = "NR"                          # Lakshadweep: low volume in every year observed
    v[no_scheme] = "NS"
    v[portal_gap & ~no_scheme] = "NR"
    v[v.isna()] = "NR"
    out["nrega_timely_pct"] = v.values
    out["source_document"] = NREGA_TXT.name + "; R14-9_Timely_payment_stage_<FY>.xls"
    out["notes"] = note
    print(f"nrega: {(t.status == 'ok').sum()} usable state-years, "
          f"{(t.status == 'NS').sum()} suspended/no activity, {(t.status == 'LOWVOL').sum()} low volume")
    return out


def main():
    out = {"population": population(), "rbi": rbi(), "trai": trai(), "nrega": nrega()}
    for src, df in out.items():
        path = RAW / src / f"{src}.csv"
        df.to_csv(path, index=False)
        print(f"  -> {path.relative_to(RAW.parents[1])} ({len(df)} rows)")


if __name__ == "__main__":
    main()
