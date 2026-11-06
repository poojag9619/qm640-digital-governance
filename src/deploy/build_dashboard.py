"""Deployment: score the latest year and publish a static monitoring dashboard.

    python -m src.deploy.build_dashboard            # scores the latest year in the database
    python -m src.deploy.build_dashboard --year 2025

What it does
  1. Reads the analysis sample from the SQLite database (src/db/build_db.py).
  2. Trains the selected models on all years before the scoring year:
       * risk model: random forest classifier for "lagging" (< 80% of wage payments on time),
         the best classifier by test AUC in src/models/final_analysis.py;
       * expectation model: the naive own-history benchmark (last year's value), the best
         regression model in the same evaluation.
  3. Scores the scoring year, writes the predictions to the database (table
     rolling_predictions) and writes reports/dashboard/index.html, a single self-contained
     page that can be hosted on GitHub Pages or any intranet web server.
"""
from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import REPORTS  # noqa: E402
from src.db.build_db import DB  # noqa: E402

SEED, THRESH = 640, 80.0
FEATURES = ["ln_upi_txn_pc", "internet_per100", "literacy_pct", "year", "state_hist", "lag1"]
OUT = REPORTS / "dashboard"


def load(con) -> pd.DataFrame:
    q = """SELECT s.state_code, st.state_name, s.year, s.nrega_timely_pct AS timely, s.ln_upi_txn_pc,
                  s.internet_per100, s.literacy_pct
           FROM state_year s JOIN states st USING (state_code) WHERE s.sample_rq1 = 1"""
    d = pd.read_sql(q, con).sort_values(["state_code", "year"])
    prev = d[["state_code", "year", "timely"]].rename(columns={"timely": "lag1"})
    prev["year"] += 1
    return d.merge(prev, on=["state_code", "year"], how="left")


def score(d: pd.DataFrame, year: int) -> pd.DataFrame:
    tr = d[d.year < year].copy()
    te = d[(d.year == year) & d.lag1.notna()].copy()
    hist = tr.groupby("state_code").timely.mean()
    tr["state_hist"], te["state_hist"] = tr.state_code.map(hist), te.state_code.map(hist)
    tr = tr[tr.lag1.notna()]
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    clf.fit(tr[FEATURES], (tr.timely < THRESH).astype(int))
    te["risk"] = clf.predict_proba(te[FEATURES])[:, 1]
    te["expected"] = te.lag1                                  # own-history benchmark
    te["gap"] = te.timely - te.expected
    te["flag"] = np.select([(te.risk >= .5) & (te.timely < THRESH), te.risk >= .5, te.gap <= -10],
                           ["Lagging (predicted and observed)", "High risk", "Below own history"], "On track")
    return te.sort_values("risk", ascending=False)


def sparkline(vals: list[float], w=120, h=28) -> str:
    v = [x for x in vals if x == x]
    if len(v) < 2:
        return ""
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - (x / 100) * h:.1f}" for i, x in enumerate(vals) if x == x)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<line x1="0" x2="{w}" y1="{h * .2:.1f}" y2="{h * .2:.1f}" class="ref"/>'
            f'<polyline points="{pts}" class="spark"/></svg>')


def page(s: pd.DataFrame, d: pd.DataFrame, year: int) -> str:
    n_risk = int((s.risk >= .5).sum())
    n_below = int((s.flag == "Below own history").sum())
    rows = []
    for _, r in s.iterrows():
        series = d[d.state_code == r.state_code].set_index("year").timely.reindex(range(d.year.min(), year + 1)).tolist()
        cls = {"Lagging (predicted and observed)": "bad", "High risk": "warn", "Below own history": "warn"}.get(r.flag, "ok")
        rows.append(f"<tr><td>{html.escape(r.state_name)}</td><td class='num'>{r.lag1:.1f}</td><td class='num'>{r.timely:.1f}</td>"
                    f"<td class='num'>{r.gap:+.1f}</td><td class='num'>{r.risk:.2f}</td><td><span class='pill {cls}'>{r.flag}</span></td>"
                    f"<td>{sparkline(series)}</td></tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MGNREGS Payment Timeliness Monitor</title>
<style>
:root {{ --bg:#ffffff; --fg:#1d2430; --muted:#5b6675; --line:#e3e7ec; --ok:#2e7d4f; --warn:#a86a00; --bad:#a33b3b; --accent:#2f5d8a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181d; --fg:#e8ecf1; --muted:#9aa5b1; --line:#2a3139; --ok:#5cbf86; --warn:#e0a53a; --bad:#e07070; --accent:#7fb0e0; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p.sub {{ color:var(--muted); margin:0 0 20px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:20px; }}
.kpi {{ border:1px solid var(--line); border-radius:8px; padding:12px 14px; }} .kpi b {{ display:block; font-size:24px; }}
.kpi span {{ color:var(--muted); font-size:13px; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--muted); font-weight:600; font-size:12.5px; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.pill {{ padding:2px 8px; border-radius:10px; font-size:12px; border:1px solid currentColor; }}
.ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .bad {{ color:var(--bad); }}
svg .spark {{ fill:none; stroke:var(--accent); stroke-width:1.6; }} svg .ref {{ stroke:var(--line); stroke-dasharray:3 2; }}
footer {{ color:var(--muted); font-size:12.5px; margin-top:18px; }}
</style></head><body><main>
<h1>MGNREGS wage-payment timeliness monitor, FY {year}-{str(year + 1)[-2:]}</h1>
<p class="sub">Share of wage payments generated within 15 days (MGNREGS MIS report R14.9), by state. Risk = random-forest probability that a state is lagging (&lt; 80% on time), trained on {d.year.min()}–{year - 1}.</p>
<div class="kpis">
<div class="kpi"><b>{len(s)}</b><span>states and UTs scored</span></div>
<div class="kpi"><b>{n_risk}</b><span>flagged high risk (probability ≥ .50)</span></div>
<div class="kpi"><b>{n_below}</b><span>other states ≥ 10 points below last year</span></div>
<div class="kpi"><b>{s.timely.median():.1f}%</b><span>median share paid on time</span></div>
</div>
<div class="wrap"><table>
<thead><tr><th>State / UT</th><th>Last year %</th><th>This year %</th><th>Change</th><th>Risk</th><th>Status</th><th>{d.year.min()}–{year}</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<footer>Built {date.today().isoformat()} by src/deploy/build_dashboard.py from data/processed/qm640.sqlite. The models describe associations, not causes; benchmark each state against its own history.</footer>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    a = ap.parse_args()
    con = sqlite3.connect(DB)
    d = load(con)
    year = a.year or int(d.year.max())
    s = score(d, year)
    s.assign(scored_on=date.today().isoformat())[["state_code", "year", "timely", "lag1", "expected", "gap", "risk", "flag", "scored_on"]] \
        .to_sql("rolling_predictions", con, if_exists="replace", index=False)
    con.commit(); con.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page(s, d, year), encoding="utf-8")
    summary = {"year": year, "states": int(len(s)), "high_risk": int((s.risk >= .5).sum()),
               "lagging_observed": int((s.timely < THRESH).sum()),
               "flag_counts": s.flag.value_counts().to_dict(),
               "top5": s.head(5)[["state_name", "timely", "risk"]].round(3).to_dict(orient="records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("->", OUT / "index.html")


if __name__ == "__main__":
    main()
