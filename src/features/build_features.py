"""Derived variables, analysis-sample flags and the data dictionary.

Input : data/processed/state_year_panel.csv
        data/processed/sentiment_state_year.csv   (optional, from src/features/sentiment.py)
Output: data/processed/analysis_panel.csv
        data/processed/data_dictionary.csv

Construction rules (also stated in the synopsis, section "Variable construction"):
  * Per-capita measures divide by projected mid-year population.
  * Skewed adoption measures enter as natural logs: ln(1 + x).
  * z-scores are computed over the pooled 2018-2025 panel (mean 0, SD 1), so an
    index value of +1 means one SD above the average state-year in the panel.
  * adoption_index = mean z of ln UPI per capita, ln e-Taal per capita,
    ln DBT per capita and internet per 100; needs >= 3 of the 4 parts.
  * sdei (secondary outcome) = mean z of timely-wage % and ePoS-authentication %;
    needs both parts. PC1 weights are printed as a sensitivity check.
  * Nothing is imputed here. Structural non-coverage (NS) is flagged and excluded;
    other gaps stay missing (handled in the models by listwise deletion, with
    multiple imputation as a sensitivity check when a variable is >10% missing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import CONFIG, PROCESSED  # noqa: E402

ADOPTION_PARTS = ["ln_upi_txn_pc", "ln_etaal_txn_pc", "ln_dbt_pc", "internet_per100"]
SDEI_PARTS = ["nrega_timely_pct", "pds_epos_auth_pct"]
SENTIMENT_COLS = ["n_posts", "share_positive", "share_negative", "net_sentiment", "sentiment_sufficient"]
MIN_POSTS = 30

SAMPLES = {  # analysis sample: complete cases of the variables each model needs
    "sample_rq1": ["nrega_timely_pct", "ln_upi_txn_pc", "ln_etaal_txn_pc", "ln_dbt_pc",
                   "internet_per100", "ln_nsdp_pc"],
    "sample_rq2": ["nrega_timely_pct", "adoption_index", "ln_nsdp_pc", "literacy_pct"],
    "sample_rq3": ["net_sentiment", "adoption_index"],
    "sample_rq4": ["nrega_women_pd_pct", "adoption_index", "ln_nsdp_pc"],
}

DERIVED = [  # variable, description, unit, role
    ("upi_txn_pc", "PhonePe UPI transactions per person = upi_txn / population", "count/person", "Input to adoption"),
    ("ln_upi_txn_pc", "ln(1 + upi_txn_pc)", "log", "RQ1 predictor"),
    ("upi_users_pct", "PhonePe registered users as % of population (can exceed 100: multiple accounts)", "%", "Descriptive"),
    ("etaal_txn_pc", "Government e-transactions per person = etaal_txn / population", "count/person", "Input to adoption"),
    ("ln_etaal_txn_pc", "ln(1 + etaal_txn_pc)", "log", "RQ1 predictor"),
    ("dbt_pc", "DBT transferred per person = dbt_amount_cr x 10^7 / population", "Rs/person", "Input to adoption"),
    ("ln_dbt_pc", "ln(1 + dbt_pc)", "log", "RQ1 predictor"),
    ("ln_nsdp_pc", "ln(nsdp_pc_real)", "log Rs", "Control; RQ2 moderator"),
    ("adoption_index", "Mean z-score of ln_upi_txn_pc, ln_etaal_txn_pc, ln_dbt_pc, internet_per100 (>= 3 of 4 parts)", "z-score", "RQ2-RQ4 predictor"),
    ("sdei", "Service Delivery Efficiency Index: mean z of nrega_timely_pct and pds_epos_auth_pct (both required)", "z-score", "RQ1 secondary outcome"),
    ("upi_txn_growth", "Year-on-year growth in upi_txn", "%", "Descriptive"),
    ("n_posts", "Classified English-language posts geolocated to the state in the year", "count", "RQ3 (weight / filter)"),
    ("share_positive", "Share of posts labelled positive", "proportion 0-1", "RQ3 descriptive"),
    ("share_negative", "Share of posts labelled negative", "proportion 0-1", "RQ3 descriptive"),
    ("net_sentiment", "share_positive - share_negative", "-1 to 1", "RQ3 outcome"),
    ("sentiment_sufficient", f"1 if n_posts >= {MIN_POSTS} (state-year enters RQ3)", "0/1", "RQ3 filter"),
    ("sample_rq1", "1 if all RQ1 model variables are present and MGNREGS operates", "0/1", "Sample flag"),
    ("sample_rq2", "1 if all RQ2 model variables are present", "0/1", "Sample flag"),
    ("sample_rq3", "1 if sentiment_sufficient and adoption_index present", "0/1", "Sample flag"),
    ("sample_rq4", "1 if all RQ4 model variables are present", "0/1", "Sample flag"),
]


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def _mean_z(df: pd.DataFrame, cols: list[str], min_parts: int) -> pd.Series:
    z = pd.concat([_z(df[c].astype(float)) if df[c].notna().sum() > 2
                   else pd.Series(np.nan, index=df.index) for c in cols], axis=1)
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= min_parts)


def _pc1_weights(df: pd.DataFrame, cols: list[str]) -> str:
    X = df[cols].dropna()
    if len(X) < 10:
        return "SDEI PC1 sensitivity: fewer than 10 complete state-years"
    Z = X.apply(_z)
    vals, vecs = np.linalg.eigh(np.cov(Z.values, rowvar=False))
    w = vecs[:, -1] * (1 if vecs[:, -1].sum() >= 0 else -1)
    return f"SDEI PC1 sensitivity: weights {np.round(w, 2).tolist()}, explains {vals[-1] / vals.sum():.0%} of variance (n={len(X)})"


def build(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = panel.copy().sort_values(["state_code", "year"])
    notes = []
    pop = df["population"].astype(float)
    if pop.notna().sum() == 0:
        notes.append("Per-capita measures not computed yet: population template not filled")
    df["upi_txn_pc"] = df["upi_txn"] / pop
    df["upi_users_pct"] = df["upi_users"] / pop * 100
    df["etaal_txn_pc"] = df["etaal_txn"].astype(float) / pop
    df["dbt_pc"] = df["dbt_amount_cr"].astype(float) * 1e7 / pop
    for c in ("upi_txn_pc", "etaal_txn_pc", "dbt_pc"):
        df["ln_" + c] = np.log1p(df[c])
    df["ln_nsdp_pc"] = np.log(df["nsdp_pc_real"].astype(float))
    df["adoption_index"] = _mean_z(df, ADOPTION_PARTS, 3)
    df["sdei"] = _mean_z(df, SDEI_PARTS, 2)
    df["upi_txn_growth"] = df.groupby("state_code")["upi_txn"].pct_change(fill_method=None) * 100

    sent = PROCESSED / "sentiment_state_year.csv"
    if sent.exists():
        df = df.merge(pd.read_csv(sent), on=["state_code", "year"], how="left")
    for c in SENTIMENT_COLS:
        if c not in df:
            df[c] = np.nan
    df["sentiment_sufficient"] = (df["n_posts"] >= MIN_POSTS).astype(int)
    df.loc[df.sentiment_sufficient == 0, "net_sentiment"] = np.nan

    for flag, cols in SAMPLES.items():
        df[flag] = df[cols].notna().all(axis=1).astype(int)
    df.loc[df.flag_no_mgnregs == 1, ["sample_rq1", "sample_rq2", "sample_rq4"]] = 0

    notes.append(_pc1_weights(df, SDEI_PARTS))
    for flag in SAMPLES:
        notes.append(f"{flag}: {int(df[flag].sum())} state-years")
    return df, notes


def dictionary(df: pd.DataFrame) -> pd.DataFrame:
    spec = pd.read_csv(CONFIG / "manual_sources.csv")
    rows = {
        "state_code": ("Project 2-letter state/UT code (config/states.csv)", "code", "config", "ID"),
        "state_name": ("State/UT name", "text", "config", "ID"),
        "type": ("State or Union Territory", "text", "config", "ID"),
        "region": ("Geographic region", "text", "config", "Descriptive"),
        "year": ("Calendar year; financial year t-(t+1) recorded as t", "year", "-", "ID"),
        "upi_txn": ("PhonePe UPI transactions in the year (sum of 4 quarters)", "count", "PhonePe Pulse", "Input"),
        "upi_txn_p2p": ("PhonePe person-to-person transactions", "count", "PhonePe Pulse", "Descriptive"),
        "upi_txn_retail": ("PhonePe merchant/retail transactions", "count", "PhonePe Pulse", "Descriptive"),
        "upi_txn_utility": ("PhonePe recharge and bill-payment transactions", "count", "PhonePe Pulse", "Descriptive"),
        "upi_users": ("PhonePe registered users at 31 Dec (cumulative, Q4 value)", "count", "PhonePe Pulse", "Descriptive"),
        "upi_merchants": ("PhonePe registered merchants at 31 Dec (cumulative, Q4 value)", "count", "PhonePe Pulse", "Descriptive"),
        "flag_no_mgnregs": ("1 = MGNREGS does not operate (NS code); excluded from RQ1, RQ2, RQ4", "0/1", "derived", "Flag"),
        "flag_no_pds_epos": ("1 = no ePoS grain distribution (NS code); SDEI missing", "0/1", "derived", "Flag"),
        "flag_jk_incl_ladakh": ("1 = J&K value includes Ladakh (before 2019)", "0/1", "derived", "Flag"),
        "flag_ladakh_not_separate": ("1 = Ladakh not yet a separate UT (before 2019)", "0/1", "derived", "Flag"),
        "flag_dh_premerger": ("1 = DNH and Daman & Diu combined before the 2020 merger", "0/1", "derived", "Flag"),
    }
    for _, s in spec.iterrows():
        rows[s.variable] = (s.label, s.unit, s.source, s.research_question)
    for v, lab, unit, role in DERIVED:
        rows[v] = (lab, unit, "derived", role)
    out = []
    for c in df.columns:
        lab, unit, src, role = rows.get(c, ("UNDOCUMENTED", "", "", ""))
        out.append({"variable": c, "description": lab, "unit": unit, "source": src, "role": role,
                    "dtype": str(df[c].dtype), "non_missing_%": round(df[c].notna().mean() * 100, 1)})
    d = pd.DataFrame(out)
    if (d.description == "UNDOCUMENTED").any():
        raise SystemExit(f"Undocumented columns: {d.loc[d.description == 'UNDOCUMENTED', 'variable'].tolist()}")
    return d


def main():
    panel = pd.read_csv(PROCESSED / "state_year_panel.csv")
    df, notes = build(panel)
    df.to_csv(PROCESSED / "analysis_panel.csv", index=False)
    dictionary(df).to_csv(PROCESSED / "data_dictionary.csv", index=False)
    print("\n".join("  - " + n for n in notes))


if __name__ == "__main__":
    main()
