"""Merge every source onto the 36 x 2018-2025 skeleton.

Output: data/processed/state_year_panel.csv   (one row per state/UT and year)
        reports/missing_data_report.md        (coverage by variable and year)
        reports/figures/coverage_heatmap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import CONFIG, FIGURES, INTERIM, PROCESSED, REPORTS, YEARS, skeleton  # noqa: E402

ID = ["state_code", "state_name", "type", "region", "year"]


def _status() -> pd.DataFrame:
    files = sorted(INTERIM.glob("_status_*.csv"))
    if not files:
        return pd.DataFrame(columns=["state_code", "year", "variable", "status"])
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build() -> pd.DataFrame:
    panel = skeleton()
    # PhonePe (automated)
    pp = INTERIM / "phonepe_state_year.csv"
    if pp.exists():
        panel = panel.merge(pd.read_csv(pp).drop(columns="n_quarters"), on=["state_code", "year"], how="left")
    # manual sources, in the order of config/manual_sources.csv
    spec = pd.read_csv(CONFIG / "manual_sources.csv")
    for source, g in spec.groupby("source", sort=False):
        f = INTERIM / f"{source}.csv"
        if f.exists():
            panel = panel.merge(pd.read_csv(f), on=["state_code", "year"], how="left")
        for v in g.variable:            # keep every planned column, even if empty
            if v not in panel:
                panel[v] = pd.NA
    # structural non-coverage, from the NS codes in the filled templates
    st = _status()
    for var, flag in (("nrega_timely_pct", "flag_no_mgnregs"), ("pds_epos_auth_pct", "flag_no_pds_epos")):
        ns = st[(st.variable == var) & (st.status == "not_applicable")][["state_code", "year"]]
        panel[flag] = panel.set_index(["state_code", "year"]).index.isin(
            ns.set_index(["state_code", "year"]).index).astype(int)
    # boundary flags
    panel["flag_jk_incl_ladakh"] = ((panel.state_code == "JK") & (panel.year < 2019)).astype(int)
    panel["flag_ladakh_not_separate"] = ((panel.state_code == "LA") & (panel.year < 2019)).astype(int)
    panel["flag_dh_premerger"] = ((panel.state_code == "DH") & (panel.year < 2020)).astype(int)
    return panel.sort_values(["state_code", "year"]).reset_index(drop=True)


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    vals = [c for c in panel.columns if c not in ID and not c.startswith("flag_")]
    cov = panel.groupby("year")[vals].apply(lambda d: d.notna().mean() * 100).T.round(0)
    cov["overall_%"] = (panel[vals].notna().mean() * 100).round(1)
    return cov


def heatmap(cov: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = cov[YEARS]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(m) + 1.2), dpi=150)
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(YEARS)), YEARS, fontsize=8)
    ax.set_yticks(range(len(m)), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 60 else "#333")
    ax.set_title("Share of 36 states/UTs with data (%)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_heatmap.png")
    plt.close(fig)


def report(panel: pd.DataFrame, cov: pd.DataFrame) -> None:
    spec = pd.read_csv(CONFIG / "manual_sources.csv")
    src = dict(zip(spec.variable, spec.source))
    lines = ["# Missing-data report", "",
             f"Panel: {panel.state_code.nunique()} states/UTs x {len(YEARS)} years "
             f"= {len(panel)} rows. Values are % of the 36 units with data in that year.", "",
             "Variables at 0% have not been collected yet; see README > Data collection status.", "",
             "| Variable | Source | " + " | ".join(map(str, YEARS)) + " | Overall |",
             "|---|---|" + "---|" * (len(YEARS) + 1)]
    for v, r in cov.iterrows():
        s = "phonepe" if v.startswith("upi_") else src.get(v, "derived")
        lines.append(f"| {v} | {s} | " + " | ".join(f"{r[y]:.0f}" for y in YEARS) + f" | {r['overall_%']} |")
    st = _status()
    if len(st):
        counts = st.pivot_table(index="variable", columns="status", values="year", aggfunc="count", fill_value=0)
        lines += ["", "## Value status in filled templates", "",
                  "ok = numeric value; not_reported (NR) = not published for that year; "
                  "not_applicable (NS) = scheme absent, structurally excluded; blank = not collected yet.", "",
                  "| Variable | " + " | ".join(counts.columns) + " |",
                  "|---|" + "---|" * len(counts.columns)]
        lines += [f"| {v} | " + " | ".join(str(int(x)) for x in r) + " |" for v, r in counts.iterrows()]
    gaps = panel.loc[panel.upi_txn.isna(), ["state_code", "year"]] if "upi_txn" in panel else pd.DataFrame()
    lines += ["", "## Boundary changes handled by flags", "",
              "- `flag_jk_incl_ladakh`: J&K values before 2019 include Ladakh.",
              "- `flag_ladakh_not_separate`: Ladakh had no separate figures before 2019 in most sources.",
              "- `flag_dh_premerger`: Dadra & Nagar Haveli and Daman & Diu combined before the 2020 merger.",
              "", f"PhonePe state-years without a full year of data: {len(gaps)}", ""]
    (REPORTS / "missing_data_report.md").write_text("\n".join(lines))


def main():
    panel = build()
    panel.to_csv(PROCESSED / "state_year_panel.csv", index=False)
    cov = coverage(panel)
    heatmap(cov)
    report(panel, cov)
    print(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} columns -> {PROCESSED / 'state_year_panel.csv'}")


if __name__ == "__main__":
    main()
