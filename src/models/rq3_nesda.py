"""RQ3 (revised): availability of online public services (NeSDA) vs. state digital adoption.

    python -m src.models.rq3_nesda

Why revised: the original RQ3 outcome (citizen sentiment from Reddit posts) could not be
collected. Reddit closed new public Data API requests on 31 October 2026 and declined the
project's Reddit for Researchers application on 9 November 2026. The revised outcome is the
official NeSDA Way Forward measure of e-service availability published by DARPG for every
state/UT.

Input : data/raw/nesda/nesda_eservices.csv (36 units x 3 snapshots, transcribed from the
        NeSDA Way Forward reports for Jan-Feb 2024, Oct 2024 and Mar 2026; each snapshot's
        national totals were checked against the totals printed in the report)
        data/processed/analysis_panel.csv (adoption_index, ln_upi_txn_pc, internet_per100)
Output: reports/final/rq3_results.json, reports/figures/final_rq3_scatter.png,
        data/processed/nesda_state_year.csv

Snapshot -> panel year: Jan-Feb 2024 -> 2023, Oct 2024 -> 2024, Mar 2026 -> 2025.
Outcomes: mandatory_sat = % of the mandatory e-services (56, later 59) offered online
          (primary); ln_eservices = ln(total e-services offered online) (secondary).
Tests (alpha = .05, two-sided):
  * Spearman rho between outcome and adoption index, pooled over the three snapshots;
    p-value from a state-cluster permutation test (each state's 3-year adoption vector is
    re-assigned to another state, 9,999 draws), 95% CI from a state-cluster bootstrap.
  * Spearman rho within each snapshot (36 units each).
  * Two-way fixed-effects regression of the outcome on the adoption index (within-state).
Power: n for rho = .30 (Fisher z) x design effect 1 + (m - 1) * icc, m = 3 snapshots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import PROCESSED, RAW, REPORTS, to_state_code  # noqa: E402

SEED, B = 640, 9999
OUT = REPORTS / "final"
FIG = REPORTS / "figures"


def load() -> pd.DataFrame:
    n = pd.read_csv(RAW / "nesda" / "nesda_eservices.csv")
    n["state_code"] = n.state.map(to_state_code)
    assert n.state_code.notna().all(), n[n.state_code.isna()].state.tolist()
    n["mandatory_sat"] = 100 * n.mandatory_available / n.mandatory_total
    n["ln_eservices"] = np.log(n.total_eservices)
    p = pd.read_csv(PROCESSED / "analysis_panel.csv")[["state_code", "state_name", "year", "adoption_index", "ln_upi_txn_pc", "internet_per100"]]
    d = n.merge(p, on=["state_code", "year"], how="left")
    d[["state_code", "year", "total_eservices", "mandatory_available", "mandatory_total", "mandatory_sat", "ln_eservices", "report", "reference"]] \
        .to_csv(PROCESSED / "nesda_state_year.csv", index=False)
    return d.dropna(subset=["adoption_index"])


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)


def cluster_perm_p(d, y, x, rng):
    """Permute whole states' x-trajectories across states (keeps within-state dependence)."""
    obs = spearman(d[x], d[y])
    wide = d.pivot(index="state_code", columns="year", values=x)
    states = wide.index.to_numpy()
    keyed = d.set_index(["state_code", "year"])[y]
    count = 0
    for _ in range(B):
        perm = pd.DataFrame(wide.values[rng.permutation(len(states))], index=states, columns=wide.columns)
        xx = perm.stack().rename("x")
        j = pd.concat([xx, keyed], axis=1, join="inner").dropna()
        count += abs(spearman(j.x, j[y])) >= abs(obs)
    return obs, (count + 1) / (B + 1)


def cluster_boot_ci(d, y, x, rng, reps=4000):
    states = d.state_code.unique()
    g = {s: d[d.state_code == s] for s in states}
    vals = []
    for _ in range(reps):
        b = pd.concat([g[s] for s in rng.choice(states, len(states), replace=True)])
        vals.append(spearman(b[x], b[y]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def icc(d, y):
    """One-way intraclass correlation of the outcome across snapshots (state as group)."""
    g = d.groupby("state_code")[y]
    k = g.size().mean()
    msb = (g.mean().sub(d[y].mean()) ** 2 * g.size()).sum() / (g.ngroups - 1)
    msw = ((d[y] - g.transform("mean")) ** 2).sum() / (len(d) - g.ngroups)
    return float((msb - msw) / (msb + (k - 1) * msw))


def n_for_rho(r=0.30, alpha=0.05, power=0.80):
    return int(np.ceil(((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.arctanh(r)) ** 2 + 3))


def power_rho(n_eff, r=0.30, alpha=0.05):
    z = np.arctanh(r) * np.sqrt(n_eff - 3)
    c = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(c - z) + stats.norm.cdf(-c - z))


def main():
    rng = np.random.default_rng(SEED)
    d = load()
    R = {"n": int(len(d)), "units": int(d.state_code.nunique()), "years": sorted(int(y) for y in d.year.unique()),
         "outcome_means": d.groupby("year")[["mandatory_sat", "total_eservices"]].mean().round(2).to_dict(orient="index")}
    for y in ("mandatory_sat", "ln_eservices"):
        rho, p = cluster_perm_p(d, y, "adoption_index", rng)
        ci = cluster_boot_ci(d, y, "adoption_index", rng)
        by_year = {int(t): {"rho": spearman(g.adoption_index, g[y]), "p": float(stats.spearmanr(g.adoption_index, g[y]).pvalue), "n": int(len(g))}
                   for t, g in d.groupby("year")}
        comps = {c: spearman(d[c], d[y]) for c in ("ln_upi_txn_pc", "internet_per100")}
        fe = smf.ols(f"{y} ~ adoption_index + C(state_code) + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.state_code})
        R[y] = {"rho": rho, "p_perm": p, "ci95": ci, "by_year": by_year, "components_rho": comps,
                "fe": {"b": float(fe.params["adoption_index"]), "se": float(fe.bse["adoption_index"]),
                       "p": float(fe.pvalues["adoption_index"]), "lo": float(fe.conf_int().loc["adoption_index", 0]),
                       "hi": float(fe.conf_int().loc["adoption_index", 1]), "within_sd_x": float((d.adoption_index - d.groupby("state_code").adoption_index.transform("mean")).std())},
                "icc": icc(d, y)}
        print(y, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[y].items() if k in ("rho", "p_perm", "ci95")}, flush=True)
    n0 = n_for_rho()
    ic = R["mandatory_sat"]["icc"]
    deff_plan, deff_hat = 1 + 2 * 0.10, 1 + 2 * max(ic, 0)
    R["power"] = {"n_srs": n0, "required_rho10": int(np.ceil(n0 * deff_plan)), "icc_hat": ic,
                  "deff_hat": deff_hat, "n_eff_hat": R["n"] / deff_hat,
                  "power_rho10": power_rho(R["n"] / deff_plan), "power_hat": power_rho(R["n"] / deff_hat),
                  "mde_rho_hat": float(np.tanh((stats.norm.ppf(.975) + stats.norm.ppf(.80)) / np.sqrt(R["n"] / deff_hat - 3)))}

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1))
    cols = {2023: "#8a96a3", 2024: "#2f5d8a", 2025: "#b07a2a"}
    for t, g in d.groupby("year"):
        ax[0].scatter(g.adoption_index, g.mandatory_sat, s=14, color=cols[t], label=str(t), alpha=.85)
        ax[1].scatter(g.adoption_index, g.total_eservices, s=14, color=cols[t], alpha=.85)
    ax[0].set_ylabel("Mandatory e-services online (%)"); ax[1].set_ylabel("Total e-services online (log scale)")
    ax[1].set_yscale("log")
    for a, y, lab in ((ax[0], "mandatory_sat", "a"), (ax[1], "ln_eservices", "b")):
        a.set_xlabel("Digital adoption index (z)")
        a.set_title(f"({lab}) Spearman ρ = {R[y]['rho']:.2f}, p = {R[y]['p_perm']:.3f}".replace("0.", ".").replace("-", "−"), loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7.5, title="Year", title_fontsize=7.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "final_rq3_scatter.png", dpi=220, bbox_inches="tight", pad_inches=.05)
    OUT.mkdir(exist_ok=True)
    (OUT / "rq3_results.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R["power"], indent=1))


if __name__ == "__main__":
    main()
