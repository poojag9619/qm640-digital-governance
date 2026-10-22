"""Core models for RQ1-RQ4, as specified in the synopsis.

    python -m src.models.rq_models          # writes reports/model_results.md

Inference: OLS with state and year fixed effects, standard errors clustered by state
(Cameron & Miller, 2015). Hypotheses are tested with cluster-robust Wald tests.
Out-of-sample check (RQ1): rolling-origin validation - fit on years <= t, predict t+1,
for t = 2021..2024, so no future information leaks into training. The benchmark is the
naive forecast "same value as last year". Each model runs only if its analysis sample
meets the minimum from the power analysis; otherwise the report says what is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import PROCESSED, REPORTS  # noqa: E402

MIN_N = {"RQ1": 216, "RQ2": 213, "RQ3": 102, "RQ4": 206}
ADOPT = ["ln_upi_txn_pc", "ln_etaal_txn_pc", "ln_dbt_pc", "internet_per100"]
FE = " + C(state_code) + C(year)"


def _fit(formula: str, d: pd.DataFrame):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["state_code"]})


def _wald(res, terms: list[str]) -> str:
    R = np.zeros((len(terms), len(res.params)))
    for i, t in enumerate(terms):
        R[i, list(res.params.index).index(t)] = 1
    w = res.wald_test(R, use_f=True, scalar=True)
    return f"F({len(terms)}, {int(w.df_denom)}) = {w.statistic:.2f}, p = {w.pvalue:.4f}"


def _coef_table(res, terms: list[str]) -> list[str]:
    ci = res.conf_int()
    out = ["| Term | b | SE | 95% CI | p |", "|---|---|---|---|---|"]
    for t in terms:
        out.append(f"| {t} | {res.params[t]:.3f} | {res.bse[t]:.3f} | "
                   f"[{ci.loc[t, 0]:.3f}, {ci.loc[t, 1]:.3f}] | {res.pvalues[t]:.4f} |")
    return out


def _residual_icc(res, d: pd.DataFrame) -> float:
    e = pd.Series(res.resid, index=d.index).groupby(d.state_code)
    lagged = e.shift(1)
    ok = lagged.notna()
    return float(np.corrcoef(pd.Series(res.resid, index=d.index)[ok], lagged[ok])[0, 1])


def _need(lab: str, d: pd.DataFrame) -> str | None:
    return None if len(d) >= MIN_N[lab] else f"_Not run: {len(d)} complete state-years, minimum {MIN_N[lab]}._"


def rq1(df):
    d = df[df.sample_rq1 == 1].copy()
    out = ["## RQ1 - adoption and timely wage payment", ""]
    if (msg := _need("RQ1", d)):
        return out + [msg]
    res = _fit("nrega_timely_pct ~ " + " + ".join(ADOPT) + " + ln_nsdp_pc" + FE, d)
    out += [f"N = {int(res.nobs)}, states = {d.state_code.nunique()}; joint test of 4 adoption measures: {_wald(res, ADOPT)}",
            f"Lag-1 residual autocorrelation within states (input to the design effect): {_residual_icc(res, d):.3f}", ""]
    out += _coef_table(res, ADOPT + ["ln_nsdp_pc"])
    # rolling-origin validation
    rows = []
    for t in range(2021, 2025):
        tr, te = d[d.year <= t], d[d.year == t + 1]
        te = te[te.state_code.isin(tr.state_code)]
        if len(te) == 0:
            continue
        m = smf.ols("nrega_timely_pct ~ " + " + ".join(ADOPT) + " + ln_nsdp_pc + C(state_code) + year", data=tr).fit()
        pred = m.predict(te)
        last = tr[tr.year == t].set_index("state_code").nrega_timely_pct.reindex(te.state_code).values
        rows.append((t + 1, np.sqrt(np.mean((te.nrega_timely_pct - pred) ** 2)),
                     np.nanmean(np.abs(te.nrega_timely_pct - pred)),
                     np.sqrt(np.nanmean((te.nrega_timely_pct.values - last) ** 2))))
    if rows:
        out += ["", "Rolling-origin validation (train <= t, test t+1; linear year trend replaces year dummies):", "",
                "| Test year | RMSE model | MAE model | RMSE naive (last year) |", "|---|---|---|---|"]
        out += [f"| {y} | {a:.2f} | {b:.2f} | {c:.2f} |" for y, a, b, c in rows]
    return out


def rq2(df):
    d = df[df.sample_rq2 == 1].copy()
    out = ["", "## RQ2 - moderation by literacy and income", ""]
    if (msg := _need("RQ2", d)):
        return out + [msg]
    d["lit_c"] = d.literacy_pct - d.literacy_pct.mean()
    d["nsdp_c"] = d.ln_nsdp_pc - d.ln_nsdp_pc.mean()
    res = _fit("nrega_timely_pct ~ adoption_index + nsdp_c + adoption_index:lit_c + adoption_index:nsdp_c" + FE, d)
    terms = ["adoption_index:lit_c", "adoption_index:nsdp_c"]
    out += [f"N = {int(res.nobs)}; joint test of the two interactions: {_wald(res, terms)}", ""]
    return out + _coef_table(res, ["adoption_index"] + terms)


def rq3(df):
    d = df[df.sample_rq3 == 1]
    out = ["", "## RQ3 - online sentiment and digital adoption", ""]
    if len(d) < 10:
        return out + [f"_Not run: {len(d)} state-years with >= 30 scored posts._"]
    rho, p = stats.spearmanr(d.net_sentiment, d.adoption_index)
    rng = np.random.default_rng(640)
    states = d.state_code.unique()
    boots = []
    for _ in range(2000):  # cluster bootstrap: resample states, keep their years together
        s = rng.choice(states, len(states), replace=True)
        b = pd.concat([d[d.state_code == k] for k in s])
        boots.append(stats.spearmanr(b.net_sentiment, b.adoption_index)[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    tag = "inferential" if len(d) >= MIN_N["RQ3"] else f"EXPLORATORY ONLY (n < {MIN_N['RQ3']})"
    return out + [f"n = {len(d)} state-years ({tag}); Spearman rho = {rho:.3f}, p = {p:.4f}; "
                  f"state-cluster bootstrap 95% CI [{lo:.3f}, {hi:.3f}]"]


def rq4(df):
    out = ["", "## RQ4 - adoption and inclusion", ""]
    d = df[df.sample_rq4 == 1].copy()
    if (msg := _need("RQ4", d)):
        return out + [msg]
    rows, pvals = [], []
    for y in ("nrega_women_pd_pct", "pmjdy_women_pct", "pmjdy_rural_pct"):
        dd = d.dropna(subset=[y])
        if len(dd) < 30:
            continue
        res = _fit(f"{y} ~ adoption_index + ln_nsdp_pc" + FE, dd)
        rows.append((y, len(dd), res.params["adoption_index"], res.bse["adoption_index"]))
        pvals.append(res.pvalues["adoption_index"])
    order = np.argsort(pvals)
    holm = np.empty(len(pvals))
    for rank, i in enumerate(order):  # Holm step-down adjustment
        holm[i] = min(1, max(holm[order[rank - 1]] if rank else 0, (len(pvals) - rank) * pvals[i]))
    out += ["| Outcome | N | b (adoption_index) | SE | p | Holm p |", "|---|---|---|---|---|---|"]
    out += [f"| {y} | {n} | {b:.3f} | {se:.3f} | {p:.4f} | {h:.4f} |"
            for (y, n, b, se), p, h in zip(rows, pvals, holm)]
    return out


def main():
    df = pd.read_csv(PROCESSED / "analysis_panel.csv")
    lines = ["# Model results", "", "Generated by `python -m src.models.rq_models`.", ""]
    for f in (rq1, rq2, rq3, rq4):
        lines += f(df)
    (REPORTS / "model_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
