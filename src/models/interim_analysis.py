"""Interim analysis: data-quality checks, EDA figures and preliminary models.

    python -m src.models.interim_analysis

Writes reports/interim/results.json (every number quoted in the interim report),
reports/interim/*.csv tables and reports/figures/interim_*.png.

Analysis window: 2018-2025. Primary outcome from MIS report R14.9 (FY 2018-19 to 2025-26);
women's person-days (RQ4) come from the dashboard and cover 2021-2025 only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import FIGURES, INTERIM, PROCESSED, REPORTS  # noqa: E402

OUT = REPORTS / "interim"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 640
R: dict = {}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
BLUE, GREY, OCHRE, RED = "#2f5d8a", "#8a96a3", "#b07a2a", "#a23b3b"
FE = " + C(state_code) + C(year)"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"interim_{name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def cl(formula, d):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["state_code"]})


def wald(res, terms):
    Rm = np.zeros((len(terms), len(res.params)))
    for i, t in enumerate(terms):
        Rm[i, list(res.params.index).index(t)] = 1
    w = res.wald_test(Rm, use_f=True, scalar=True)
    return {"F": float(w.statistic), "df1": len(terms), "df2": int(w.df_denom), "p": float(w.pvalue)}


def coefs(res, terms):
    ci = res.conf_int()
    return {t: {"b": float(res.params[t]), "se": float(res.bse[t]), "p": float(res.pvalues[t]),
                "lo": float(ci.loc[t, 0]), "hi": float(ci.loc[t, 1])} for t in terms}


def within_r2(res, d, y):
    """R2 of the model after removing state and year means (two-way within transformation)."""
    yy = d[y] - d.groupby("state_code")[y].transform("mean") - d.groupby("year")[y].transform("mean") + d[y].mean()
    resid = res.resid
    return float(1 - (resid ** 2).sum() / (yy ** 2).sum())


def power_regression(n, u, p, f2, alpha=0.05):
    df2 = n - p - 1
    crit = stats.f.ppf(1 - alpha, u, df2)
    return float(1 - stats.ncf.cdf(crit, u, df2, f2 * n))


def mde_f2(n, u, p, power=0.8):
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power_regression(n, u, p, mid) < power else (lo, mid)
    return float(hi)


# =========================================================================== data
panel = pd.read_csv(PROCESSED / "analysis_panel.csv")
status = pd.concat([pd.read_csv(f) for f in INTERIM.glob("_status_*.csv")], ignore_index=True)
win = panel[panel.year >= 2018].copy()
R["panel_rows"], R["panel_cols"] = int(panel.shape[0]), int(panel.shape[1])
R["window_rows"] = int(len(win))

# ---------------------------------------------------------------- data quality
dq = {}
key_vars = ["nrega_timely_pct", "nrega_women_pd_pct", "upi_txn", "internet_per100", "population",
            "nsdp_pc_real", "literacy_pct"]
for v in key_vars:
    s = win[v]
    st = status[(status.variable == v) & (status.year >= 2018)].status.value_counts().to_dict() if v in set(status.variable) else {}
    dq[v] = {"n": int(s.notna().sum()), "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean() * 100), 1),
             "status": {k: int(c) for k, c in st.items()}}
R["missingness"] = dq
R["duplicates_panel"] = int(panel.duplicated(["state_code", "year"]).sum())
R["duplicates_phonepe_quarter"] = int(pd.read_csv(INTERIM / "phonepe_state_quarter.csv").duplicated(["state_code", "year", "quarter"]).sum())
R["dh_combined_rows"] = 8  # from reports/validation/population.txt

# outliers: IQR rule on the outcome and on the predictors (flag, keep)
def iqr_flags(s):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr), float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)

out = {}
for v in ["nrega_timely_pct", "nrega_women_pd_pct", "ln_upi_txn_pc", "internet_per100", "upi_txn_pc"]:
    s = win[v].dropna()
    f, lo, hi = iqr_flags(s)
    out[v] = {"n_flagged": int(f.sum()), "pct": round(float(f.mean() * 100), 1), "lower": round(lo, 2), "upper": round(hi, 2),
              "units": sorted(win.loc[s[f].index, "state_code"].unique().tolist())}
R["outliers"] = out

# ---------------------------------------------------------------- descriptives
desc_vars = {"nrega_timely_pct": "Timely wage payment (%)", "nrega_women_pd_pct": "Women's share of person-days (%)",
             "upi_txn_pc": "UPI transactions per person", "ln_upi_txn_pc": "ln(1 + UPI per person)",
             "internet_per100": "Internet subscribers per 100", "nsdp_pc_real": "Real NSDP per capita (Rs)",
             "literacy_pct": "Literacy rate, 2011 (%)", "population": "Population (millions)"}
rows = []
for v, lab in desc_vars.items():
    s = win[v].astype(float) / (1e6 if v == "population" else 1)
    rows.append({"Variable": lab, "N": int(s.notna().sum()), "Mean": s.mean(), "SD": s.std(), "Min": s.min(),
                 "Median": s.median(), "Max": s.max(), "Skew": stats.skew(s.dropna())})
desc = pd.DataFrame(rows)
desc.to_csv(OUT / "descriptives.csv", index=False)
R["descriptives"] = desc.round(3).to_dict(orient="records")

# =========================================================================== EDA figures
m = win[win.sample_rq1 == 1].copy()
R["rq1_n"], R["rq1_states"] = int(len(m)), int(m.state_code.nunique())

# Figure: outcome distribution + by-year boxplot
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
ax[0].hist(m.nrega_timely_pct, bins=20, color=BLUE, edgecolor="white")
ax[0].axvline(m.nrega_timely_pct.median(), color=OCHRE, ls="--", lw=1)
ax[0].set_xlabel("% of wage payments generated within 15 days"); ax[0].set_ylabel("State-years")
ax[0].set_title("(a) Distribution, 2018-2025", loc="left", fontsize=9.5)
yrs = sorted(m.year.unique())
ax[1].boxplot([m.loc[m.year == y, "nrega_timely_pct"] for y in yrs], tick_labels=[str(y) for y in yrs],
              widths=.55, flierprops=dict(marker="o", ms=3, mfc=RED, mec=RED), medianprops=dict(color=OCHRE))
ax[1].set_title("(b) By year (FY t to t+1)", loc="left", fontsize=9.5); ax[1].set_ylabel("%"); ax[1].tick_params(axis="x", labelrotation=45)
save(fig, "outcome_distribution")
R["timely_median"] = float(m.nrega_timely_pct.median()); R["timely_mean"] = float(m.nrega_timely_pct.mean())
R["timely_share_ge95"] = float((m.nrega_timely_pct >= 95).mean()); R["timely_share_lt80"] = float((m.nrega_timely_pct < 80).mean())
R["timely_by_year"] = m.groupby("year").nrega_timely_pct.agg(["mean", "median", "min"]).round(2).to_dict(orient="index")

# Figure: heatmap state x year
hm = m.pivot(index="state_name", columns="year", values="nrega_timely_pct")
hm = hm.loc[hm.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(6.0, 6.4))
im = ax.imshow(hm.values, cmap="RdYlBu", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=45); ax.set_yticks(range(hm.shape[0]), hm.index, fontsize=7.5)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v < 30 or v > 90 else "#222")
fig.colorbar(im, ax=ax, fraction=0.03, label="% within 15 days")
save(fig, "timely_heatmap")
R["laggards"] = hm.mean(axis=1).sort_values().head(6).round(1).to_dict()
R["n_units_always_ge95"] = int((hm.min(axis=1) >= 95).sum())

# Figure: adoption trends
nat = panel.groupby("year").apply(lambda d: pd.Series({"upi": d.upi_txn.sum() / d.population.sum(),
                                                       "net": np.average(d.internet_per100, weights=d.population)}),
                                  include_groups=False)
q = panel.groupby("year").upi_txn_pc.quantile([.1, .5, .9]).unstack()
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.95))
ax[0].fill_between(q.index, q[.1], q[.9], color=BLUE, alpha=.18, label="10th-90th percentile of states")
ax[0].plot(q.index, q[.5], color=BLUE, marker="o", ms=3, label="Median state")
ax[0].plot(nat.index, nat.upi, color=OCHRE, marker="s", ms=3, label="India")
ax[0].set_yscale("log"); ax[0].set_ylabel("Transactions per person (log)")
ax[0].set_title("(a) PhonePe UPI transactions per person", loc="left", fontsize=9.5); ax[0].legend(frameon=False, fontsize=7.5)
qn = panel.groupby("year").internet_per100.quantile([.1, .5, .9]).unstack()
ax[1].fill_between(qn.index, qn[.1], qn[.9], color=GREY, alpha=.25)
ax[1].plot(qn.index, qn[.5], color=GREY, marker="o", ms=3, label="Median state"); ax[1].plot(nat.index, nat.net, color=OCHRE, marker="s", ms=3, label="India")
ax[1].set_title("(b) Internet subscribers per 100", loc="left", fontsize=9.5); ax[1].legend(frameon=False, fontsize=7.5)
save(fig, "adoption_trends")
R["upi_pc_india"] = nat.upi.round(2).to_dict(); R["internet_india"] = nat.net.round(1).to_dict()
R["upi_pc_p90_p10_2025"] = float(q.loc[2025, .9] / q.loc[2025, .1])
R["upi_pc_p90_p10_2018"] = float(q.loc[2018, .9] / q.loc[2018, .1])
top = panel[panel.year == 2025].nlargest(3, "upi_txn_pc")[["state_name", "upi_txn_pc"]]
bot = panel[panel.year == 2025].nsmallest(3, "upi_txn_pc")[["state_name", "upi_txn_pc"]]
R["upi_top_2025"] = dict(zip(top.state_name, top.upi_txn_pc.round(1))); R["upi_bottom_2025"] = dict(zip(bot.state_name, bot.upi_txn_pc.round(1)))

# Figure: between vs within relationship
b = m.groupby("state_code")[["ln_upi_txn_pc", "nrega_timely_pct"]].mean()
wd = m.copy()
for c in ["ln_upi_txn_pc", "nrega_timely_pct"]:
    wd[c + "_w"] = wd[c] - wd.groupby("state_code")[c].transform("mean") - wd.groupby("year")[c].transform("mean") + wd[c].mean()
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
ax[0].scatter(b.ln_upi_txn_pc, b.nrega_timely_pct, s=14, color=BLUE)
for code, r in b.iterrows():
    if r.nrega_timely_pct < 75:
        ax[0].annotate(code, (r.ln_upi_txn_pc, r.nrega_timely_pct), fontsize=7, xytext=(3, 0), textcoords="offset points")
rb = stats.spearmanr(b.ln_upi_txn_pc, b.nrega_timely_pct)
ax[0].set_xlabel("State mean ln(1 + UPI per person)"); ax[0].set_ylabel("State mean timely payment (%)")
ax[0].set_title(f"(a) Between states (Spearman ρ = {rb[0]:.2f})", loc="left", fontsize=9.5)
ax[1].scatter(wd.ln_upi_txn_pc_w, wd.nrega_timely_pct_w, s=9, color=GREY)
z = np.polyfit(wd.ln_upi_txn_pc_w, wd.nrega_timely_pct_w, 1)
xs = np.linspace(wd.ln_upi_txn_pc_w.min(), wd.ln_upi_txn_pc_w.max(), 20)
ax[1].plot(xs, np.polyval(z, xs), color=OCHRE)
rw = stats.pearsonr(wd.ln_upi_txn_pc_w, wd.nrega_timely_pct_w)
ax[1].set_xlabel("ln UPI per person, net of means"); ax[1].set_ylabel("Timely payment, net of means")
ax[1].set_title(f"(b) Within states over time (r = {rw[0]:.2f})", loc="left", fontsize=9.5)
save(fig, "between_within")
R["between_rho"], R["between_p"] = float(rb[0]), float(rb[1]); R["within_r"], R["within_p"] = float(rw[0]), float(rw[1])

# Figure: correlation matrix
cvars = {"nrega_timely_pct": "Timely pay %", "nrega_women_pd_pct": "Women PD %", "ln_upi_txn_pc": "ln UPI pc",
         "internet_per100": "Internet/100", "literacy_pct": "Literacy", "ln_nsdp_base": "ln NSDP 2021"}
cm = m[list(cvars)].corr(method="spearman").rename(index=cvars, columns=cvars)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm)), cm.columns, rotation=40, ha="right"); ax.set_yticks(range(len(cm)), cm.index)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(cm.values[i, j]) > .6 else "#222")
fig.colorbar(im, ax=ax, fraction=.04)
save(fig, "correlations")
R["corr"] = cm.round(2).to_dict()

# Figure: women's share of person-days
wp = m.groupby("state_name").nrega_women_pd_pct.mean().sort_values()
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.barh(wp.index, wp.values, color=[RED if v < 33 else BLUE for v in wp.values], height=.65)
ax.axvline(33.3, color=OCHRE, ls="--", lw=1); ax.text(34, 0.2, "one-third", fontsize=7.5, color=OCHRE)
ax.set_xlabel("Women's share of person-days, mean 2021-2025 (%)"); ax.tick_params(axis="y", labelsize=6.5)
save(fig, "women_share")
R["women_below_third"] = wp[wp < 33.3].round(1).to_dict(); R["women_mean"] = float(m.nrega_women_pd_pct.mean())
R["women_change"] = float(m[m.year == 2025].nrega_women_pd_pct.mean() - m[m.year == 2021].nrega_women_pd_pct.mean())

# Figure: TRAI state-table anomaly (documented in cleaning)
# (values read from the 2021 report, Table 1.33 vs Table 1.32; see reports/validation and README)

# =========================================================================== models
models = {}
# RQ1 - primary two-way FE
f1 = "nrega_timely_pct ~ ln_upi_txn_pc + internet_per100" + FE
r1 = cl(f1, m)
models["rq1_fe"] = {"n": int(r1.nobs), "coef": coefs(r1, ["ln_upi_txn_pc", "internet_per100"]),
                    "wald": wald(r1, ["ln_upi_txn_pc", "internet_per100"]), "within_r2": within_r2(r1, m, "nrega_timely_pct"),
                    "r2": float(r1.rsquared)}
# residual serial correlation -> design effect
e = pd.Series(r1.resid.values, index=m.index); lag = e.groupby(m.state_code).shift(1)
rho = float(np.corrcoef(e[lag.notna()], lag[lag.notna()])[0, 1]); models["rq1_fe"]["resid_rho"] = rho
# pooled OLS benchmark (between + within)
r1p = cl("nrega_timely_pct ~ ln_upi_txn_pc + internet_per100 + literacy_pct + C(year)", m)
models["rq1_pooled"] = {"n": int(r1p.nobs), "coef": coefs(r1p, ["ln_upi_txn_pc", "internet_per100", "literacy_pct"]),
                        "r2": float(r1p.rsquared)}
# robustness: transformed outcome, lag, NSDP control, excluding outliers / small units
r1t = cl("nrega_shortfall_log ~ ln_upi_txn_pc + internet_per100" + FE, m)
models["rq1_shortfall"] = {"n": int(r1t.nobs), "coef": coefs(r1t, ["ln_upi_txn_pc", "internet_per100"]),
                           "wald": wald(r1t, ["ln_upi_txn_pc", "internet_per100"])}
ml = m.sort_values(["state_code", "year"]).copy()
ml["ln_upi_lag"] = ml.groupby("state_code").ln_upi_txn_pc.shift(1); ml["net_lag"] = ml.groupby("state_code").internet_per100.shift(1)
ml = ml.dropna(subset=["ln_upi_lag", "net_lag"])
r1l = cl("nrega_timely_pct ~ ln_upi_lag + net_lag" + FE, ml)
models["rq1_lagged"] = {"n": int(r1l.nobs), "coef": coefs(r1l, ["ln_upi_lag", "net_lag"]), "wald": wald(r1l, ["ln_upi_lag", "net_lag"])}
mn = m.dropna(subset=["ln_nsdp_pc"])
r1n = cl("nrega_timely_pct ~ ln_upi_txn_pc + internet_per100 + ln_nsdp_pc" + FE, mn)
models["rq1_nsdp"] = {"n": int(r1n.nobs), "years": sorted(mn.year.unique().tolist()),
                      "coef": coefs(r1n, ["ln_upi_txn_pc", "internet_per100", "ln_nsdp_pc"])}
big = m[(m.population > 2e6)]
r1b = cl(f1, big)
models["rq1_large_states"] = {"n": int(r1b.nobs), "states": int(big.state_code.nunique()),
                              "coef": coefs(r1b, ["ln_upi_txn_pc", "internet_per100"])}
# post hoc power and MDE: see src/models/power_recheck.py (single source for Table 7)

# RQ1 predictive comparison: rolling origin (train <= t, test t+1)
def fit_predict(kind, tr, te):
    if kind == "naive":
        last = tr[tr.year == tr.year.max()].set_index("state_code").nrega_timely_pct
        return te.state_code.map(last).values
    if kind == "state_mean":
        mu = tr.groupby("state_code").nrega_timely_pct.mean()
        return te.state_code.map(mu).values
    if kind == "pooled":
        r = smf.ols("nrega_timely_pct ~ ln_upi_txn_pc + internet_per100 + literacy_pct + year", data=tr).fit()
        return r.predict(te).values
    if kind == "fe":
        r = smf.ols("nrega_timely_pct ~ ln_upi_txn_pc + internet_per100 + C(state_code) + year", data=tr).fit()
        return r.predict(te).values
    if kind == "rf":
        X = ["ln_upi_txn_pc", "internet_per100", "literacy_pct", "year"]
        trr = tr.copy(); tee = te.copy()
        mu = tr.groupby("state_code").nrega_timely_pct.mean()          # state history as a feature
        trr["state_hist"] = trr.state_code.map(mu); tee["state_hist"] = tee.state_code.map(mu)
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=SEED)
        rf.fit(trr[X + ["state_hist"]], trr.nrega_timely_pct)
        pi_ = permutation_importance(rf, tee[X + ["state_hist"]], tee.nrega_timely_pct, n_repeats=30,
                                     random_state=SEED, scoring="neg_root_mean_squared_error")
        OOS_PI.append(dict(zip(X + ["state_hist"], pi_.importances_mean)))
        return rf.predict(tee[X + ["state_hist"]])

kinds = ["naive", "state_mean", "pooled", "fe", "rf"]
OOS_PI = []
res_rows = []
for t in (2019, 2020, 2021, 2022, 2023, 2024):
    tr, te = m[m.year <= t], m[m.year == t + 1]
    te = te[te.state_code.isin(tr.state_code)]
    for k in kinds:
        pred = fit_predict(k, tr, te)
        err = te.nrega_timely_pct.values - pred
        res_rows.append({"model": k, "test_year": t + 1, "n": len(te), "rmse": float(np.sqrt(np.nanmean(err ** 2))),
                         "mae": float(np.nanmean(np.abs(err)))})
cv = pd.DataFrame(res_rows)
cv.to_csv(OUT / "rolling_origin.csv", index=False)
summ = cv.groupby("model").agg(rmse=("rmse", "mean"), mae=("mae", "mean")).reindex(kinds)
models["rolling"] = summ.round(3).to_dict(orient="index")
models["rolling_by_year"] = cv.round(3).to_dict(orient="records")

# leave-one-state-out for pooled vs RF without state history (generalisation to unseen states)
X = ["ln_upi_txn_pc", "internet_per100", "literacy_pct", "year"]
lo_rows = []
for s in m.state_code.unique():
    tr, te = m[m.state_code != s], m[m.state_code == s]
    rp = smf.ols("nrega_timely_pct ~ ln_upi_txn_pc + internet_per100 + literacy_pct + year", data=tr).fit().predict(te)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED).fit(tr[X], tr.nrega_timely_pct).predict(te[X])
    mu = tr.nrega_timely_pct.mean()
    for k, pr in (("pooled", rp.values), ("rf", rf), ("grand_mean", np.full(len(te), mu))):
        lo_rows.append({"model": k, "state": s, "se": float(np.mean((te.nrega_timely_pct.values - pr) ** 2)),
                        "ae": float(np.mean(np.abs(te.nrega_timely_pct.values - pr)))})
lo = pd.DataFrame(lo_rows).groupby("model").agg(rmse=("se", lambda x: float(np.sqrt(x.mean()))), mae=("ae", "mean"))
models["loso"] = lo.round(3).to_dict(orient="index")

# feature importance, out of sample: permutation on each rolling-origin test year (increase in RMSE, points)
oos = pd.DataFrame(OOS_PI)
models["perm_importance_oos"] = {k: {"mean": float(oos[k].mean()), "sd": float(oos[k].std())} for k in oos.columns}
# in-sample importance kept for comparison only (fitted and scored on the same data)
rf_all = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=SEED).fit(m[X], m.nrega_timely_pct)
pi = permutation_importance(rf_all, m[X], m.nrega_timely_pct, n_repeats=30, random_state=SEED)
models["perm_importance"] = {x: {"mean": float(a), "sd": float(b)} for x, a, b in zip(X, pi.importances_mean, pi.importances_std)}
# standardised FE coefficients
sd = m[["ln_upi_txn_pc", "internet_per100", "nrega_timely_pct"]].std()
models["rq1_fe"]["std_coef"] = {v: float(r1.params[v] * sd[v] / sd["nrega_timely_pct"]) for v in ["ln_upi_txn_pc", "internet_per100"]}

# RQ2 - moderation
m2 = win[win.sample_rq2 == 1].copy()
m2["lit_c"] = m2.literacy_pct - m2.literacy_pct.mean(); m2["inc_c"] = m2.ln_nsdp_base - m2.ln_nsdp_base.mean()
r2 = cl("nrega_timely_pct ~ adoption_index + adoption_index:lit_c + adoption_index:inc_c" + FE, m2)
terms2 = ["adoption_index", "adoption_index:lit_c", "adoption_index:inc_c"]
q25, q75 = m2.literacy_pct.quantile([.25, .75])
slopes = {}
for lab, lit in (("lit_p25", q25), ("lit_p75", q75)):
    slopes[lab] = float(r2.params["adoption_index"] + r2.params["adoption_index:lit_c"] * (lit - m2.literacy_pct.mean()))
models["rq2"] = {"n": int(r2.nobs), "states": int(m2.state_code.nunique()), "coef": coefs(r2, terms2),
                 "wald_interactions": wald(r2, terms2[1:]), "slopes": slopes, "lit_q": [float(q25), float(q75)]}

# RQ4 - inclusion
m4 = win[win.sample_rq4 == 1].copy()
r4 = cl("nrega_women_pd_pct ~ adoption_index" + FE, m4)
r4c = cl("nrega_women_pd_pct ~ ln_upi_txn_pc + internet_per100" + FE, m4)
models["rq4"] = {"n": int(r4.nobs), "coef": coefs(r4, ["adoption_index"]), "within_r2": within_r2(r4, m4, "nrega_women_pd_pct"),
                 "components": coefs(r4c, ["ln_upi_txn_pc", "internet_per100"]),
                 "pooled_spearman": float(stats.spearmanr(m4.adoption_index, m4.nrega_women_pd_pct)[0])}
# partial correlation from t: r = t / sqrt(t^2 + df)
t4 = r4.tvalues["adoption_index"]; models["rq4"]["partial_r"] = float(t4 / np.sqrt(t4 ** 2 + r4.df_resid))

# robustness: original 2021-2025 window, and pre/post-2021 split
m21 = m[m.year >= 2021]
r1w = cl(f1, m21)
models["rq1_2021on"] = {"n": int(r1w.nobs), "coef": coefs(r1w, ["ln_upi_txn_pc", "internet_per100"]),
                        "wald": wald(r1w, ["ln_upi_txn_pc", "internet_per100"])}
mpre = m[m.year <= 2021]
r1e = cl(f1, mpre)
models["rq1_2018_21"] = {"n": int(r1e.nobs), "coef": coefs(r1e, ["ln_upi_txn_pc", "internet_per100"]),
                         "wald": wald(r1e, ["ln_upi_txn_pc", "internet_per100"])}
nb = m[(m.flag_jk_incl_ladakh != 1) & (m.flag_ladakh_not_separate != 1) & (m.flag_dh_premerger != 1)
       & ~((m.state_code.isin(["JK", "LA"])) & (m.year <= 2019))]
r1x = cl(f1, nb)
models["rq1_excl_boundary"] = {"n": int(r1x.nobs), "coef": coefs(r1x, ["ln_upi_txn_pc", "internet_per100"]),
                               "wald": wald(r1x, ["ln_upi_txn_pc", "internet_per100"])}
# outcome validation: R14.9 vs live dashboard (same indicator, different snapshot)
vd = win[["nrega_timely_pct", "nrega_timely_dash_pct"]].dropna()
R["outcome_validation"] = {"n": int(len(vd)), "pearson": float(vd.corr().iloc[0, 1]),
                           "spearman": float(vd.corr(method="spearman").iloc[0, 1]),
                           "mean_abs_diff": float((vd.iloc[:, 0] - vd.iloc[:, 1]).abs().mean()),
                           "median_abs_diff": float((vd.iloc[:, 0] - vd.iloc[:, 1]).abs().median())}
# RQ2 robustness on the 2021-2025 window
m2w = m2[m2.year >= 2021]
r2w = cl("nrega_timely_pct ~ adoption_index + adoption_index:lit_c + adoption_index:inc_c" + FE, m2w)
models["rq2_2021on"] = {"n": int(r2w.nobs), "coef": coefs(r2w, terms2), "wald_interactions": wald(r2w, terms2[1:])}

R["models"] = models
(OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
print(json.dumps({k: models[k] for k in ["rq1_fe", "rolling", "loso", "perm_importance_oos", "rq2", "rq4"]}, indent=1, default=float)[:6000])
