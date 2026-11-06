"""Sensitivity analyses for the RQ1, RQ2 and RQ4 models.

    python -m src.models.sensitivity

Writes reports/interim/sensitivity.json and reports/interim/rq1_leave_one_state_out.csv.
All analyses use the same samples as src/models/interim_analysis.py and seed 640.

  1. Wild cluster bootstrap (restricted, Rademacher weights, 1,999 draws) p-values for the
     key coefficients, because 29-31 clusters is few for asymptotic clustered inference
     (Cameron et al., 2008).
  2. Leave-one-state-out influence: the RQ1 primary model re-estimated 31 times, each time
     without one state.
  3. Quantile regression with state and year dummies at q = .25 and .50, with state-cluster
     bootstrap SEs, to check that the results are not an artefact of the 100% ceiling.
  4. Multiple imputation (m = 20, chained equations with posterior draws) of the missing
     annual NSDP values, so the income-controlled RQ1 model can use all eight years;
     estimates pooled with Rubin's rules.
  5. Holm adjustment for the two exploratory RQ4 component coefficients.
  6. Minimum detectable coefficients: (t_.975 + t_.80) x SE with G - 1 df, the effect size
     that would have been detected with 80% power given the observed precision.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import PROCESSED, REPORTS  # noqa: E402

warnings.filterwarnings("ignore")
SEED, B = 640, 1999
FE = " + C(state_code) + C(year)"
OUT = REPORTS / "interim" / "sensitivity.json"


def cl(formula, d):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["state_code"]})


# ------------------------------------------------------------------ wild cluster bootstrap
def _design(formula, d):
    import patsy
    y, X = patsy.dmatrices(formula, d, return_type="dataframe")
    return y.values.ravel(), X


def _cluster_t(y, X, g, k):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gg in np.unique(g):
        s = X[g == gg].T @ u[g == gg]
        meat += np.outer(s, s)
    G, n, K = len(np.unique(g)), len(y), np.linalg.matrix_rank(X)
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (n - 1) / (n - K)
    return b[k] / np.sqrt(V[k, k])


def wild_p(formula, d, term, rng):
    y, Xdf = _design(formula, d)
    X, g = Xdf.values, d["state_code"].values
    k = list(Xdf.columns).index(term)
    t0 = _cluster_t(y, X, g, k)
    Xr = np.delete(X, k, axis=1)                       # impose H0: beta_k = 0
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fit_r, u_r = Xr @ br, y - Xr @ br
    groups = np.unique(g)
    idx = np.searchsorted(groups, g)
    ts = np.empty(B)
    for i in range(B):
        w = rng.choice([-1.0, 1.0], size=len(groups))[idx]
        ts[i] = _cluster_t(fit_r + u_r * w, X, g, k)
    return float(t0), float((np.abs(ts) >= abs(t0)).mean())


def _cluster_wald(y, X, g, ks):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gg in np.unique(g):
        s = X[g == gg].T @ u[g == gg]
        meat += np.outer(s, s)
    G, n, K = len(np.unique(g)), len(y), np.linalg.matrix_rank(X)
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (n - 1) / (n - K)
    bk = b[ks]
    return float(bk @ np.linalg.pinv(V[np.ix_(ks, ks)]) @ bk / len(ks))


def wild_joint_p(formula, d, terms, rng):
    """Wild cluster bootstrap p-value for the joint Wald test that all `terms` are zero."""
    y, Xdf = _design(formula, d)
    X, g = Xdf.values, d["state_code"].values
    ks = [list(Xdf.columns).index(t) for t in terms]
    w0 = _cluster_wald(y, X, g, ks)
    Xr = np.delete(X, ks, axis=1)
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fit_r, u_r = Xr @ br, y - Xr @ br
    groups = np.unique(g)
    idx = np.searchsorted(groups, g)
    ws = np.empty(B)
    for i in range(B):
        w = rng.choice([-1.0, 1.0], size=len(groups))[idx]
        ws[i] = _cluster_wald(fit_r + u_r * w, X, g, ks)
    return w0, float((ws >= w0).mean())


# ------------------------------------------------------------------ helpers
def mdc(res, term, G):
    """Minimum detectable coefficient at 80% power, two-sided alpha = .05, df = G - 1."""
    df = G - 1
    return float((stats.t.ppf(.975, df) + stats.t.ppf(.80, df)) * res.bse[term])


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def rubin(ests, ses):
    ests, ses = np.asarray(ests), np.asarray(ses)
    m = len(ests)
    qbar, ubar, bvar = ests.mean(), (ses ** 2).mean(), ests.var(ddof=1)
    T = ubar + (1 + 1 / m) * bvar
    df = (m - 1) * (1 + ubar / ((1 + 1 / m) * bvar)) ** 2 if bvar > 0 else np.inf
    t = qbar / np.sqrt(T)
    return {"b": float(qbar), "se": float(np.sqrt(T)), "p": float(2 * stats.t.sf(abs(t), df)),
            "fmi": float((1 + 1 / m) * bvar / T)}


def main():
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv")
    m1 = panel[panel.sample_rq1 == 1].copy()
    m2 = panel[panel.sample_rq2 == 1].copy()
    m2["lit_c"] = m2.literacy_pct - m2.literacy_pct.mean()
    m2["inc_c"] = m2.ln_nsdp_base - m2.ln_nsdp_base.mean()
    m4 = panel[panel.sample_rq4 == 1].copy()
    S = {}

    f1 = "nrega_timely_pct ~ ln_upi_txn_pc + internet_per100" + FE
    f2 = "nrega_timely_pct ~ adoption_index + adoption_index:lit_c + adoption_index:inc_c" + FE
    f4 = "nrega_women_pd_pct ~ adoption_index" + FE
    f4c = "nrega_women_pd_pct ~ ln_upi_txn_pc + internet_per100" + FE
    r1, r2, r4, r4c = cl(f1, m1), cl(f2, m2), cl(f4, m4), cl(f4c, m4)

    # 1. wild cluster bootstrap
    wb = {}
    for lab, f, d, term in (("rq1_upi", f1, m1, "ln_upi_txn_pc"), ("rq1_internet", f1, m1, "internet_per100"),
                            ("rq2_lit", f2, m2, "adoption_index:lit_c"), ("rq2_inc", f2, m2, "adoption_index:inc_c"),
                            ("rq4_index", f4, m4, "adoption_index"),
                            ("rq4_upi", f4c, m4, "ln_upi_txn_pc"), ("rq4_internet", f4c, m4, "internet_per100")):
        t0, p = wild_p(f, d, term, rng)
        wb[lab] = {"t": t0, "p_wild": p}
        print(f"wild bootstrap {lab}: t = {t0:.2f}, p = {p:.3f}", flush=True)
    jw = {}
    for lab, f, d, terms in (("rq1_joint", f1, m1, ["ln_upi_txn_pc", "internet_per100"]),
                             ("rq2_joint", f2, m2, ["adoption_index:lit_c", "adoption_index:inc_c"])):
        w0, p = wild_joint_p(f, d, terms, rng)
        jw[lab] = {"F": w0, "p_wild": p}
        print(f"wild bootstrap joint {lab}: F = {w0:.2f}, p = {p:.3f}", flush=True)
    S["wild_bootstrap"] = {"draws": B, "weights": "Rademacher", "null_imposed": True, "results": wb, "joint": jw}

    # 2. leave-one-state-out influence (RQ1)
    rows = []
    for s in sorted(m1.state_code.unique()):
        d = m1[m1.state_code != s]
        r = cl(f1, d)
        rows.append({"dropped": s, "n": int(r.nobs),
                     "b_internet": float(r.params["internet_per100"]), "p_internet": float(r.pvalues["internet_per100"]),
                     "b_upi": float(r.params["ln_upi_txn_pc"]), "p_upi": float(r.pvalues["ln_upi_txn_pc"])})
    inf = pd.DataFrame(rows)
    inf.to_csv(REPORTS / "interim" / "rq1_leave_one_state_out.csv", index=False)
    S["influence"] = {"b_internet_min": float(inf.b_internet.min()), "b_internet_max": float(inf.b_internet.max()),
                      "state_min": inf.loc[inf.b_internet.idxmin(), "dropped"], "state_max": inf.loc[inf.b_internet.idxmax(), "dropped"],
                      "n_states": int(len(inf)), "n_min": int(inf.n.min()), "n_max": int(inf.n.max()),
                      "n_p_above_05": int((inf.p_internet > .05).sum()),
                      "states_p_above_05": inf.loc[inf.p_internet > .05, "dropped"].tolist(),
                      "b_upi_min": float(inf.b_upi.min()), "b_upi_max": float(inf.b_upi.max()),
                      "n_upi_p_below_05": int((inf.p_upi < .05).sum())}
    print("influence:", S["influence"], flush=True)

    # 3. quantile regression with state and year dummies, cluster bootstrap SEs
    qr = {}
    states = m1.state_code.unique()
    for q in (.25, .50):
        fit = smf.quantreg(f1, m1).fit(q=q, max_iter=1000)
        boots = []
        for _ in range(200):
            pick = rng.choice(states, size=len(states), replace=True)
            d = pd.concat([m1[m1.state_code == s].assign(state_code=f"{s}_{j}") for j, s in enumerate(pick)])
            try:
                bf = smf.quantreg(f1, d).fit(q=q, max_iter=1000)
                boots.append([bf.params["ln_upi_txn_pc"], bf.params["internet_per100"]])
            except Exception:
                continue
        boots = np.array(boots)
        out = {}
        for j, term in enumerate(["ln_upi_txn_pc", "internet_per100"]):
            se = float(boots[:, j].std(ddof=1)); b = float(fit.params[term])
            out[term] = {"b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                         "p": float(2 * stats.norm.sf(abs(b / se)))}
        qr[f"q{int(q * 100)}"] = {"n": int(len(m1)), "boot_draws": int(len(boots)), "coef": out}
        print(f"quantile {q}:", {k: (round(v["b"], 2), round(v["p"], 3)) for k, v in out.items()}, flush=True)
    S["quantile"] = qr

    # 4. multiple imputation of annual ln NSDP (RQ1 with income control, all eight years)
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    base = m1[["state_code", "year", "nrega_timely_pct", "ln_upi_txn_pc", "internet_per100", "literacy_pct", "ln_nsdp_pc"]].copy()
    D = pd.get_dummies(base.state_code, prefix="s", dtype=float)
    Z = pd.concat([base[["year", "ln_upi_txn_pc", "internet_per100", "literacy_pct", "nrega_timely_pct", "ln_nsdp_pc"]], D], axis=1)
    est = {t: ([], []) for t in ("ln_upi_txn_pc", "internet_per100", "ln_nsdp_pc")}
    for i in range(20):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=25, random_state=SEED + i)
        filled = pd.DataFrame(imp.fit_transform(Z), columns=Z.columns, index=Z.index)
        d = base.copy(); d["ln_nsdp_pc"] = filled["ln_nsdp_pc"]
        r = cl("nrega_timely_pct ~ ln_upi_txn_pc + internet_per100 + ln_nsdp_pc" + FE, d)
        for t in est:
            est[t][0].append(r.params[t]); est[t][1].append(r.bse[t])
    S["mi_nsdp"] = {"m": 20, "n": int(len(base)), "n_imputed": int(base.ln_nsdp_pc.isna().sum()),
                    "coef": {t: rubin(*est[t]) for t in est}}
    print("MI:", {t: (round(v["b"], 2), round(v["p"], 3)) for t, v in S["mi_nsdp"]["coef"].items()}, flush=True)

    # 5. Holm for exploratory RQ4 components (family of two)
    raw = {t: float(r4c.pvalues[t]) for t in ("ln_upi_txn_pc", "internet_per100")}
    S["holm_rq4_components"] = {"raw": raw, "adjusted": holm(raw),
                                "b": {t: float(r4c.params[t]) for t in raw},
                                "adjusted_wild": holm({"ln_upi_txn_pc": wb["rq4_upi"]["p_wild"], "internet_per100": wb["rq4_internet"]["p_wild"]})}

    # 6. minimum detectable coefficients
    G1, G2, G4 = m1.state_code.nunique(), m2.state_code.nunique(), m4.state_code.nunique()
    S["mdc"] = {
        "rq1_upi": {"b": float(r1.params["ln_upi_txn_pc"]), "se": float(r1.bse["ln_upi_txn_pc"]), "mdc": mdc(r1, "ln_upi_txn_pc", G1), "G": int(G1)},
        "rq1_internet": {"b": float(r1.params["internet_per100"]), "se": float(r1.bse["internet_per100"]), "mdc": mdc(r1, "internet_per100", G1), "G": int(G1)},
        "rq2_lit": {"b": float(r2.params["adoption_index:lit_c"]), "se": float(r2.bse["adoption_index:lit_c"]), "mdc": mdc(r2, "adoption_index:lit_c", G2), "G": int(G2)},
        "rq2_inc": {"b": float(r2.params["adoption_index:inc_c"]), "se": float(r2.bse["adoption_index:inc_c"]), "mdc": mdc(r2, "adoption_index:inc_c", G2), "G": int(G2)},
        "rq4_index": {"b": float(r4.params["adoption_index"]), "se": float(r4.bse["adoption_index"]), "mdc": mdc(r4, "adoption_index", G4), "G": int(G4)},
    }
    S["sd"] = {"lit": float(m2.literacy_pct.std()), "inc": float(m2.ln_nsdp_base.std()),
               "idx4": float(m4.adoption_index.std()), "upi1": float(m1.ln_upi_txn_pc.std()), "net1": float(m1.internet_per100.std())}
    for v in S["mdc"].values():
        tc = stats.t.ppf(.975, v["G"] - 1)
        v["lo_t"], v["hi_t"] = float(v["b"] - tc * v["se"]), float(v["b"] + tc * v["se"])
    print("MDC:", {k: round(v["mdc"], 3) for k, v in S["mdc"].items()}, flush=True)

    OUT.write_text(json.dumps(S, indent=1, default=float))
    print("->", OUT)


if __name__ == "__main__":
    main()
