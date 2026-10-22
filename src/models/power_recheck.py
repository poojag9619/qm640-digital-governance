"""Re-check of required and achieved sample sizes for the interim specifications.

    python -m src.models.power_recheck

Same method as the synopsis (src/models/power_analysis.py): power of the F-test for an
R-squared increase, alpha = .05, lambda = f2 * N, df2 = N - p - 1, where p counts every
model term including the state and year fixed-effect dummies. Repeated observations are
handled with the Kish design effect 1 + (m - 1) * rho, where m = years per state; rho is
set to the planning value .10 and to the residual serial correlation estimated from each
model (floored at 0). Writes reports/interim/power_recheck.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import PROCESSED, REPORTS  # noqa: E402

FE = " + C(state_code) + C(year)"
SPECS = {  # RQ: (sample flag, formula, tested terms u, f2 from the synopsis)
    "RQ1": ("sample_rq1", "nrega_timely_pct ~ ln_upi_txn_pc + internet_per100", 2, 0.10),
    "RQ2": ("sample_rq2", "nrega_timely_pct ~ adoption_index + adoption_index:lit_c + adoption_index:inc_c", 2, 0.08),
    "RQ4": ("sample_rq4", "nrega_women_pd_pct ~ adoption_index", 1, 0.25 ** 2 / (1 - 0.25 ** 2)),
}


def power(n, u, p, f2, alpha=0.05):
    df2 = n - p - 1
    if df2 <= 0:
        return 0.0
    crit = stats.f.ppf(1 - alpha, u, df2)
    return float(1 - stats.ncf.cdf(crit, u, df2, f2 * n))


def main():
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv")
    out = {}
    for rq, (flag, formula, u, f2) in SPECS.items():
        d = panel[panel[flag] == 1].copy()
        d["lit_c"] = d.literacy_pct - d.literacy_pct.mean()
        d["inc_c"] = d.ln_nsdp_base - d.ln_nsdp_base.mean()
        S, T = d.state_code.nunique(), d.year.nunique()
        n_terms = formula.split("~")[1].count("+") + 1
        p = n_terms + (S - 1) + (T - 1)
        n_srs = next(n for n in range(p + 3, 5000) if power(n, u, p, f2) >= 0.80)
        res = smf.ols(formula + FE, data=d).fit()
        e = pd.Series(res.resid.values, index=d.index)
        lag = e.groupby(d.state_code).shift(1)
        rho = float(np.corrcoef(e[lag.notna()], lag[lag.notna()])[0, 1])
        m = len(d) / S
        deff10, deffhat = 1 + (m - 1) * 0.10, 1 + (m - 1) * max(rho, 0)
        out[rq] = {"u": u, "p": int(p), "f2": round(f2, 4), "units": int(S), "years": int(T),
                   "m": round(m, 2), "n_srs": int(n_srs), "df2": int(n_srs - p - 1),
                   "fcrit": round(float(stats.f.ppf(.95, u, n_srs - p - 1)), 3), "lam": round(f2 * n_srs, 2),
                   "power": round(power(n_srs, u, p, f2), 4),
                   "deff_rho10": round(deff10, 3), "nominal_rho10": int(math.ceil(n_srs * deff10)),
                   "n_actual": int(len(d)), "rho_hat": round(rho, 3), "deff_rhohat": round(deffhat, 3),
                   "power_actual_rho10": round(power(int(len(d) / deff10), u, p, f2), 3),
                   "power_actual_rhohat": round(power(int(len(d) / deffhat), u, p, f2), 3),
                   "power_actual_srs": round(power(len(d), u, p, f2), 3)}
    path = REPORTS / "interim" / "power_recheck.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
