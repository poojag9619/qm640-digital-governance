"""Figures for the interim report built from reports/interim/results.json."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
R = json.load(open(ROOT / "reports/interim/results.json")); M = R["models"]
FIG = ROOT / "reports/figures"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
BLUE, GREY, OCHRE = "#2f5d8a", "#8a96a3", "#b07a2a"
cv = pd.DataFrame(M["rolling_by_year"])
lab = {"naive": "Naive (last year)", "state_mean": "State mean", "pooled": "Pooled OLS", "fe": "Two-way FE", "rf": "Random forest"}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
for k, c, mk in (("naive", GREY, "o"), ("state_mean", "#bbbbbb", "s"), ("pooled", OCHRE, "^"), ("fe", BLUE, "D"), ("rf", "#4b8b5a", "v")):
    d = cv[cv.model == k]; ax[0].plot(d.test_year, d.rmse, marker=mk, ms=4, color=c, label=lab[k])
ax[0].set_xticks(sorted(cv.test_year.unique())); ax[0].set_ylabel("RMSE (percentage points)"); ax[0].set_xlabel("Test year")
ax[0].set_title("(a) Rolling-origin RMSE by test year", loc="left", fontsize=8.5); ax[0].set_ylim(0, 55); ax[0].legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.0))
s = pd.DataFrame(M["rolling"]).T.reindex(list(lab))
ax[1].barh([lab[k] for k in s.index], s.rmse, color=[GREY, "#bbbbbb", OCHRE, BLUE, "#4b8b5a"], height=.6)
for i, v in enumerate(s.rmse): ax[1].text(v + .2, i, f"{v:.2f}", va="center", fontsize=7)
ax[1].set_xlabel("Mean RMSE, 2020-2025"); ax[1].set_title("(b) Average across test years", loc="left", fontsize=8.5); ax[1].invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "interim_model_performance.png"); plt.close(fig)
# coefficient plot
rows = [("RQ1: ln UPI per person → timely %", M["rq1_fe"]["coef"]["ln_upi_txn_pc"]),
        ("RQ1: internet per 100 (×10) → timely %", {k: v * 10 for k, v in M["rq1_fe"]["coef"]["internet_per100"].items() if k in ("b", "lo", "hi")}),
        ("RQ4: adoption index → women %", M["rq4"]["coef"]["adoption_index"]),
        ("RQ4: ln UPI per person → women %", M["rq4"]["components"]["ln_upi_txn_pc"]),
        ("RQ4: internet per 100 (×10) → women %", {k: v * 10 for k, v in M["rq4"]["components"]["internet_per100"].items() if k in ("b", "lo", "hi")})]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for i, (name, c) in enumerate(rows):
    ax.plot([c["lo"], c["hi"]], [i, i], color=BLUE, lw=2); ax.plot(c["b"], i, "o", color=OCHRE, ms=5)
ax.axvline(0, color="#555", lw=.8, ls="--"); ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("Coefficient (percentage points) with 95% state-clustered CI")
fig.tight_layout(); fig.savefig(FIG / "interim_coefficients.png"); plt.close(fig)
print("ok")
