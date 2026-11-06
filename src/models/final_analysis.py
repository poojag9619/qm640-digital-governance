"""Final-report predictive evaluation: training vs test accuracy, hold-out split, classification
of lagging state-years (ROC, confusion matrix) and out-of-sample feature importance.

    python -m src.models.final_analysis

Inputs : data/processed/analysis_panel.csv (RQ1 sample: sample_rq1 == 1)
Outputs: reports/final/predictive_results.json, rolling_regression.csv, rolling_classification.csv,
         holdout_predictions.csv, and figures reports/figures/final_*.png

Design (no information leakage):
  * Rolling origin: for t = 2019..2024, train on years <= t and test on year t + 1. Every
    preprocessing step (state-history mean, scaling) is fitted on the training years only.
  * Common test set: test rows must have the previous year's outcome, so every model is scored
    on exactly the same state-years.
  * Hold-out split: train 2018-2023, test 2024-2025 (one-step-ahead: the 2025 prediction may use
    the observed 2024 value, as it would in practice).
  * Feature set A: ln UPI per person, internet per 100, literacy, year, state-history mean.
    Feature set B: A + last year's timely-payment rate (lag1).
  * Hyperparameters are fixed in advance (no tuning on test years). The neural network gets no
    calendar-year input, because a network extrapolates linearly in year beyond the training range.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,  # noqa: E402
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import PROCESSED, REPORTS  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 640
OUT = REPORTS / "final"
FIG = REPORTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
Y = "nrega_timely_pct"
XA = ["ln_upi_txn_pc", "internet_per100", "literacy_pct", "year", "state_hist"]
XB = XA + ["lag1"]
THRESH = 80.0                     # a state-year is "lagging" when fewer than 80% of payments are on time

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220})
BLUE, GREY, OCHRE, GREEN, RED, PURPLE = "#2f5d8a", "#8a96a3", "#b07a2a", "#4b8b5a", "#a33b3b", "#6a4c93"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def add_features(tr: pd.DataFrame, te: pd.DataFrame):
    """State-history mean from training rows only; lag1 from the full outcome history before each year."""
    mu = tr.groupby("state_code")[Y].mean()
    tr, te = tr.copy(), te.copy()
    tr["state_hist"] = tr.state_code.map(mu)
    te["state_hist"] = te.state_code.map(mu)
    return tr, te


REG = {
    "naive": "Naive (last year)",
    "state_mean": "State mean",
    "pooled": "Pooled OLS",
    "fe": "Two-way FE (state FE + trend)",
    "rf_a": "Random forest, set A",
    "gbm_a": "Gradient boosting, set A",
    "mlp_a": "Neural network (MLP), set A",
    "rf_b": "Random forest, set B (+ lag)",
    "gbm_b": "Gradient boosting, set B (+ lag)",
    "mlp_b": "Neural network (MLP), set B (+ lag)",
}


def reg_model(kind):
    if kind.startswith("rf"):
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    if kind.startswith("gbm"):
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED)
    if kind.startswith("mlp"):
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, learning_rate_init=0.01,
                                                            max_iter=3000, random_state=SEED))
    raise ValueError(kind)


def fit_predict_reg(kind, tr, te):
    """Return (train predictions, test predictions)."""
    if kind == "naive":
        return tr["lag1"].values, te["lag1"].values
    if kind == "state_mean":
        mu = tr.groupby("state_code")[Y].mean()
        return tr.state_code.map(mu).values, te.state_code.map(mu).values
    if kind == "pooled":
        r = smf.ols(f"{Y} ~ ln_upi_txn_pc + internet_per100 + literacy_pct + year", data=tr).fit()
        return r.fittedvalues.values, r.predict(te).values
    if kind == "fe":
        r = smf.ols(f"{Y} ~ ln_upi_txn_pc + internet_per100 + C(state_code) + year", data=tr).fit()
        return r.fittedvalues.values, r.predict(te).values
    cols = XB if kind.endswith("_b") else XA
    if kind.startswith("mlp"):            # a neural network extrapolates linearly in "year" beyond the
        cols = [c for c in cols if c != "year"]   # training range, so it gets no calendar-year input
    mdl = reg_model(kind).fit(tr[cols], tr[Y])
    return mdl.predict(tr[cols]), mdl.predict(te[cols])


CLS = {
    "naive": "Naive (last year's status)",
    "logit": "Logistic regression",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "mlp": "Neural network (MLP)",
}


def cls_model(kind):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    if kind == "gbm":
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED)
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, learning_rate_init=0.01,
                                                             max_iter=3000, random_state=SEED))
    raise ValueError(kind)


def fit_predict_cls(kind, tr, te):
    ytr = (tr[Y] < THRESH).astype(int)
    if kind == "naive":
        return (tr.lag1 < THRESH).astype(float).values, (te.lag1 < THRESH).astype(float).values
    cols = [c for c in XB if c != "year"] if kind == "mlp" else XB
    mdl = cls_model(kind).fit(tr[cols], ytr)
    return mdl.predict_proba(tr[cols])[:, 1], mdl.predict_proba(te[cols])[:, 1]


def cls_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, yhat)),
            "precision": float(precision_score(y, yhat, zero_division=0)), "recall": float(recall_score(y, yhat)),
            "f1": float(f1_score(y, yhat)), "n": int(len(y)), "positives": int(y.sum())}


def main():
    t0 = time.time()
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv").sort_values(["state_code", "year"])
    m = panel[panel.sample_rq1 == 1].copy()
    # previous year's outcome (only if the previous year is in the sample)
    prev = m[["state_code", "year", Y]].rename(columns={Y: "lag1"})
    prev["year"] += 1
    m = m.merge(prev, on=["state_code", "year"], how="left")
    m["lagging"] = (m[Y] < THRESH).astype(int)
    R = {"n_sample": int(len(m)), "n_with_lag": int(m.lag1.notna().sum()), "threshold": THRESH,
         "share_lagging": float(m.lagging.mean())}

    # ------------------------------------------------------------- rolling-origin regression
    reg_rows, preds_store, pi_rows = [], [], []
    for t in range(2019, 2025):
        tr_raw = m[(m.year <= t)]
        te_raw = m[(m.year == t + 1) & m.lag1.notna() & m.state_code.isin(tr_raw.state_code)]
        tr, te = add_features(tr_raw, te_raw)
        tr_l = tr[tr.lag1.notna()]                 # training rows usable by lag-based models
        for k in REG:
            trd = tr_l if (k == "naive" or k.endswith("_b")) else tr
            ptr, pte = fit_predict_reg(k, trd, te)
            reg_rows.append({"model": k, "test_year": t + 1, "n_train": len(trd), "n_test": len(te),
                             "train_rmse": rmse(trd[Y], ptr), "test_rmse": rmse(te[Y], pte),
                             "train_mae": float(mean_absolute_error(trd[Y], ptr)), "test_mae": float(mean_absolute_error(te[Y], pte))})
            preds_store.append(pd.DataFrame({"model": k, "state_code": te.state_code.values, "year": te.year.values,
                                             "actual": te[Y].values, "pred": pte}))
        # out-of-sample permutation importance for the set-B forest
        rf = reg_model("rf_b").fit(tr_l[XB], tr_l[Y])
        pi = permutation_importance(rf, te[XB], te[Y], n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
        pi_rows.append(dict(zip(XB, pi.importances_mean)))
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / "rolling_regression.csv", index=False)
    allp = pd.concat(preds_store)
    summ = reg.groupby("model").agg(train_rmse=("train_rmse", "mean"), test_rmse=("test_rmse", "mean"),
                                    train_mae=("train_mae", "mean"), test_mae=("test_mae", "mean"),
                                    n_test=("n_test", "sum")).reindex(list(REG))
    pooled_r2 = allp.groupby("model").apply(lambda d: r2_score(d.actual, d.pred)).reindex(list(REG))
    summ["test_r2_pooled"] = pooled_r2
    naive_rmse = summ.loc["naive", "test_rmse"]
    summ["skill_vs_naive"] = 1 - summ.test_rmse / naive_rmse
    # paired comparison of squared errors (best vs naive), by state-year, with a sign-flip permutation test
    best_reg = summ.test_rmse.idxmin()
    e_best = allp[allp.model == best_reg].set_index(["state_code", "year"])
    e_nai = allp[allp.model == "naive"].set_index(["state_code", "year"])
    d = ((e_nai.actual - e_nai.pred) ** 2 - (e_best.actual - e_best.pred) ** 2).dropna()
    rng = np.random.default_rng(SEED)
    flips = np.array([np.mean(d.values * rng.choice([-1, 1], len(d))) for _ in range(9999)])
    R["regression"] = {"summary": summ.round(4).to_dict(orient="index"), "best": best_reg,
                       "best_vs_naive": {"mean_sq_error_reduction": float(d.mean()),
                                         "p_permutation": float((np.abs(flips) >= abs(d.mean())).mean()),
                                         "share_state_years_better": float((d > 0).mean()), "n": int(len(d))},
                       "labels": REG}
    oos = pd.DataFrame(pi_rows)
    R["perm_importance_oos_rf_b"] = {k: {"mean": float(oos[k].mean()), "sd": float(oos[k].std())} for k in XB}
    print("rolling regression:\n", summ.round(2), flush=True)

    # ------------------------------------------------------------- hold-out split 2018-23 / 2024-25
    tr_raw = m[m.year <= 2023]
    te_raw = m[(m.year >= 2024) & m.lag1.notna() & m.state_code.isin(tr_raw.state_code)]
    tr, te = add_features(tr_raw, te_raw)
    tr_l = tr[tr.lag1.notna()]
    ho_rows, ho_preds = [], []
    for k in REG:
        trd = tr_l if (k == "naive" or k.endswith("_b")) else tr
        ptr, pte = fit_predict_reg(k, trd, te)
        ho_rows.append({"model": k, "n_train": len(trd), "n_test": len(te),
                        "train_rmse": rmse(trd[Y], ptr), "test_rmse": rmse(te[Y], pte),
                        "train_mae": float(mean_absolute_error(trd[Y], ptr)), "test_mae": float(mean_absolute_error(te[Y], pte)),
                        "train_r2": float(r2_score(trd[Y], ptr)), "test_r2": float(r2_score(te[Y], pte))})
        ho_preds.append(pd.DataFrame({"model": k, "state_code": te.state_code.values, "year": te.year.values,
                                      "actual": te[Y].values, "pred": pte}))
    ho = pd.DataFrame(ho_rows).set_index("model")
    pd.concat(ho_preds).to_csv(OUT / "holdout_predictions.csv", index=False)
    R["holdout"] = {"train_years": "2018-2023", "test_years": "2024-2025", "table": ho.round(4).to_dict(orient="index")}
    print("hold-out:\n", ho.round(2), flush=True)

    # ------------------------------------------------------------- classification: lagging (< 80%) state-years
    cls_rows, cls_store = [], []
    for t in range(2019, 2025):
        tr_raw = m[(m.year <= t)]
        te_raw = m[(m.year == t + 1) & m.lag1.notna() & m.state_code.isin(tr_raw.state_code)]
        tr, te = add_features(tr_raw, te_raw)
        tr_l = tr[tr.lag1.notna()]
        for k in CLS:
            ptr, pte = fit_predict_cls(k, tr_l, te)
            cls_rows.append({"model": k, "test_year": t + 1, **{f"train_{a}": b for a, b in cls_metrics((tr_l[Y] < THRESH).astype(int).values, ptr).items()}})
            cls_store.append(pd.DataFrame({"model": k, "state_code": te.state_code.values, "year": te.year.values,
                                           "y": (te[Y] < THRESH).astype(int).values, "p": pte}))
    cls_tr = pd.DataFrame(cls_rows)
    cp = pd.concat(cls_store)
    cp.to_csv(OUT / "rolling_classification.csv", index=False)
    cls_summ = {}
    for k in CLS:
        d_ = cp[cp.model == k]
        trm = cls_tr[cls_tr.model == k]
        cls_summ[k] = {"test": cls_metrics(d_.y.values, d_.p.values),
                       "train": {a: float(trm[f"train_{a}"].mean()) for a in ("auc", "accuracy", "precision", "recall", "f1")}}
    best_cls = max((k for k in CLS if k != "naive"), key=lambda k: cls_summ[k]["test"]["auc"])
    R["classification"] = {"summary": cls_summ, "best": best_cls, "labels": CLS,
                           "confusion_best": confusion_matrix(cp[cp.model == best_cls].y, (cp[cp.model == best_cls].p >= .5).astype(int)).tolist(),
                           "confusion_naive": confusion_matrix(cp[cp.model == "naive"].y, (cp[cp.model == "naive"].p >= .5).astype(int)).tolist()}
    # bootstrap CI (by state cluster) for AUC difference best vs logistic/naive
    states = cp.state_code.unique()
    diffs = []
    for _ in range(1000):
        pick = rng.choice(states, len(states), replace=True)
        bb = pd.concat([cp[cp.state_code == s] for s in pick])
        try:
            a1 = roc_auc_score(bb[bb.model == best_cls].y, bb[bb.model == best_cls].p)
            a0 = roc_auc_score(bb[bb.model == "naive"].y, bb[bb.model == "naive"].p)
            diffs.append(a1 - a0)
        except ValueError:
            continue
    R["classification"]["auc_diff_vs_naive_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    print("classification:", {k: round(v["test"]["auc"], 3) for k, v in cls_summ.items()}, "best", best_cls, flush=True)

    # ------------------------------------------------------------- figures
    lab = REG
    s = summ.copy()
    order = list(REG)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    yy = np.arange(len(order))
    cols = [GREY, "#bbbbbb", OCHRE, BLUE, GREEN, GREEN, PURPLE, GREEN, GREEN, PURPLE]
    ax.barh(yy, s.loc[order, "test_rmse"], color=cols, height=.6, label="Test RMSE (held-out year)")
    ax.plot(s.loc[order, "train_rmse"], yy, "k|", ms=12, mew=2, label="Training RMSE")
    for i, k in enumerate(order):
        ax.text(s.loc[k, "test_rmse"] + .4, i, f"{s.loc[k, 'test_rmse']:.1f}", va="center", fontsize=8)
    ax.axvline(naive_rmse, color=RED, ls="--", lw=1)
    ax.set_yticks(yy, [lab[k] for k in order], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("RMSE, percentage points (rolling origin, test years 2020-2025)")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "final_model_comparison.png", bbox_inches="tight", pad_inches=.05); plt.close(fig)

    hp = pd.read_csv(OUT / "holdout_predictions.csv")
    best_ho = ho.drop(index=["naive", "state_mean"]).test_rmse.idxmin()      # best learned model
    R["holdout"]["best_overall"] = str(ho.test_rmse.idxmin()); R["holdout"]["best_learned"] = str(best_ho)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1))
    for a, k, c in ((ax[0], "naive", GREY), (ax[1], best_ho, GREEN)):
        d_ = hp[hp.model == k]
        a.scatter(d_.pred, d_.actual, s=14, color=c, alpha=.85)
        a.plot([0, 100], [0, 100], color="k", lw=.8, ls="--")
        a.set_xlim(-2, 102); a.set_ylim(-2, 102)
        a.set_title(f"({'a' if a is ax[0] else 'b'}) {lab[k]}\nRMSE {ho.loc[k, 'test_rmse']:.1f}, R² {ho.loc[k, 'test_r2']:.2f}", loc="left", fontsize=9)
        a.set_xlabel("Predicted timely %"); a.set_ylabel("Actual timely %")
    fig.tight_layout(); fig.savefig(FIG / "final_pred_vs_actual.png", bbox_inches="tight", pad_inches=.05); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for k, c in zip(CLS, [GREY, BLUE, GREEN, OCHRE, PURPLE]):
        d_ = cp[cp.model == k]
        fpr, tpr, _ = roc_curve(d_.y, d_.p)
        ax[0].plot(fpr, tpr, color=c, lw=1.6 if k == best_cls else 1.1,
                   label=f"{CLS[k]} (AUC {cls_summ[k]['test']['auc']:.2f})", marker="o" if k == "naive" else None, ms=3)
    ax[0].plot([0, 1], [0, 1], color="k", lw=.7, ls=":")
    ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("True-positive rate")
    ax[0].set_title("(a) ROC curves, test years 2020-2025", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=6.8, loc="lower right")
    cm = np.array(R["classification"]["confusion_best"])
    ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax[1].set_xticks([0, 1], ["On track", "Lagging"]); ax[1].set_yticks([0, 1], ["On track", "Lagging"])
    ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")
    ax[1].set_title(f"(b) Confusion matrix, {CLS[best_cls].lower()}", loc="left", fontsize=9)
    ax[1].spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "final_roc_confusion.png", bbox_inches="tight", pad_inches=.05); plt.close(fig)

    imp = pd.Series({k: v["mean"] for k, v in R["perm_importance_oos_rf_b"].items()}).sort_values()
    nice = {"ln_upi_txn_pc": "ln UPI per person", "internet_per100": "Internet per 100", "literacy_pct": "Literacy %",
            "year": "Year", "state_hist": "State-history mean", "lag1": "Last year's timely %"}
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.barh([nice[k] for k in imp.index], imp.values, color=[BLUE if k in ("ln_upi_txn_pc", "internet_per100") else GREY for k in imp.index])
    ax.set_xlabel("Rise in test RMSE when the feature is shuffled (percentage points)")
    fig.tight_layout(); fig.savefig(FIG / "final_feature_importance.png", bbox_inches="tight", pad_inches=.05); plt.close(fig)

    R["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "predictive_results.json").write_text(json.dumps(R, indent=1, default=float))
    print("->", OUT / "predictive_results.json", f"({R['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
