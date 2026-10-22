"""Reproduce every sample-size figure in the synopsis (section "Sample Size Calculation").

    python -m src.models.power_analysis

Regression power follows G*Power's "Linear multiple regression: fixed model, R2 increase"
(Faul et al., 2009): lambda = f2 * N, numerator df = u (terms tested), denominator
df = N - p - 1 (p = all terms in the model, including fixed-effect dummies).
Clustering is handled with Kish's design effect, Deff = 1 + (m - 1) * rho.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

ALPHA, POWER = 0.05, 0.80
S, T = 30, 8                    # planning assumption: 30 units with MGNREGS data x 8 years
FE = (S - 1) + (T - 1)          # 29 state + 7 year dummies = 36
RHO = 0.10                      # assumed residual intra-state correlation after fixed effects


def reg_power(n: int, u: int, p: int, f2: float) -> tuple[float, int, float, float]:
    df2 = n - p - 1
    crit = stats.f.ppf(1 - ALPHA, u, df2)
    lam = f2 * n
    return 1 - stats.ncf.cdf(crit, u, df2, lam), df2, crit, lam


def reg_n(u: int, p: int, f2: float) -> int:
    n = p + 2
    while reg_power(n, u, p, f2)[0] < POWER:
        n += 1
    return n


def corr_n(r: float) -> float:
    z = stats.norm.ppf
    return ((z(1 - ALPHA / 2) + z(POWER)) / np.arctanh(r)) ** 2 + 3


def deff(m: int, rho: float = RHO) -> float:
    return 1 + (m - 1) * rho


def main() -> None:
    rows = []
    models = [  # label, terms tested u, other terms, f2, description
        ("RQ1", 4, 1 + FE, 0.10, "joint test of 4 adoption measures; control ln NSDP; state + year FE"),
        ("RQ2", 2, 2 + FE, 0.08, "2 interaction terms; adoption index, ln NSDP; state + year FE"),
        ("RQ4", 1, 1 + FE, 0.25 ** 2 / (1 - 0.25 ** 2), "adoption index (partial r = .25); ln NSDP; state + year FE"),
    ]
    for lab, u, k, f2, desc in models:
        p = u + k
        n = reg_n(u, p, f2)
        pw, df2, crit, lam = reg_power(n, u, p, f2)
        nom = int(np.ceil(n * deff(T)))
        rows.append((lab, n, nom))
        print(f"{lab}: {desc}\n   f2={f2:.4f}, u={u}, p={p}: N_srs={n} (df2={df2}, F_crit={crit:.3f}, "
              f"lambda={lam:.2f}, power={pw:.4f}); Deff={deff(T):.2f} -> nominal N={nom} state-years")
    n3 = corr_n(0.30)
    nom3 = int(np.ceil(np.ceil(n3) * deff(3)))
    print(f"RQ3: r=.30 -> n={n3:.2f} -> {int(np.ceil(n3))}; Deff (m=3)={deff(3):.2f} -> nominal N={nom3} state-years")
    p_acc, e = 0.70, 0.04
    n_lab = stats.norm.ppf(0.975) ** 2 * p_acc * (1 - p_acc) / e ** 2
    print(f"Classifier validation: p={p_acc}, e={e} -> n={n_lab:.1f} -> {int(np.ceil(n_lab))} labelled posts")
    panel_max = max(r[2] for r in rows)
    print(f"\nFINAL (state-year panel, RQ1/RQ2/RQ4): N = max{[r[2] for r in rows]} = {panel_max}; "
          f"planned analytic sample {S}x{T} = {S * T}")
    rho_max = (S * T / rows[0][1] - 1) / (T - 1)
    print(f"RQ1 keeps power >= .80 at N={S * T} while residual rho <= {rho_max:.3f}")
    print("\nSensitivity of nominal N to rho:")
    for rho in (0.05, 0.10, 0.20):
        print(f"   rho={rho:.2f}: Deff={deff(T, rho):.2f}; " +
              ", ".join(f"{lab} {int(np.ceil(n * deff(T, rho)))}" for lab, n, _ in rows))


if __name__ == "__main__":
    main()
