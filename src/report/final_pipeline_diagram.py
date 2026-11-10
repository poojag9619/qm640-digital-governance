"""Architecture / workflow diagram for the final report -> reports/figures/final_pipeline.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

FIG = Path(__file__).resolve().parents[2] / "reports" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "savefig.dpi": 240})
C = {"data": "#dbe7f3", "proc": "#f3ead9", "model": "#dcefe1", "out": "#efe0ef"}
E = {"data": "#2f5d8a", "proc": "#b07a2a", "model": "#4b8b5a", "out": "#6a4c93"}

stages = [  # (x, y, kind, title, body)
    (0.1, 3.1, "data", "1. Data sources", "PhonePe Pulse (GitHub);\nMGNREGS reports;\nTRAI, RBI, NCP, NeSDA"),
    (2.95, 3.1, "proc", "2. Ingestion", "Git clone, Excel and\nPDF parsers; value status\n(ok / NR / NS)"),
    (5.8, 3.1, "proc", "3. Preprocessing", "Name harmonisation,\nmerge, range checks,\nmissing & outlier rules"),
    (8.65, 3.1, "data", "SQLite database", "states, state_year,\nvalue_status, dictionary;\nviews v_rq1 to v_rq4"),
    (8.65, 1.35, "proc", "4. EDA", "Distributions, heatmaps,\nbetween vs within,\ncorrelations"),
    (5.8, 1.35, "proc", "5. Features", "Per capita, logs, index,\nstate history, lag;\nscaling, one-hot effects"),
    (2.95, 1.35, "model", "6. Models", "Two-way FE (inference);\nnaive, OLS, RF, GBM,\nMLP, logit (prediction)"),
    (0.1, 1.35, "model", "7. Evaluation", "Rolling origin, hold-out,\nleave one state out;\nRMSE, AUC, bootstrap"),
    (0.1, -0.4, "out", "8. Deployment", "Scores to SQLite;\nstatic HTML dashboard;\nscheduled refresh"),
    (2.95, -0.4, "out", "Users", "MoRD, state MGNREGS\ncells, auditors,\nresearchers"),
]
W, H = 2.55, 1.2
fig, ax = plt.subplots(figsize=(7.4, 4.5))
pos = {}
for x, y, k, t, b in stages:
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08", fc=C[k], ec=E[k], lw=1.2))
    ax.text(x + W / 2, y + H - 0.2, t, ha="center", va="center", fontweight="bold", fontsize=8.6)
    ax.text(x + W / 2, y + 0.43, b, ha="center", va="center", fontsize=7.0, linespacing=1.3)
    pos[t] = (x, y)


def arrow(a, b, side_a, side_b, label=None, style="-|>"):
    def pt(t, s):
        x, y = pos[t]
        return {"r": (x + W, y + H / 2), "l": (x, y + H / 2), "b": (x + W / 2, y), "t": (x + W / 2, y + H)}[s]
    p1, p2 = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11, color="#333", lw=1.1))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + .05, (p1[1] + p2[1]) / 2 + .08, label, fontsize=6.6, color="#444")


arrow("1. Data sources", "2. Ingestion", "r", "l")
arrow("2. Ingestion", "3. Preprocessing", "r", "l")
arrow("3. Preprocessing", "SQLite database", "r", "l")
arrow("SQLite database", "4. EDA", "b", "t", "SQL / pandas")
arrow("4. EDA", "5. Features", "l", "r")
arrow("5. Features", "6. Models", "l", "r")
arrow("6. Models", "7. Evaluation", "l", "r")
arrow("7. Evaluation", "8. Deployment", "b", "t", "best models")
arrow("8. Deployment", "Users", "r", "l")
ax.legend(handles=[Patch(fc=C["data"], ec=E["data"], label="Data source / store"),
                   Patch(fc=C["proc"], ec=E["proc"], label="Processing step"),
                   Patch(fc=C["model"], ec=E["model"], label="Modelling and evaluation"),
                   Patch(fc=C["out"], ec=E["out"], label="Output and users")],
          loc="lower right", bbox_to_anchor=(1.0, -0.02), frameon=False, fontsize=7.3, ncol=1)
ax.text(11.15, 0.62, "Orchestrated by run_all.py and the\nscripts in src/ (Appendix A)", ha="right", fontsize=6.8, color="#555", style="italic")
ax.set_xlim(0, 11.3); ax.set_ylim(-0.55, 4.35); ax.axis("off")
fig.savefig(FIG / "final_pipeline.png", bbox_inches="tight", pad_inches=0.04)
print("->", FIG / "final_pipeline.png")
