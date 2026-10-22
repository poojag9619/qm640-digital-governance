"""Render an excerpt of the analysis file as an image (dataset preview figure)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "data/processed/analysis_panel.csv")
cols = ["state_code", "year", "population", "upi_txn", "upi_txn_pc", "internet_per100", "nsdp_pc_real",
        "literacy_pct", "nrega_timely_pct", "nrega_women_pd_pct", "sample_rq1"]
x = d[d.year.isin([2021, 2025]) & d.state_code.isin(["BR", "KA", "MN", "WB"])][cols].sort_values(["state_code", "year"])
fmt = lambda v, c: "NaN" if pd.isna(v) else (f"{v:,.0f}" if c in ("population", "upi_txn") else (f"{v:,.2f}" if isinstance(v, float) else str(v)))
fig, ax = plt.subplots(figsize=(11, 2.4), dpi=200); ax.axis("off")
t = ax.table(cellText=[[fmt(r[c], c) for c in cols] for _, r in x.iterrows()], colLabels=cols, loc="center", cellLoc="right")
t.auto_set_font_size(False); t.set_fontsize(6); t.scale(1, 1.4)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor("#cccccc")
    if i == 0: c.set_facecolor("#e8edf3"); c.set_text_props(weight="bold")
ax.set_title(f"data/processed/analysis_panel.csv  ({d.shape[0]} rows x {d.shape[1]} columns; excerpt)", fontsize=7, loc="left", family="monospace", pad=14)
fig.tight_layout(); fig.savefig(ROOT / "reports/figures/interim_dataset_preview.png")
