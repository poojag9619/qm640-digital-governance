"""Load the cleaned project data into a single SQLite database.

    python -m src.db.build_db            # -> data/processed/qm640.sqlite

Tables
  states            one row per state/UT (config/states.csv)
  state_year        the analysis panel: one row per state x year (data/processed/analysis_panel.csv)
  value_status      value status of every source cell: ok / NR (not reported) / NS (not applicable)
  data_dictionary   description, unit, source and role of every state_year column
  rolling_predictions, model_metrics   written by src/deploy/build_dashboard.py
Views
  v_rq1, v_rq2, v_rq4  the analysis samples of each research question
  v_latest_status      the latest year of the outcome for each state
SQLite is used because the data are small (288 rows), it needs no server, and the single
file can be versioned and opened from Python, R, Excel (ODBC) or a BI tool.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import CONFIG, INTERIM, PROCESSED  # noqa: E402

DB = PROCESSED / "qm640.sqlite"

VIEWS = {
    "v_rq1": "SELECT state_code, year, nrega_timely_pct, ln_upi_txn_pc, internet_per100 FROM state_year WHERE sample_rq1 = 1",
    "v_rq2": "SELECT state_code, year, nrega_timely_pct, adoption_index, literacy_pct, ln_nsdp_base FROM state_year WHERE sample_rq2 = 1",
    "v_rq4": "SELECT state_code, year, nrega_women_pd_pct, adoption_index FROM state_year WHERE sample_rq4 = 1",
    "v_latest_status": """SELECT s.state_code, st.state_name, s.year, s.nrega_timely_pct
                          FROM state_year s JOIN states st USING (state_code)
                          WHERE s.nrega_timely_pct IS NOT NULL
                            AND s.year = (SELECT MAX(year) FROM state_year x
                                          WHERE x.state_code = s.state_code AND x.nrega_timely_pct IS NOT NULL)""",
}


def connect() -> sqlite3.Connection:
    """Open the project database (read access for analysis scripts and the dashboard)."""
    return sqlite3.connect(DB)


def main():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    states = pd.read_csv(CONFIG / "states.csv")[["state_code", "state_name", "type", "region"]]
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv")
    status = pd.concat([pd.read_csv(f).assign(source=f.stem.replace("_status_", ""))
                        for f in sorted(INTERIM.glob("_status_*.csv"))], ignore_index=True)
    dd = pd.read_csv(PROCESSED / "data_dictionary.csv")
    con.execute("CREATE TABLE states (state_code TEXT PRIMARY KEY, state_name TEXT NOT NULL, type TEXT, region TEXT)")
    states.to_sql("states", con, if_exists="append", index=False)
    panel.to_sql("state_year", con, index=False)
    con.execute("CREATE UNIQUE INDEX ix_state_year ON state_year (state_code, year)")
    status.to_sql("value_status", con, index=False)
    dd.to_sql("data_dictionary", con, index=False)
    for name, sql in VIEWS.items():
        con.execute(f"CREATE VIEW {name} AS {sql}")
    con.commit()
    for t in ["states", "state_year", "value_status", "data_dictionary", "v_rq1", "v_rq2", "v_rq4", "v_latest_status"]:
        print(f"{t:18s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>6} rows")
    con.close()
    print("->", DB)


if __name__ == "__main__":
    main()
