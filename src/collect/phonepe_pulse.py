"""Download and parse PhonePe Pulse open data (UPI adoption proxy).

Source : https://github.com/PhonePe/pulse  (licence: CDLA-Permissive-2.0)
Output : data/interim/phonepe_state_quarter.csv
         data/interim/phonepe_state_year.csv

Variables
  upi_txn            transactions in the period (sum of P2P, Retail/Business, Utility)
  upi_txn_p2p, upi_txn_retail, upi_txn_utility   category split
  upi_users          registered users, cumulative at period end
  upi_merchants      registered merchants, cumulative at period end

The repository is cloned with git (shallow). Re-running pulls the latest commit;
the commit hash used is written to data/raw/phonepe_pulse_COMMIT.txt so results
can be reproduced. Pulse restated its whole history in its latest release, so do
not mix these figures with numbers downloaded from older releases.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import INTERIM, RAW, YEARS, states  # noqa: E402

REPO = "https://github.com/PhonePe/pulse.git"
DEST = RAW / "phonepe_pulse"
BASE = DEST / "data" / "aggregated"


def fetch() -> str:
    if (DEST / ".git").exists():
        subprocess.run(["git", "-C", str(DEST), "pull", "--ff-only", "-q"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-q", REPO, str(DEST)], check=True)
    sha = subprocess.run(["git", "-C", str(DEST), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    (RAW / "phonepe_pulse_COMMIT.txt").write_text(sha + "\n")
    return sha


def _files(kind: str):
    """Yield (slug, year, quarter, path) for every state-level JSON of one kind."""
    root = BASE / kind / "country" / "india" / "state"
    for p in sorted(root.glob("*/*/*.json")):
        yield p.parts[-3], int(p.parts[-2]), int(p.stem), p


def parse() -> pd.DataFrame:
    slug_to_code = dict(zip(states().phonepe_slug, states().state_code))
    recs: dict[tuple, dict] = {}

    def rec(slug, y, q):
        code = slug_to_code.get(slug)
        if code is None:
            raise ValueError(f"Unknown PhonePe state folder '{slug}' - add it to config/states.csv")
        return recs.setdefault((code, y, q), {"state_code": code, "year": y, "quarter": q})

    for slug, y, q, p in _files("transaction"):
        r = rec(slug, y, q)
        total = 0
        for cat in json.loads(p.read_text())["data"]["transactionData"] or []:
            n = sum(int(i.get("count") or 0) for i in cat["paymentInstruments"])
            key = "upi_txn_" + {"p2p": "p2p", "retail": "retail", "business": "retail",
                                "utility": "utility", "utilities": "utility"}.get(
                                    cat["name"].lower(), "other")
            r[key] = r.get(key, 0) + n
            total += n
        r["upi_txn"] = total

    for kind, col in (("user", "upi_users"), ("merchant", "upi_merchants")):
        for slug, y, q, p in _files(kind):
            agg = json.loads(p.read_text())["data"].get("aggregated") or {}
            rec(slug, y, q)[col] = agg.get("registeredCount")

    df = pd.DataFrame(recs.values()).sort_values(["state_code", "year", "quarter"])
    return df.reset_index(drop=True)


def to_year(q: pd.DataFrame) -> pd.DataFrame:
    """Flows are summed over the year; stocks (users, merchants) take the Q4 value."""
    q = q[q.year.isin(YEARS)]
    flows = [c for c in q.columns if c.startswith("upi_txn")]
    n_q = q.groupby(["state_code", "year"]).quarter.nunique().rename("n_quarters")
    y = q.groupby(["state_code", "year"])[flows].sum(min_count=1)
    last = q.sort_values("quarter").groupby(["state_code", "year"]).last()
    y = y.join(last[["upi_users", "upi_merchants"]]).join(n_q).reset_index()
    # a stock taken from an incomplete year is not comparable
    y.loc[y.n_quarters < 4, flows + ["upi_users", "upi_merchants"]] = pd.NA
    return y


def main():
    sha = fetch()
    q = parse()
    y = to_year(q)
    q.to_csv(INTERIM / "phonepe_state_quarter.csv", index=False)
    y.to_csv(INTERIM / "phonepe_state_year.csv", index=False)
    print(f"PhonePe Pulse @ {sha[:10]}: {q.state_code.nunique()} states, "
          f"{len(q)} state-quarters ({q.year.min()}Q1-{q.year.max()}Q{q[q.year == q.year.max()].quarter.max()}), "
          f"{len(y)} state-years")


if __name__ == "__main__":
    main()
