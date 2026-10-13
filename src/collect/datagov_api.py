"""Download any dataset from India's Open Government Data platform (data.gov.in).

Many state-wise tables (literacy, PMJDY, MGNREGA, PDS, telecom) are published on
data.gov.in as "resources" with a free API.

  1. Register at https://www.data.gov.in/ and copy your API key (My Account).
  2. Open the dataset page; the resource id is the UUID in the API tab / URL.
  3. Run:
       export DATAGOV_API_KEY=your_key
       python -m src.collect.datagov_api <resource-id> --out data/raw/rbi/literacy_datagov.csv

The output is the table exactly as published. Reshape it into the matching
template (templates/<source>.csv) before running `manual_sources load`, or use it
as the evidence file for the values you enter.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.data.gov.in/resource/{rid}"


def download(resource_id: str, key: str, page: int = 1000) -> pd.DataFrame:
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        r = requests.get(API.format(rid=resource_id),
                         params={"api-key": key, "format": "json", "limit": page, "offset": offset},
                         timeout=60)
        r.raise_for_status()
        js = r.json()
        recs = js.get("records", [])
        total = int(js.get("total", len(recs)))
        rows += recs
        if not recs:
            break
        offset += len(recs)
        time.sleep(0.5)  # be polite to the API
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("resource_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=os.environ.get("DATAGOV_API_KEY"))
    a = ap.parse_args()
    if not a.key:
        sys.exit("Set DATAGOV_API_KEY or pass --key")
    df = download(a.resource_id, a.key)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows -> {a.out}")
