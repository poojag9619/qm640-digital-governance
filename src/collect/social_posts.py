"""Collect citizen posts about government digital services (RQ3).

Two collectors share one output format (data/raw/social/posts_<platform>.csv):

  reddit  - official Reddit API through PRAW. Free for non-commercial research.
            Create a "script" app at https://www.reddit.com/prefs/apps and set
            REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.
  x       - X (Twitter) API v2 through tweepy. Needs a paid access tier; the
            full-archive endpoint (search_all_tweets) needs Pro or academic
            access, otherwise only the last 7 days are searchable.
            Set X_BEARER_TOKEN.

Run:  python -m src.collect.social_posts reddit
      python -m src.collect.social_posts x --max 5000

Limitations to record in the report: Reddit search returns at most ~1,000 results
per query and favours recent posts; geolocation is inferred from the subreddit or
from state names in the text (state_code is left blank when neither is clear).
Raw text stays in data/raw/social/ (git-ignored); only post ids and aggregated
scores are committed, in line with platform terms.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import RAW, states  # noqa: E402

OUT = RAW / "social"
KEYWORDS = ["DigiLocker", "UMANG app", "ABHA card", "ABHA ID", "Ayushman Bharat Digital",
            "CoWIN", "UPI", "Aadhaar update", "Aadhaar authentication", "e-district",
            "online ration card", "DBT not received", "seva sindhu", "aaple sarkar",
            "meeseva", "e-sevai", "jan aadhaar", "e-mitra", "serviceplus"]

# subreddits whose posts can be assigned to one state
STATE_SUBS = {"karnataka": "KA", "bangalore": "KA", "mumbai": "MH", "pune": "MH",
              "maharashtra": "MH", "delhi": "DL", "chennai": "TN", "tamilnadu": "TN",
              "kerala": "KL", "hyderabad": "TS", "telangana": "TS", "andhra_pradesh": "AP",
              "kolkata": "WB", "westbengal": "WB", "gujarat": "GJ", "ahmedabad": "GJ",
              "rajasthan": "RJ", "jaipur": "RJ", "punjab": "PB", "uttarpradesh": "UP",
              "lucknow": "UP", "bihar": "BR", "patna": "BR", "odisha": "OD", "assam": "AS",
              "goa": "GA", "chandigarh": "CH", "uttarakhand": "UK", "himachal": "HP",
              "madhyapradesh": "MP", "indore": "MP", "jharkhand": "JH", "chhattisgarh": "CG",
              "kashmir": "JK", "noida": "UP", "gurgaon": "HR", "haryana": "HR"}
NATIONAL_SUBS = ["india", "IndiaSpeaks", "indianews", "IndianTechSupport", "personalfinanceindia"]


def _state_from_text(text: str) -> str | None:
    hits = set()
    low = text.lower()
    for _, r in states().iterrows():
        for name in [r.state_name] + [a for a in r.aliases.split("|") if len(a) > 4]:
            if re.search(rf"\b{re.escape(name.lower())}\b", low):
                hits.add(r.state_code)
    return hits.pop() if len(hits) == 1 else None


def reddit(limit_per_query: int = 250) -> pd.DataFrame:
    import praw  # pip install praw
    api = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                      client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                      user_agent=os.environ.get("REDDIT_USER_AGENT", "qm640-capstone research"))
    rows = []
    for sub in list(STATE_SUBS) + NATIONAL_SUBS:
        for kw in KEYWORDS:
            try:
                for p in api.subreddit(sub).search(f'"{kw}"', sort="new", time_filter="all",
                                                   limit=limit_per_query):
                    text = f"{p.title}\n{p.selftext or ''}"
                    rows.append({"platform": "reddit", "post_id": p.id,
                                 "created_utc": pd.to_datetime(p.created_utc, unit="s"),
                                 "source": sub, "keyword": kw, "text": text,
                                 "score": p.score, "n_comments": p.num_comments,
                                 "url": f"https://www.reddit.com{p.permalink}",
                                 "state_code": STATE_SUBS.get(sub) or _state_from_text(text)})
            except Exception as e:  # private/banned subreddit, rate limit
                print(f"skip r/{sub} '{kw}': {e}")
    return pd.DataFrame(rows).drop_duplicates("post_id")


def x_posts(max_results: int = 2000, full_archive: bool = False) -> pd.DataFrame:
    import tweepy  # pip install tweepy
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    search = client.search_all_tweets if full_archive else client.search_recent_tweets
    query = "(" + " OR ".join(f'"{k}"' for k in KEYWORDS) + ") (lang:en OR lang:hi) -is:retweet"
    rows = []
    for page in tweepy.Paginator(search, query=query, max_results=100,
                                 tweet_fields=["created_at", "public_metrics", "lang"],
                                 expansions=["author_id"], user_fields=["location"],
                                 limit=max(1, max_results // 100)):
        users = {u.id: u for u in (page.includes or {}).get("users", [])}
        for t in page.data or []:
            loc = getattr(users.get(t.author_id), "location", "") or ""
            rows.append({"platform": "x", "post_id": t.id, "created_utc": t.created_at,
                         "source": "x", "keyword": "", "text": t.text,
                         "score": t.public_metrics.get("like_count"),
                         "n_comments": t.public_metrics.get("reply_count"),
                         "url": f"https://x.com/i/web/status/{t.id}",
                         "state_code": _state_from_text(loc) or _state_from_text(t.text)})
    return pd.DataFrame(rows).drop_duplicates("post_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["reddit", "x"])
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--full-archive", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = reddit() if a.platform == "reddit" else x_posts(a.max, a.full_archive)
    path = OUT / f"posts_{a.platform}.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} posts ({df.state_code.notna().mean():.0%} geolocated) -> {path}")
