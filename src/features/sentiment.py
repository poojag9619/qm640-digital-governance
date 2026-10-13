"""RQ3: score, validate and aggregate citizen sentiment about digital government services.

Steps (run in order):
  python -m src.features.sentiment score      # VADER labels -> data/raw/social/scored_posts.csv
  python -m src.features.sentiment sample     # 505-post labelling sheet (+60 double-coded)
      ... two coders fill coder1_label / coder2_label (positive, neutral, negative) ...
  python -m src.features.sentiment evaluate   # accuracy, macro-F1, kappa -> reports/sentiment_validation.txt
  python -m src.features.sentiment aggregate  # -> data/processed/sentiment_state_year.csv
  python -m src.features.sentiment purge      # end of project: delete raw post text

Design choices
  * Window 2023-2025 (posts older than that are hard to retrieve through the Reddit API).
  * English/romanised posts only (ASCII-letter share >= 0.9); Devanagari posts are counted and
    reported but not scored in the core design.
  * VADER compound >= 0.05 positive, <= -0.05 negative, otherwise neutral (Hutto & Gilbert, 2014).
  * A state-year enters RQ3 only with >= 30 scored posts.
  * Ethics: no usernames are collected; text stays in data/raw/social/ (git-ignored) and is
    deleted by `purge`; only post ids and state-year aggregates are shared.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import PROCESSED, RAW, REPORTS  # noqa: E402

SOCIAL = RAW / "social"
WINDOW = range(2023, 2026)
N_LABEL, N_OVERLAP, SEED = 505, 60, 640
LABELS = ["negative", "neutral", "positive"]


def _posts() -> pd.DataFrame:
    files = sorted(SOCIAL.glob("posts_*.csv"))
    if not files:
        raise SystemExit("No posts found: run `python -m src.collect.social_posts reddit` first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("post_id")
    df["year"] = pd.to_datetime(df.created_utc, utc=True, errors="coerce").dt.year
    return df


def _ascii_share(t: str) -> float:
    letters = [c for c in str(t) if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def score() -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    df = _posts()
    n0 = len(df)
    df = df[df.year.isin(WINDOW)]
    df = df[df.text.astype(str).str.split().str.len() >= 5]
    df["english"] = df.text.map(_ascii_share) >= 0.9
    log = [f"posts collected: {n0}", f"in 2023-2025 with >= 5 words: {len(df)}",
           f"non-English (not scored): {int((~df.english).sum())}"]
    df = df[df.english].copy()
    sia = SentimentIntensityAnalyzer()
    df["compound"] = df.text.map(lambda t: sia.polarity_scores(str(t))["compound"])
    df["model_label"] = np.select([df.compound >= 0.05, df.compound <= -0.05],
                                  ["positive", "negative"], "neutral")
    df.to_csv(SOCIAL / "scored_posts.csv", index=False)
    log.append(f"scored: {len(df)} ({df.state_code.notna().mean():.0%} geolocated)")
    print("\n".join(log))
    return df


def sample() -> None:
    df = pd.read_csv(SOCIAL / "scored_posts.csv")
    n = min(N_LABEL, len(df))
    # stratify by model label so all three classes are represented
    s = (df.groupby("model_label", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), int(np.ceil(n / 3))), random_state=SEED)))
    s = s.sample(frac=1, random_state=SEED).head(n)
    s["double_coded"] = 0
    s.iloc[:N_OVERLAP, s.columns.get_loc("double_coded")] = 1
    out = s[["post_id", "text", "double_coded"]].assign(coder1_label="", coder2_label="")
    out.to_csv(SOCIAL / "labelling_sample.csv", index=False)   # model label hidden from coders
    print(f"{len(out)} posts to label ({N_OVERLAP} by both coders) -> {SOCIAL / 'labelling_sample.csv'}")


def _kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in LABELS)
    return (po - pe) / (1 - pe)


def evaluate() -> None:
    lab = pd.read_csv(SOCIAL / "labelling_sample.csv")
    pred = pd.read_csv(SOCIAL / "scored_posts.csv")[["post_id", "model_label"]]
    lab = lab.merge(pred, on="post_id")
    gold = lab.dropna(subset=["coder1_label"])
    y, yhat = gold.coder1_label.str.lower(), gold.model_label
    f1s = []
    lines = [f"labelled posts: {len(gold)}", f"accuracy: {(y == yhat).mean():.3f}"]
    for k in LABELS:
        tp = ((yhat == k) & (y == k)).sum(); fp = ((yhat == k) & (y != k)).sum(); fn = ((yhat != k) & (y == k)).sum()
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        f1s.append(f1)
        lines.append(f"{k:>8}: precision {p:.3f} recall {r:.3f} F1 {f1:.3f}")
    lines.append(f"macro-F1: {np.mean(f1s):.3f}")
    lines.append(f"kappa (model vs coder 1): {_kappa(y, yhat):.3f}")
    dc = gold[(gold.double_coded == 1) & gold.coder2_label.notna()]
    if len(dc):
        lines.append(f"inter-coder kappa (n={len(dc)}): {_kappa(dc.coder1_label.str.lower(), dc.coder2_label.str.lower()):.3f}")
    ok = np.mean(f1s) >= 0.70 and _kappa(y, yhat) >= 0.60
    lines.append("DECISION: " + ("accept classifier for aggregation" if ok else
                                 "below threshold (macro-F1 0.70 / kappa 0.60): report RQ3 as exploratory"))
    (REPORTS / "sentiment_validation.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def aggregate(min_posts: int = 30) -> None:
    df = pd.read_csv(SOCIAL / "scored_posts.csv").dropna(subset=["state_code"])
    g = df.groupby(["state_code", "year"])
    out = pd.DataFrame({
        "n_posts": g.size(),
        "share_positive": g.model_label.apply(lambda s: (s == "positive").mean()),
        "share_negative": g.model_label.apply(lambda s: (s == "negative").mean()),
    }).reset_index()
    out["net_sentiment"] = out.share_positive - out.share_negative
    out.to_csv(PROCESSED / "sentiment_state_year.csv", index=False)
    k = int((out.n_posts >= min_posts).sum())
    print(f"{len(out)} state-years with posts; {k} have >= {min_posts} posts "
          f"(RQ3 needs >= 102 for the inferential test)")


def purge() -> None:
    for f in SOCIAL.glob("*.csv"):
        d = pd.read_csv(f)
        if "text" in d:
            d.drop(columns="text").to_csv(f, index=False)
            print(f"text removed from {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "sample", "evaluate", "aggregate", "purge"])
    step = ap.parse_args().step
    {"score": score, "sample": sample, "evaluate": evaluate, "aggregate": aggregate, "purge": purge}[step]()
