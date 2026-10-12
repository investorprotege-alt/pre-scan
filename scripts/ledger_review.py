"""
ledger_review.py — Try to kill the gates.

The gates are a hypothesis: "ideas that pass all six outperform the ones I reject."
This script tests that hypothesis on the idea ledger instead of trusting it.

For every ledger row old enough, it measures the forward 5/10/20-session return
from the evaluation price, then compares:
  - passed vs rejected (mean, median, hit rate, bootstrap 90% CI of the gap)
  - each gate's kills vs everything that got past it (is that gate pulling its weight?)
  - QQQE scenario buckets (does the regime gate earn its keep?)

Small samples are flagged rather than celebrated. The same ticker shows up on
many days, so rows overlap and the bootstrap CI is optimistic — treat "edge" as
"not yet killed", never as proof. Nothing here changes the gates;
it tells you which ones to argue with.

Outputs: data/ledger_review.json
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
AEST = pytz.timezone("Australia/Sydney")
HORIZONS = [5, 10, 20]
MIN_SAMPLE = 30
GATES = ["regime", "structure", "leadership", "location", "contrary", "cost_risk"]


def load_ledger() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "idea_ledger.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def add_forward_returns(df: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        df[f"fwd_{h}"] = np.nan
    for i, r in df.iterrows():
        if r["ticker"] not in closes.columns:
            continue
        c = closes[r["ticker"]].dropna()
        # First session strictly after the evaluation date (briefings run pre-market)
        after = c[c.index.tz_localize(None) > r["date"]] if c.index.tz is not None else c[c.index > r["date"]]
        for h in HORIZONS:
            if len(after) >= h:
                df.at[i, f"fwd_{h}"] = (float(after.iloc[h - 1]) / r["price"] - 1) * 100
    return df


def bootstrap_gap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 7) -> tuple:
    rng = np.random.default_rng(seed)
    gaps = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return round(float(np.percentile(gaps, 5)), 2), round(float(np.percentile(gaps, 95)), 2)


def summarise(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {"n": 0}
    return {
        "n":        int(len(x)),
        "mean":     round(float(x.mean()), 2),
        "median":   round(float(x.median()), 2),
        "hit_rate": round(float((x > 0).mean() * 100), 1),
        "small_sample": bool(len(x) < MIN_SAMPLE),
    }


def compare(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    out = {"a": summarise(a), "b": summarise(b)}
    if len(a) >= 5 and len(b) >= 5:
        out["gap_mean"] = round(float(a.mean() - b.mean()), 2)
        out["gap_ci90"] = bootstrap_gap(a.values, b.values)
        lo, hi = out["gap_ci90"]
        out["verdict"] = ("edge" if lo > 0 else "anti-edge" if hi < 0 else "inconclusive")
        if len(a) < MIN_SAMPLE or len(b) < MIN_SAMPLE:
            out["verdict"] += " (small sample)"
    return out


def run() -> dict:
    df = load_ledger()
    result = {"generated_at": datetime.now(AEST).isoformat(), "rows": int(len(df))}
    if df.empty:
        result["note"] = "Ledger empty — run idea_gates.py for a few weeks first."
        _save(result)
        return result

    tickers = sorted(df["ticker"].unique())
    start = (df["date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    df = add_forward_returns(df, closes)

    result["by_horizon"] = {}
    for h in HORIZONS:
        col = f"fwd_{h}"
        block = {"passed_vs_rejected": compare(df.loc[df["passed"], col], df.loc[~df["passed"], col])}
        # Per-gate: what it killed vs what got past it (survivors of earlier gates)
        per_gate = {}
        for gi, g in enumerate(GATES):
            killed = df.loc[df["killed_by"] == g, col]
            got_past = df.loc[df["killed_by"].isin(GATES[gi + 1:]) | df["passed"], col]
            per_gate[g] = compare(got_past, killed)
        block["gate_kills_vs_got_past"] = per_gate
        block["by_qqqe_scenario"] = {str(s): summarise(df.loc[df["qqqe_scenario"] == s, col])
                                     for s in sorted(df["qqqe_scenario"].dropna().unique())}
        result["by_horizon"][str(h)] = block

    _save(result)
    pr = result["by_horizon"]["10"]["passed_vs_rejected"]
    print(f"Ledger rows: {len(df)}  |  10-day passed vs rejected: {pr.get('verdict', 'not enough data')}")
    return result


def _save(result: dict) -> None:
    with open(os.path.join(DATA_DIR, "ledger_review.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    run()
