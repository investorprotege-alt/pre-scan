"""
rs_ranker.py — IBD-style Relative Strength ranking for US and ASX universes.

Formula:
  RS = (0.40 × 3M return) + (0.20 × 6M return) + (0.20 × 9M return) + (0.20 × 12M return)

Percentile-ranks the universe so each stock gets an RS score 1-99.
Run weekly (Sunday). Outputs: data/rs_rankings.json
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import pytz

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.json")
os.makedirs(DATA_DIR, exist_ok=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_sp500_tickers() -> list:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"  WARNING: S&P 500 list fetch failed: {e}")
        return []


def get_asx200_tickers() -> list:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/S%26P/ASX_200")
        for t in tables:
            for col in t.columns:
                if "ticker" in str(col).lower() or "code" in str(col).lower():
                    raw = t[col].dropna().tolist()
                    return [f"{str(r).strip()}.AX" for r in raw if len(str(r).strip()) <= 4]
    except Exception as e:
        print(f"  WARNING: ASX 200 list fetch failed: {e}")
    return []


def compute_rs_score(ticker: str, closes_df: pd.DataFrame) -> dict | None:
    """Compute IBD-style RS score for a single ticker using pre-downloaded closes."""
    if ticker not in closes_df.columns:
        return None

    prices = closes_df[ticker].dropna()
    if len(prices) < 252:
        return None

    price = float(prices.iloc[-1])

    def period_return(days: int) -> float | None:
        if len(prices) < days + 1:
            return None
        return (prices.iloc[-1] / prices.iloc[-days - 1] - 1) * 100

    r3m  = period_return(63)
    r6m  = period_return(126)
    r9m  = period_return(189)
    r12m = period_return(252)

    if any(v is None for v in [r3m, r6m, r9m, r12m]):
        return None

    rs_raw = (0.40 * r3m) + (0.20 * r6m) + (0.20 * r9m) + (0.20 * r12m)

    return {
        "ticker":  ticker,
        "price":   round(price, 4),
        "rs_raw":  round(rs_raw, 2),
        "3m_roc":  round(r3m, 2),
        "6m_roc":  round(r6m, 2),
        "9m_roc":  round(r9m, 2),
        "12m_roc": round(r12m, 2),
    }


def rank_universe(tickers: list, label: str) -> list:
    """Download closes for the universe and compute RS percentile rankings."""
    if not tickers:
        return []

    print(f"  Downloading {label} universe ({len(tickers)} tickers)...")

    # Batch download in chunks of 100
    all_closes = []
    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"    Chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1}...", end="", flush=True)
        try:
            raw = yf.download(chunk, period="14mo", auto_adjust=True, progress=False)
            if not raw.empty:
                closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
                all_closes.append(closes)
            print(" done")
        except Exception as e:
            print(f" ERROR: {e}")

    if not all_closes:
        return []

    closes_df = pd.concat(all_closes, axis=1)
    closes_df = closes_df.loc[:, ~closes_df.columns.duplicated()]

    # Compute raw RS scores
    rows = []
    for ticker in tickers:
        result = compute_rs_score(ticker, closes_df)
        if result:
            rows.append(result)

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Percentile rank rs_raw -> RS score 1-99
    df["rs_score"] = df["rs_raw"].rank(pct=True).mul(99).round(0).astype(int)
    df = df.sort_values("rs_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["top_quartile"] = df["rs_score"] >= 75

    return df.to_dict("records")


def run():
    print("Loading config...")
    cfg = load_config()

    print("Fetching S&P 500 universe...")
    sp500 = get_sp500_tickers()

    print("Fetching ASX 200 universe...")
    asx200 = get_asx200_tickers()

    # Always include watchlist leaders even if not in the broad indices
    extra_us = [item["ticker"] for item in cfg.get("indices", []) if "." not in item["ticker"]]
    extra_asx = [
        item["ticker"]
        for group in cfg["asx_leaders"].values()
        for item in group
    ]

    us_universe  = list(dict.fromkeys(sp500 + extra_us))   # dedup, preserve order
    asx_universe = list(dict.fromkeys(asx200 + extra_asx))

    print(f"\nUS universe: {len(us_universe)} tickers")
    print(f"ASX universe: {len(asx_universe)} tickers")

    us_ranked  = rank_universe(us_universe, "US")
    asx_ranked = rank_universe(asx_universe, "ASX")

    # Enrich with names from config where available
    asx_name_map = {
        item["ticker"]: item["name"]
        for group in cfg["asx_leaders"].values()
        for item in group
    }
    for row in asx_ranked:
        if row["ticker"] in asx_name_map:
            row["name"] = asx_name_map[row["ticker"]]

    output = {
        "generated_at": datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
        "us_rankings": us_ranked[:200],   # top 200
        "asx_rankings": asx_ranked[:100], # top 100
        "us_top10":  [r for r in us_ranked if r["rank"] <= 10],
        "asx_top10": [r for r in asx_ranked if r["rank"] <= 10],
        "us_top_quartile_count":  sum(1 for r in us_ranked if r.get("top_quartile")),
        "asx_top_quartile_count": sum(1 for r in asx_ranked if r.get("top_quartile")),
    }

    out_path = os.path.join(DATA_DIR, "rs_rankings.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"  US ranked: {len(us_ranked)}  |  ASX ranked: {len(asx_ranked)}")
    if us_ranked:
        top3 = [f"{r['ticker']} ({r['rs_score']})" for r in us_ranked[:3]]
        print(f"  US top 3: {', '.join(top3)}")
    return output


if __name__ == "__main__":
    run()
