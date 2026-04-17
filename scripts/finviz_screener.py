"""
finviz_screener.py — Jeff Sun's 4 momentum screeners + pullback-to-MA scan.

Replaces finvizfinance (blocked on GitHub Actions cloud IPs) with yfinance batch
downloads. Screens the S&P 500 universe directly.

Screeners:
  1. 1-Week Movers  >20%
  2. 1-Month Movers >30%
  3. 3-Month Movers >50%
  4. 6-Month Movers >100%
  + Daily Pullback-to-MA (near 10/21 EMA, above 50/200 SMA, within 20% of 52w high)

Outputs: data/finviz_screener.json
"""

import json
import os
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pytz

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardcoded fallback — top-100 S&P 500 names by market cap (updated 2025)
# ---------------------------------------------------------------------------
SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","TSLA","AVGO","JPM",
    "LLY","UNH","XOM","V","COST","MA","JNJ","PG","HD","ABBV","MRK","CVX",
    "KO","BAC","PEP","ORCL","WMT","MCD","CRM","ACN","TMO","ABT","CSCO","ADBE",
    "AMD","NFLX","LIN","DHR","TXN","NEE","PM","AMGN","ISRG","CAT","RTX","HON",
    "INTU","LOW","QCOM","GS","IBM","UNP","BKNG","SPGI","AXP","SYK","VRTX","GILD",
    "ELV","MDT","BLK","REGN","ADI","LRCX","MU","PANW","KLAC","AMAT","SNPS","CDNS",
    "MELI","INTC","PLD","CB","SBUX","CME","TJX","PYPL","TMUS","SO","CI","DUK",
    "AON","SCHW","ZTS","BMY","BSX","MMC","ITW","FI","ICE","NOC","MMM","WM",
    "APH","ETN","PH","EMR","GD","HCA","MCO","FCX","ORLY","ADP","CL","MO",
]


# ---------------------------------------------------------------------------
# Fetch S&P 500 tickers + company/sector info from Wikipedia
# ---------------------------------------------------------------------------

def get_sp500_universe() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ticker, company, sector, industry.
    Falls back to the hardcoded list with empty metadata if Wikipedia is unavailable.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; pre-market-bot/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_html(StringIO(resp.text))[0]
        df = df.rename(columns={
            "Symbol":          "ticker",
            "Security":        "company",
            "GICS Sector":     "sector",
            "GICS Sub-Industry": "industry",
        })
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        print(f"  Got {len(df)} S&P 500 entries from Wikipedia")
        return df[["ticker", "company", "sector", "industry"]].copy()
    except Exception as e:
        print(f"  WARNING: Wikipedia fetch failed ({e}) — using fallback list")
        return pd.DataFrame({
            "ticker":   SP500_FALLBACK,
            "company":  [""] * len(SP500_FALLBACK),
            "sector":   [""] * len(SP500_FALLBACK),
            "industry": [""] * len(SP500_FALLBACK),
        })


# ---------------------------------------------------------------------------
# Batch download + indicator computation
# ---------------------------------------------------------------------------

def download_universe(tickers: list) -> pd.DataFrame:
    """
    Batch-download ~7 months of daily OHLCV for the ticker list.
    Returns a flat DataFrame with columns:
      ticker, date, close, volume, ret_1w, ret_1m, ret_3m, ret_6m,
      sma50, sma200, ema10, ema21, high_52w, avg_vol_20
    """
    print(f"  Downloading price data for {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        period="8mo",        # 8 months = enough for 6M return + buffer
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        print("  ERROR: yfinance download returned empty DataFrame")
        return pd.DataFrame()

    # Extract OHLCV panels
    if isinstance(raw.columns, pd.MultiIndex):
        closes  = raw["Close"].dropna(how="all", axis=1)
        volumes = raw["Volume"].reindex(columns=closes.columns)
        highs   = raw["High"].reindex(columns=closes.columns)
    else:
        # Single-ticker edge case
        closes  = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
        highs   = raw[["High"]].rename(columns={"High": tickers[0]})

    # Need at least 130 rows for 6M return
    if len(closes) < 130:
        print(f"  WARNING: Only {len(closes)} rows — not enough for 6M return")
        return pd.DataFrame()

    results = []
    for ticker in closes.columns:
        c = closes[ticker].dropna()
        v = volumes[ticker].reindex(c.index).fillna(0)
        h = highs[ticker].reindex(c.index)

        if len(c) < 130:
            continue

        price = float(c.iloc[-1])
        if price < 5:          # skip penny stocks
            continue

        avg_vol = float(v.iloc[-20:].mean())
        if avg_vol < 100_000:  # skip illiquid
            continue

        def pct_ret(n):
            if len(c) <= n:
                return np.nan
            return round((float(c.iloc[-1]) / float(c.iloc[-1 - n]) - 1) * 100, 2)

        ret_1w  = pct_ret(5)
        ret_1m  = pct_ret(21)
        ret_3m  = pct_ret(63)
        ret_6m  = pct_ret(126)

        sma50   = float(c.rolling(50).mean().iloc[-1])
        sma200  = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else np.nan
        ema10   = float(c.ewm(span=10, adjust=False).mean().iloc[-1])
        ema21   = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        high_52w = float(h.iloc[-252:].max()) if len(h) >= 252 else float(h.max())

        results.append({
            "ticker":    ticker,
            "price":     price,
            "avg_vol_20": avg_vol,
            "ret_1w":    ret_1w,
            "ret_1m":    ret_1m,
            "ret_3m":    ret_3m,
            "ret_6m":    ret_6m,
            "sma50":     sma50,
            "sma200":    sma200,
            "ema10":     ema10,
            "ema21":     ema21,
            "high_52w":  high_52w,
        })

    print(f"  Computed indicators for {len(results)} tickers")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Screener filters
# ---------------------------------------------------------------------------

SCREENERS = [
    {
        "name":       "1-Week Movers >20%",
        "field":      "ret_1w",
        "threshold":  20.0,
    },
    {
        "name":       "1-Month Movers >30%",
        "field":      "ret_1m",
        "threshold":  30.0,
    },
    {
        "name":       "3-Month Movers >50%",
        "field":      "ret_3m",
        "threshold":  50.0,
    },
    {
        "name":       "6-Month Movers >100%",
        "field":      "ret_6m",
        "threshold":  100.0,
    },
]


def run_screeners(df: pd.DataFrame) -> dict:
    """Apply Jeff Sun's 4 momentum filters. Returns dict of screener_name -> [tickers]."""
    base = df[(df["price"] >= 10) & (df["avg_vol_20"] >= 300_000)]
    results = {}
    for s in SCREENERS:
        field = s["field"]
        thresh = s["threshold"]
        hits = base[base[field].notna() & (base[field] >= thresh)].copy()
        hits = hits.sort_values(field, ascending=False)
        results[s["name"]] = hits["ticker"].tolist()
        print(f"    {s['name']}: {len(hits)} hits")
    return results


def run_pullback_scan(df: pd.DataFrame) -> list:
    """
    Stocks near 10/21 EMA, above 50/200 SMA, within 20% of 52-week high,
    quarter return >20%, avg vol >500K.
    """
    mask = (
        (df["price"] >= 10) &
        (df["avg_vol_20"] >= 500_000) &
        (df["ret_3m"].notna() & (df["ret_3m"] >= 20)) &
        (df["price"] > df["sma50"]) &
        (df["sma200"].notna() & (df["price"] > df["sma200"])) &
        (df["price"] >= df["high_52w"] * 0.80) &
        (
            (((df["price"] - df["ema10"]).abs() / df["ema10"]) <= 0.03) |
            (((df["price"] - df["ema21"]).abs() / df["ema21"]) <= 0.03)
        )
    )
    hits = df[mask].copy()
    hits["near_ema"] = hits.apply(
        lambda r: "10 EMA" if abs(r.price - r.ema10) / r.ema10 <= 0.03 else "21 EMA",
        axis=1,
    )
    print(f"    Daily Pullback-to-MA: {len(hits)} hits")
    return hits["ticker"].tolist()


# ---------------------------------------------------------------------------
# Combine, deduplicate, enrich
# ---------------------------------------------------------------------------

def build_candidates(screener_hits: dict, pullback_hits: list,
                     df: pd.DataFrame, meta: pd.DataFrame,
                     top_sectors: set) -> list:
    """Merge all screener results, deduplicate, attach metadata."""
    all_hits = {**screener_hits, "Daily Pullback-to-MA": pullback_hits}

    seen: dict[str, dict] = {}
    for screener_name, tickers in all_hits.items():
        for ticker in tickers:
            if ticker not in seen:
                row = df[df["ticker"] == ticker]
                m   = meta[meta["ticker"] == ticker]
                company  = m["company"].iloc[0]  if len(m) else ""
                sector   = m["sector"].iloc[0]   if len(m) else ""
                industry = m["industry"].iloc[0] if len(m) else ""
                price    = float(row["price"].iloc[0]) if len(row) else None
                ret_1w   = row["ret_1w"].iloc[0]  if len(row) else None
                ret_1m   = row["ret_1m"].iloc[0]  if len(row) else None
                ret_3m   = row["ret_3m"].iloc[0]  if len(row) else None
                ret_6m   = row["ret_6m"].iloc[0]  if len(row) else None
                avg_vol  = row["avg_vol_20"].iloc[0] if len(row) else None
                seen[ticker] = {
                    "ticker":   ticker,
                    "company":  str(company),
                    "sector":   str(sector),
                    "industry": str(industry),
                    "price":    round(price, 2) if price else None,
                    "ret_1w":   float(ret_1w)  if ret_1w is not None else None,
                    "ret_1m":   float(ret_1m)  if ret_1m is not None else None,
                    "ret_3m":   float(ret_3m)  if ret_3m is not None else None,
                    "ret_6m":   float(ret_6m)  if ret_6m is not None else None,
                    "avg_vol":  int(avg_vol)   if avg_vol else None,
                    "screeners": [screener_name],
                }
            else:
                if screener_name not in seen[ticker]["screeners"]:
                    seen[ticker]["screeners"].append(screener_name)

    combined = list(seen.values())
    for s in combined:
        s["screener_count"]  = len(s["screeners"])
        s["high_confluence"] = s["screener_count"] >= 2
        s["in_top_sector"]   = any(
            ts.lower() in s["sector"].lower() for ts in top_sectors
        ) if top_sectors else False

    # Sort: high confluence first, then by 1W return descending
    combined.sort(key=lambda x: (-x["screener_count"],
                                  -(x["ret_1w"] or 0)))
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("Loading S&P 500 universe...")
    meta = get_sp500_universe()
    tickers = meta["ticker"].tolist()

    print(f"Downloading market data for {len(tickers)} tickers...")
    df = download_universe(tickers)
    if df.empty:
        print("ERROR: No data downloaded — writing empty output")
        output = {
            "generated_at": datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
            "screener_breakdown": {},
            "combined_candidates": [],
            "high_confluence_count": 0,
            "in_top_sector_count": 0,
            "error": "yfinance download failed",
        }
        with open(os.path.join(DATA_DIR, "finviz_screener.json"), "w") as f:
            json.dump(output, f, indent=2)
        return output

    print("Running Jeff Sun's 4 screeners...")
    screener_hits = run_screeners(df)

    print("Running pullback-to-MA scan...")
    pullback_hits = run_pullback_scan(df)

    # Load top sectors for cross-referencing
    top_sectors: set = set()
    scan_path = os.path.join(DATA_DIR, "sector_scan.json")
    if os.path.exists(scan_path):
        with open(scan_path) as f:
            scan = json.load(f)
        top_sectors = {s["name"] for s in scan.get("us_sectors", []) if s.get("top")}

    print("Building combined candidate list...")
    all_hits = {**screener_hits, "Daily Pullback-to-MA": pullback_hits}
    combined = build_candidates(all_hits, pullback_hits, df, meta, top_sectors)

    output = {
        "generated_at": datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
        "screener_breakdown": {k: len(v) for k, v in all_hits.items()},
        "combined_candidates": combined[:60],
        "high_confluence_count": sum(1 for s in combined if s.get("high_confluence")),
        "in_top_sector_count":   sum(1 for s in combined if s.get("in_top_sector")),
        "universe_size": len(df),
        "source": "yfinance (S&P 500)",
    }

    out_path = os.path.join(DATA_DIR, "finviz_screener.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved -> {out_path}")
    print(f"  Universe: {len(df)} tickers with sufficient data")
    for name, hits in screener_hits.items():
        print(f"  {name}: {len(hits)} hits")
    print(f"  Pullback scan: {len(pullback_hits)} hits")
    print(f"  Total candidates: {len(combined)}")
    print(f"  High confluence (2+ screeners): {output['high_confluence_count']}")
    return output


if __name__ == "__main__":
    run()
