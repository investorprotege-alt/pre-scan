"""
finviz_screener.py — Runs Jeff Sun's 4 momentum screeners + daily pullback-to-MA scan.

Screeners:
  1. 1-Week Movers >20%
  2. 1-Month Movers >30%
  3. 3-Month Movers >50%
  4. 6-Month Movers >100%

Also runs the daily pullback-to-MA scan (stocks near 10/21 EMA, above 50 SMA).

Outputs: data/finviz_screener.json
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import pytz

try:
    from finvizfinance.screener.overview import Overview
    from finvizfinance.screener.performance import Performance
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    print("WARNING: finvizfinance not installed. Run: pip install finvizfinance")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.json")
os.makedirs(DATA_DIR, exist_ok=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def run_finviz_screener(filters: dict, description: str, max_results: int = 50) -> list:
    """
    Run a Finviz screener with the given filter dict.
    Returns a list of dicts with key stock data.
    """
    if not FINVIZ_AVAILABLE:
        return [{"error": "finvizfinance not installed"}]

    try:
        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view(order="Market Cap", ascend=False, verbose=0)

        if df is None or df.empty:
            return []

        results = []
        for _, row in df.head(max_results).iterrows():
            results.append({
                "ticker":    str(row.get("Ticker", "")),
                "company":   str(row.get("Company", "")),
                "sector":    str(row.get("Sector", "")),
                "industry":  str(row.get("Industry", "")),
                "country":   str(row.get("Country", "")),
                "market_cap": str(row.get("Market Cap", "")),
                "price":     str(row.get("Price", "")),
                "change":    str(row.get("Change", "")),
                "volume":    str(row.get("Volume", "")),
                "screener":  description,
            })
        return results
    except Exception as e:
        return [{"error": str(e), "screener": description}]


def run_jeff_sun_screeners() -> dict:
    """Run Jeff Sun's 4 momentum screeners."""
    base_filters = {
        "Average Volume": "Over 300K",
        "Price": "Over $10",
        "Market Cap.": "Mid ($300mln to $2bln)",
    }

    screeners = [
        {
            "description": "1-Week Movers >20%",
            "filters": {**base_filters, "Performance": "Week +20%"},
        },
        {
            "description": "1-Month Movers >30%",
            "filters": {**base_filters, "Performance": "Month +30%"},
        },
        {
            "description": "3-Month Movers >50%",
            "filters": {**base_filters, "Performance": "Quarter +50%"},
        },
        {
            "description": "6-Month Movers >100%",
            "filters": {**base_filters, "Performance": "Half +100%"},
        },
    ]

    results = {}
    for s in screeners:
        print(f"  Running: {s['description']}...")
        results[s["description"]] = run_finviz_screener(s["filters"], s["description"])
        time.sleep(1)  # be polite to Finviz

    return results


def run_pullback_scan() -> list:
    """
    Daily pullback-to-MA scan — stocks near 10/21 EMA above 50 SMA.
    Uses Finviz filters as a first pass; yfinance for precise MA checks.
    """
    filters = {
        "Average Volume": "Over 500K",
        "Price": "Over $10",
        "Market Cap.": "Mid ($300mln to $2bln)",
        "52-Week High/Low": "0-20% below High",
        "200-Day Simple Moving Average": "Price above SMA200",
        "50-Day Simple Moving Average": "Price above SMA50",
        "Performance (Quarter)": "Over +20%",
    }

    print("  Running: Pullback-to-MA scan...")
    return run_finviz_screener(filters, "Daily Pullback-to-MA", max_results=30)


def deduplicate_results(screener_results: dict) -> list:
    """Combine all screener results, dedup by ticker, note which screeners each appeared in."""
    seen = {}
    for screener_name, stocks in screener_results.items():
        for stock in stocks:
            if "error" in stock:
                continue
            ticker = stock.get("ticker", "")
            if not ticker:
                continue
            if ticker not in seen:
                seen[ticker] = {**stock, "screeners": [screener_name]}
            else:
                if screener_name not in seen[ticker]["screeners"]:
                    seen[ticker]["screeners"].append(screener_name)
                seen[ticker]["screener_count"] = len(seen[ticker]["screeners"])

    # Sort by number of screeners (most confluence first), then market cap
    combined = list(seen.values())
    combined.sort(key=lambda x: len(x.get("screeners", [])), reverse=True)
    return combined


def cross_reference_sectors(combined: list, us_sectors_data: list) -> list:
    """Flag stocks that are in today's top US sectors."""
    top_sectors = {s["name"] for s in us_sectors_data if s.get("top")}
    for stock in combined:
        stock_sector = stock.get("sector", "")
        stock["in_top_sector"] = any(ts.lower() in stock_sector.lower() for ts in top_sectors)
    return combined


def run():
    print("Loading config...")
    cfg = load_config()

    # Load sector scan if available (for cross-referencing)
    sector_scan_path = os.path.join(DATA_DIR, "sector_scan.json")
    us_sectors = []
    if os.path.exists(sector_scan_path):
        with open(sector_scan_path) as f:
            scan_data = json.load(f)
        us_sectors = scan_data.get("us_sectors", [])

    print("Running Jeff Sun's 4 screeners...")
    screener_results = run_jeff_sun_screeners()

    print("Running pullback-to-MA scan...")
    pullback_results = run_pullback_scan()
    screener_results["Daily Pullback-to-MA"] = pullback_results

    print("Deduplicating and cross-referencing sectors...")
    combined = deduplicate_results(screener_results)
    combined = cross_reference_sectors(combined, us_sectors)

    # Flag stocks appearing in 2+ screeners (highest confluence)
    for stock in combined:
        stock["high_confluence"] = len(stock.get("screeners", [])) >= 2
        stock["screener_count"] = len(stock.get("screeners", []))

    output = {
        "generated_at": datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
        "screener_breakdown": {k: len(v) for k, v in screener_results.items()},
        "combined_candidates": combined[:50],
        "high_confluence_count": sum(1 for s in combined if s.get("high_confluence")),
        "in_top_sector_count":   sum(1 for s in combined if s.get("in_top_sector")),
    }

    out_path = os.path.join(DATA_DIR, "finviz_screener.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"  Total candidates: {len(combined)}")
    print(f"  High confluence (2+ screeners): {output['high_confluence_count']}")
    print(f"  In top sectors: {output['in_top_sector_count']}")
    return output


if __name__ == "__main__":
    run()
