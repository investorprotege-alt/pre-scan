"""
sector_scan.py — Ranks US + ASX sectors by 1W/1M ROC, checks index health,
commodities, FX, and QQQE 21DMA structure scenario.

Outputs: data/sector_scan.json
"""

import json
import sys
import os
from datetime import datetime, timedelta

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


def pct_change_over(prices: pd.Series, days: int) -> float:
    """Return % change over the last `days` trading days."""
    if len(prices) < days + 1:
        return None
    return round((prices.iloc[-1] / prices.iloc[-days - 1] - 1) * 100, 2)


def get_ema(prices: pd.Series, period: int) -> float:
    if len(prices) < period:
        return None
    return round(float(prices.ewm(span=period, adjust=False).mean().iloc[-1]), 4)


def get_sma(prices: pd.Series, period: int) -> float:
    if len(prices) < period:
        return None
    return round(float(prices.rolling(period).mean().iloc[-1]), 4)


def get_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over `period` days."""
    if len(hist) < period + 1:
        return None
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 4)


def fetch_ticker_data(ticker: str, period: str = "6mo") -> dict:
    """Download OHLCV and compute key metrics for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            return {"ticker": ticker, "error": "no data"}

        close = hist["Close"]
        price = round(float(close.iloc[-1]), 4)

        result = {
            "ticker": ticker,
            "price": price,
            "1w_roc": pct_change_over(close, 5),
            "1m_roc": pct_change_over(close, 21),
            "3m_roc": pct_change_over(close, 63),
            "6m_roc": pct_change_over(close, 126),
            "ema10":  get_ema(close, 10),
            "ema21":  get_ema(close, 21),
            "sma50":  get_sma(close, 50),
            "sma200": get_sma(close, 200),
            "atr14":  get_atr(hist, 14),
        }

        # Distance from 50MA and 200MA (% above/below)
        if result["sma50"]:
            result["pct_from_50sma"] = round((price / result["sma50"] - 1) * 100, 2)
        if result["sma200"]:
            result["above_200sma"] = price > result["sma200"]
            result["pct_from_200sma"] = round((price / result["sma200"] - 1) * 100, 2)

        # ATR% from 50MA (Jeff Sun regime filter)
        if result["sma50"] and result["atr14"]:
            dist_from_50 = abs(price - result["sma50"])
            result["atr_mult_from_50sma"] = round(dist_from_50 / result["atr14"], 2)

        # 52-week high/low
        hist_1y = t.history(period="1y", auto_adjust=True)
        if not hist_1y.empty:
            result["52w_high"] = round(float(hist_1y["High"].max()), 4)
            result["52w_low"]  = round(float(hist_1y["Low"].min()), 4)
            result["pct_from_52w_high"] = round((price / result["52w_high"] - 1) * 100, 2)

        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def compute_qqqe_scenario(data: dict) -> dict:
    """
    Classify QQQE into one of Alex's (PrimeTrading) 4 scenarios using 21DMA structure.

    The 21DMA structure is a band defined by:
      - High band:  21 EMA of daily highs
      - Low band:   21 EMA of daily lows
      - Mid:        21 EMA of daily closes

    Scenarios:
      1 = Pullback in uptrend  (price above mid band, above 200MA, pulling back to structure)
      2 = Reclaim & backtest   (price recently broke below low band but recovered above mid)
      3 = Reject & higher low  (price below mid band but above 200MA, higher lows)
      4 = Reject & lower low   (price below low band, below 200MA, no longs)
    """
    try:
        t = yf.Ticker("QQQE")
        hist = t.history(period="6mo", auto_adjust=True)
        if len(hist) < 30:
            return {"scenario": None, "description": "insufficient data"}

        high_ema21  = float(hist["High"].ewm(span=21, adjust=False).mean().iloc[-1])
        low_ema21   = float(hist["Low"].ewm(span=21, adjust=False).mean().iloc[-1])
        close_ema21 = float(hist["Close"].ewm(span=21, adjust=False).mean().iloc[-1])
        sma200      = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
        price       = float(hist["Close"].iloc[-1])

        # Look at last 5 sessions to detect "recent recovery"
        recent_closes = hist["Close"].iloc[-5:].values
        recent_lows   = hist["Close"].iloc[-10:].values

        above_mid_band   = price > close_ema21
        above_high_band  = price > high_ema21
        below_low_band   = price < low_ema21
        above_200        = (price > sma200) if sma200 else True
        recent_recovery  = (recent_closes[-1] > close_ema21) and (recent_closes[0] < low_ema21)
        making_higher_lows = recent_lows[-1] > recent_lows[0]

        if above_200 and above_mid_band and not above_high_band:
            scenario, desc = 1, "Pullback in uptrend — aggressive, full Focus List execution"
        elif above_200 and above_mid_band and recent_recovery:
            scenario, desc = 2, "Reclaim & backtest — engage, build positions"
        elif above_200 and not above_mid_band and making_higher_lows:
            scenario, desc = 3, "Reject & higher low — pilot positions only (max 1-2% NER)"
        else:
            scenario, desc = 4, "Reject & lower low — NO longs, preserve capital"

        return {
            "scenario": scenario,
            "description": desc,
            "price": round(price, 4),
            "ema21_high_band": round(high_ema21, 4),
            "ema21_mid_band":  round(close_ema21, 4),
            "ema21_low_band":  round(low_ema21, 4),
            "above_200sma": above_200,
            "sma200": round(sma200, 4) if sma200 else None,
        }
    except Exception as e:
        return {"scenario": None, "error": str(e)}


def rank_sectors(sector_list: list, data_map: dict, label: str) -> list:
    """Return sector list sorted by 1W ROC descending, with top N flagged."""
    rows = []
    for item in sector_list:
        ticker = item["ticker"]
        d = data_map.get(ticker, {})
        if "error" in d:
            continue
        rows.append({
            "ticker":   ticker,
            "name":     item.get("name", ticker),
            "price":    d.get("price"),
            "1w_roc":   d.get("1w_roc"),
            "1m_roc":   d.get("1m_roc"),
            "3m_roc":   d.get("3m_roc"),
            "above_200sma": d.get("above_200sma"),
        })
    rows.sort(key=lambda x: x["1w_roc"] if x["1w_roc"] is not None else -999, reverse=True)
    top_n = 3 if label == "US" else 2
    for i, row in enumerate(rows):
        row["rank"] = i + 1
        row["top"] = i < top_n
    return rows


def run():
    print("Loading config...")
    cfg = load_config()

    # --- Collect all tickers to download ---
    all_tickers = set()
    for item in cfg["us_sector_etfs"]:     all_tickers.add(item["ticker"])
    for item in cfg["asx_sector_etfs"]:    all_tickers.add(item["ticker"])
    for item in cfg["indices"]:            all_tickers.add(item["ticker"])
    for item in cfg["commodities"]:        all_tickers.add(item["ticker"])
    for item in cfg["fx"]:                 all_tickers.add(item["ticker"])
    # ASX leaders
    for group in cfg["asx_leaders"].values():
        for item in group:
            all_tickers.add(item["ticker"])

    print(f"Fetching data for {len(all_tickers)} tickers...")
    data_map = {}
    for ticker in sorted(all_tickers):
        print(f"  {ticker}...", end="", flush=True)
        data_map[ticker] = fetch_ticker_data(ticker)
        print(" done")

    # --- QQQE 21DMA scenario ---
    print("Computing QQQE scenario...")
    qqqe_scenario = compute_qqqe_scenario(data_map)

    # --- Sector rankings ---
    us_sectors_ranked  = rank_sectors(cfg["us_sector_etfs"], data_map, "US")
    asx_sectors_ranked = rank_sectors(cfg["asx_sector_etfs"], data_map, "ASX")

    # --- Index health ---
    index_health = []
    for item in cfg["indices"]:
        d = data_map.get(item["ticker"], {})
        index_health.append({
            "ticker":     item["ticker"],
            "name":       item["name"],
            "price":      d.get("price"),
            "sma50":      d.get("sma50"),
            "sma200":     d.get("sma200"),
            "above_50sma":  d.get("price", 0) > d.get("sma50", 9e9) if d.get("sma50") else None,
            "above_200sma": d.get("above_200sma"),
            "1w_roc":     d.get("1w_roc"),
            "1m_roc":     d.get("1m_roc"),
            "atr_mult_from_50sma": d.get("atr_mult_from_50sma"),
        })

    # RSP/SPY and QQQE/QQQ equal-weight divergence
    rsp = data_map.get("RSP", {}).get("price")
    spy = data_map.get("SPY", {}).get("price")
    qqq = data_map.get("QQQ", {}).get("price")
    qqqe = data_map.get("QQQE", {}).get("price")
    equal_weight_ratios = {
        "rsp_spy_ratio": round(rsp / spy, 4) if rsp and spy else None,
        "qqqe_qqq_ratio": round(qqqe / qqq, 4) if qqqe and qqq else None,
    }

    # --- Commodities ---
    commodities = []
    for item in cfg["commodities"]:
        d = data_map.get(item["ticker"], {})
        commodities.append({
            "ticker":       item["ticker"],
            "name":         item["name"],
            "leveraged_etf": item.get("leveraged_etf"),
            "price":        d.get("price"),
            "1w_roc":       d.get("1w_roc"),
            "1m_roc":       d.get("1m_roc"),
            "above_50sma":  d.get("price", 0) > d.get("sma50", 9e9) if d.get("sma50") else None,
            "above_200sma": d.get("above_200sma"),
            "ema21":        d.get("ema21"),
            "sma50":        d.get("sma50"),
        })

    # --- FX ---
    fx_data = []
    for item in cfg["fx"]:
        d = data_map.get(item["ticker"], {})
        fx_data.append({
            "ticker":   item["ticker"],
            "name":     item["name"],
            "price":    d.get("price"),
            "1w_roc":   d.get("1w_roc"),
            "1m_roc":   d.get("1m_roc"),
            "above_200sma": d.get("above_200sma"),
        })

    # --- ASX leader status ---
    asx_leaders_status = {}
    for group_name, group in cfg["asx_leaders"].items():
        asx_leaders_status[group_name] = []
        for item in group:
            d = data_map.get(item["ticker"], {})
            status = {
                "ticker":   item["ticker"],
                "name":     item["name"],
                "sector":   item["sector"],
                "price":    d.get("price"),
                "ema10":    d.get("ema10"),
                "ema21":    d.get("ema21"),
                "sma50":    d.get("sma50"),
                "sma200":   d.get("sma200"),
                "above_200sma": d.get("above_200sma"),
                "1w_roc":   d.get("1w_roc"),
                "pct_from_52w_high": d.get("pct_from_52w_high"),
            }
            # Proximity to key MAs
            for ma_key, ma_label in [("ema10", "10EMA"), ("ema21", "21EMA"), ("sma50", "50SMA")]:
                ma_val = d.get(ma_key)
                if ma_val and d.get("price"):
                    pct = round((d["price"] / ma_val - 1) * 100, 2)
                    status[f"pct_from_{ma_label.lower()}"] = pct
                    # Flag if within 3% of the MA (potential U&R zone)
                    status[f"near_{ma_label.lower()}"] = abs(pct) <= 3.0
            asx_leaders_status[group_name].append(status)

    # --- VIX regime signal ---
    vix_d = data_map.get("^VIX", {})
    vix_price = vix_d.get("price")
    if vix_price:
        if vix_price < 15:
            vix_regime = "LOW — Complacent market"
        elif vix_price < 20:
            vix_regime = "NORMAL"
        elif vix_price < 30:
            vix_regime = "ELEVATED — Caution"
        else:
            vix_regime = "HIGH — Fear / Potential opportunity"
    else:
        vix_regime = "Unknown"

    # --- Market regime score (simple) ---
    spy_above_200 = data_map.get("SPY", {}).get("above_200sma", False)
    spy_above_50  = (data_map.get("SPY", {}).get("pct_from_50sma", -999) or -999) > 0
    qqqe_above_200 = data_map.get("QQQE", {}).get("above_200sma", False)
    qqqe_scenario_num = qqqe_scenario.get("scenario", 4)

    if qqqe_scenario_num == 1 and spy_above_200 and spy_above_50:
        regime = "CONFIRMED UPTREND — Full exposure"
        exposure_phase = "Confirmed Uptrend Pullback"
    elif qqqe_scenario_num == 2 and spy_above_200:
        regime = "RECOVERING — Build positions"
        exposure_phase = "Out of Correction"
    elif qqqe_scenario_num == 3:
        regime = "CAUTION — Pilot positions only"
        exposure_phase = "Out of Correction"
    else:
        regime = "BEAR / BREAKDOWN — Preserve capital"
        exposure_phase = "Breakdown"

    # --- Assemble output ---
    output = {
        "generated_at": datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
        "market_regime": {
            "regime": regime,
            "exposure_phase": exposure_phase,
            "vix": vix_price,
            "vix_regime": vix_regime,
            "spy_above_200sma": spy_above_200,
            "spy_above_50sma": spy_above_50,
        },
        "qqqe_scenario": qqqe_scenario,
        "equal_weight_ratios": equal_weight_ratios,
        "us_sectors": us_sectors_ranked,
        "asx_sectors": asx_sectors_ranked,
        "index_health": index_health,
        "commodities": commodities,
        "fx": fx_data,
        "asx_leaders": asx_leaders_status,
    }

    out_path = os.path.join(DATA_DIR, "sector_scan.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    return output


if __name__ == "__main__":
    run()
