"""
breadth_monitor.py — Market breadth indicators.

Computes:
  - $MMTW proxy: % of S&P 500 components above their 20-day MA
  - NH/NL: New Highs / New Lows from Barchart.com
  - MCO/MCSI: McClellan Oscillator + Summation Index on NDX components

Outputs: data/breadth.json
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pytz
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# S&P 500 component tickers (via Wikipedia)
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list:
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"  WARNING: Could not fetch S&P 500 tickers: {e}")
        # Fallback: use SPY components as a proxy via yfinance
        return []


def get_ndx100_tickers() -> list:
    """Fetch current Nasdaq 100 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            if "Ticker" in t.columns:
                return t["Ticker"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"  WARNING: Could not fetch NDX 100 tickers: {e}")
    return []


# ---------------------------------------------------------------------------
# $MMTW proxy: % of S&P 500 above 20-day MA
# ---------------------------------------------------------------------------

def compute_mmtw_proxy(tickers: list, sample_size: int = 100) -> dict:
    """
    Compute % of S&P 500 components above their 20-day SMA.
    Uses a sample for speed (set sample_size=500 for full computation — takes ~5 min).
    """
    if not tickers:
        return {"mmtw_proxy": None, "error": "no tickers"}

    # Sample for speed
    import random
    random.seed(42)
    sample = random.sample(tickers, min(sample_size, len(tickers)))

    print(f"  Computing $MMTW proxy on {len(sample)} stocks...")
    above_count = 0
    valid_count = 0

    for ticker in sample:
        try:
            hist = yf.Ticker(ticker).history(period="3mo", auto_adjust=True)
            if len(hist) < 21:
                continue
            close = hist["Close"]
            sma20 = float(close.rolling(20).mean().iloc[-1])
            price = float(close.iloc[-1])
            if price > sma20:
                above_count += 1
            valid_count += 1
        except Exception:
            pass

    if valid_count == 0:
        return {"mmtw_proxy": None, "error": "computation failed"}

    mmtw = round(above_count / valid_count * 100, 1)

    if mmtw > 70:
        signal = "OVERBOUGHT — Caution, stop adding"
    elif mmtw > 50:
        signal = "HEALTHY — Uptrend participation broad"
    elif mmtw > 40:
        signal = "NEUTRAL — Mixed conditions"
    else:
        signal = "WASHED OUT — Watch for reversal opportunity"

    return {
        "mmtw_proxy": mmtw,
        "above_count": above_count,
        "valid_count": valid_count,
        "signal": signal,
        "note": f"Sampled {len(sample)} of {len(tickers)} S&P 500 components",
    }


# ---------------------------------------------------------------------------
# NH/NL from Barchart.com
# ---------------------------------------------------------------------------

def fetch_nh_nl() -> dict:
    """Scrape New Highs / New Lows from Barchart.com."""
    url = "https://www.barchart.com/stocks/highs-lows/highs-lows-stocks"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}

        soup = BeautifulSoup(resp.text, "lxml")
        # Barchart renders some data in a summary table
        tables = soup.find_all("table")
        for table in tables:
            text = table.get_text()
            if "New High" in text or "New Low" in text:
                rows = table.find_all("tr")
                data = {}
                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if len(cells) >= 2:
                        data[cells[0]] = cells[1]
                if data:
                    return {"raw": data, "source": "Barchart.com"}

        # Fallback: look for specific text patterns
        text = soup.get_text()
        return {"raw_text_length": len(text), "note": "Table structure not found — Barchart may have changed layout", "source": "Barchart.com"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# MCO/MCSI: McClellan Oscillator + Summation Index
# ---------------------------------------------------------------------------

def compute_mco_mcsi(tickers: list) -> dict:
    """
    Compute McClellan Oscillator (MCO) and Summation Index (MCSI)
    on the provided ticker universe (ideally NDX 100 components).

    MCO = 19-day EMA of (Advances - Declines) minus 39-day EMA of (Advances - Declines)
    MCSI = running sum of MCO values

    We approximate advances/declines daily by comparing each stock's close to prior close.
    """
    if not tickers:
        return {"mco": None, "mcsi": None, "error": "no tickers"}

    print(f"  Computing MCO/MCSI on {len(tickers)} NDX components...")

    # Download 6 months of daily closes for all tickers
    try:
        raw = yf.download(
            tickers[:100],  # cap at 100 for speed
            period="6mo",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return {"mco": None, "mcsi": None, "error": "download failed"}

        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

        # Daily advance/decline line
        daily_changes = closes.pct_change()
        advances = (daily_changes > 0).sum(axis=1)
        declines  = (daily_changes < 0).sum(axis=1)
        net_adv   = advances - declines

        # 19-day and 39-day EMA of net advances
        ema19 = net_adv.ewm(span=19, adjust=False).mean()
        ema39 = net_adv.ewm(span=39, adjust=False).mean()
        mco_series = ema19 - ema39

        # MCSI = cumulative sum of MCO
        mcsi_series = mco_series.cumsum()

        # Current values
        mco_current  = round(float(mco_series.iloc[-1]), 2)
        mcsi_current = round(float(mcsi_series.iloc[-1]), 2)

        # Standard deviations for ±1σ / ±2σ bands
        mco_std  = float(mco_series.std())
        mcsi_std = float(mcsi_series.std())

        mco_1sigma_plus  = round(mco_std, 2)
        mco_1sigma_minus = round(-mco_std, 2)
        mco_2sigma_plus  = round(2 * mco_std, 2)
        mco_2sigma_minus = round(-2 * mco_std, 2)

        # MCSI trend: is it curling up or down vs its 10-day MA?
        mcsi_ma10 = mcsi_series.rolling(10).mean()
        mcsi_above_10ma = float(mcsi_series.iloc[-1]) > float(mcsi_ma10.iloc[-1])
        # "Curling" = last 3 days of MCSI vs MA10 changing direction
        mcsi_recent = mcsi_series.iloc[-3:].values
        mcsi_ma10_recent = mcsi_ma10.iloc[-3:].values
        mcsi_curling_up   = bool(mcsi_recent[-1] > mcsi_ma10_recent[-1] and mcsi_recent[-2] <= mcsi_ma10_recent[-2])
        mcsi_curling_down = bool(mcsi_recent[-1] < mcsi_ma10_recent[-1] and mcsi_recent[-2] >= mcsi_ma10_recent[-2])

        # Interpretation
        if mco_current < mco_2sigma_minus:
            mco_signal = "DEEPLY OVERSOLD (-2σ) — High-probability reversal window"
        elif mco_current < mco_1sigma_minus:
            mco_signal = "OVERSOLD (-1σ) — Timing window open for pullback entries"
        elif mco_current > mco_2sigma_plus:
            mco_signal = "DEEPLY OVERBOUGHT (+2σ) — Late stage, trim into strength"
        elif mco_current > mco_1sigma_plus:
            mco_signal = "OVERBOUGHT (+1σ) — Stop adding new positions"
        else:
            mco_signal = "NEUTRAL — Normal conditions"

        if mcsi_curling_up:
            mcsi_signal = "CURLING UP — Press with conviction, full sizing"
        elif mcsi_curling_down:
            mcsi_signal = "CURLING DOWN — Stop adding, no new positions"
        elif mcsi_above_10ma:
            mcsi_signal = "ABOVE 10MA — Participation confirmed"
        else:
            mcsi_signal = "BELOW 10MA — Caution"

        # Last 20 MCO values for sparkline
        mco_history = [round(float(v), 2) for v in mco_series.iloc[-20:].values if not np.isnan(v)]

        return {
            "mco": mco_current,
            "mcsi": mcsi_current,
            "mco_1sigma_plus":  mco_1sigma_plus,
            "mco_1sigma_minus": mco_1sigma_minus,
            "mco_2sigma_plus":  mco_2sigma_plus,
            "mco_2sigma_minus": mco_2sigma_minus,
            "mcsi_10ma": round(float(mcsi_ma10.iloc[-1]), 2),
            "mcsi_above_10ma": mcsi_above_10ma,
            "mcsi_curling_up":   mcsi_curling_up,
            "mcsi_curling_down": mcsi_curling_down,
            "mco_signal":  mco_signal,
            "mcsi_signal": mcsi_signal,
            "mco_history": mco_history,
            "universe_size": len(tickers[:100]),
        }

    except Exception as e:
        return {"mco": None, "mcsi": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("Fetching S&P 500 components for $MMTW...")
    sp500_tickers = get_sp500_tickers()

    print("Fetching NDX 100 components for MCO/MCSI...")
    ndx_tickers = get_ndx100_tickers()

    print("Computing $MMTW proxy...")
    mmtw = compute_mmtw_proxy(sp500_tickers, sample_size=100)

    print("Fetching NH/NL from Barchart...")
    nh_nl = fetch_nh_nl()

    print("Computing MCO/MCSI...")
    mco_mcsi = compute_mco_mcsi(ndx_tickers if ndx_tickers else sp500_tickers[:100])

    # Combined breadth score (simple: 0-100)
    breadth_score = 0
    breadth_factors = []

    if mmtw.get("mmtw_proxy") is not None:
        mmtw_val = mmtw["mmtw_proxy"]
        breadth_score += min(mmtw_val, 100) * 0.4
        breadth_factors.append(f"$MMTW: {mmtw_val}%")

    mco_val = mco_mcsi.get("mco")
    if mco_val is not None:
        # Normalize MCO to 0-100 range based on ±2σ bands
        mco_2s = mco_mcsi.get("mco_2sigma_plus", 1)
        normalized = min(max((mco_val / (mco_2s * 2) + 0.5) * 100, 0), 100)
        breadth_score += normalized * 0.3
        breadth_factors.append(f"MCO: {mco_val}")

    if mco_mcsi.get("mcsi_above_10ma"):
        breadth_score += 30
        breadth_factors.append("MCSI above 10MA")

    breadth_score = round(breadth_score, 1)
    if breadth_score >= 65:
        breadth_summary = "BULLISH"
    elif breadth_score >= 45:
        breadth_summary = "NEUTRAL"
    else:
        breadth_summary = "BEARISH"

    output = {
        "generated_at": datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
        "mmtw_proxy": mmtw,
        "nh_nl": nh_nl,
        "mco_mcsi": mco_mcsi,
        "breadth_score": breadth_score,
        "breadth_summary": breadth_summary,
        "breadth_factors": breadth_factors,
    }

    out_path = os.path.join(DATA_DIR, "breadth.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"  $MMTW proxy: {mmtw.get('mmtw_proxy')}%")
    print(f"  MCO: {mco_mcsi.get('mco')}  |  MCSI: {mco_mcsi.get('mcsi')}")
    print(f"  Breadth score: {breadth_score} ({breadth_summary})")
    return output


if __name__ == "__main__":
    run()
