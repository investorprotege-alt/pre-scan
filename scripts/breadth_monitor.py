"""
breadth_monitor.py — Market breadth indicators.

Computes:
  - $MMTW proxy: % of S&P 500 components above their 20-day MA
  - NH/NL: New Highs / New Lows computed via yfinance (52-week range proximity)
  - MCO/MCSI: McClellan Oscillator + Summation Index on NDX components

Outputs: data/breadth.json
"""

import json
import os
import time
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pytz

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardcoded fallback ticker lists (used when Wikipedia fetch fails)
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

NDX100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","ADBE","CSCO","PEP","TMUS","QCOM","INTC","INTU","TXN",
    "AMGN","HON","AMAT","BKNG","SBUX","MDLZ","ISRG","GILD","ADI","REGN",
    "VRTX","MU","PANW","LRCX","KLAC","SNPS","CDNS","FTNT","MRVL","MELI",
    "CHTR","ORLY","WDAY","DXCM","ABNB","IDXX","CTAS","BIIB","PCAR","ROST",
    "FAST","ODFL","CPRT","PAYX","MCHP","NXPI","DLTR","VRSK","ANSS","TTWO",
    "EBAY","ZS","TEAM","CRWD","DDOG","ON","GFS","ARM","GEHC","CEG","KHC",
    "EXC","FANG","SIRI","WBD","RIVN","ZM","OKTA","SPLK","SMCI","LCID",
    "ILMN","ALGN","ENPH","LULU","MRNA","PYPL","SGEN","BMRN","NTES","JD",
    "BIDU","ATVI","DOCU","MTCH","PTON","RGEN","SIRI","SWKS","XLNX","ZBRA",
]


# ---------------------------------------------------------------------------
# Ticker fetchers — Wikipedia with requests + StringIO, fallback to hardcoded
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list:
    """Fetch S&P 500 tickers from Wikipedia using proper User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; pre-market-bot/1.0; +https://github.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  Got {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"  WARNING: Wikipedia S&P 500 fetch failed ({e}) — using fallback list")
        return SP500_FALLBACK


def get_ndx100_tickers() -> list:
    """Fetch Nasdaq-100 tickers from Wikipedia using proper User-Agent."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; pre-market-bot/1.0; +https://github.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for t in tables:
            if "Ticker" in t.columns:
                tickers = t["Ticker"].str.replace(".", "-", regex=False).tolist()
                print(f"  Got {len(tickers)} NDX 100 tickers from Wikipedia")
                return tickers
        raise ValueError("Ticker column not found in any table")
    except Exception as e:
        print(f"  WARNING: Wikipedia NDX 100 fetch failed ({e}) — using fallback list")
        return NDX100_FALLBACK


# ---------------------------------------------------------------------------
# Batch download helper
# ---------------------------------------------------------------------------

def batch_download(tickers: list, period: str = "1y") -> pd.DataFrame:
    """Download OHLCV for a ticker list, returning a Close DataFrame."""
    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()
        # Handle both single-ticker (Series) and multi-ticker (DataFrame) results
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].dropna(how="all", axis=1)
        else:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw
        return closes
    except Exception as e:
        print(f"  WARNING: batch download failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# $MMTW proxy: % of tickers above 20-day SMA
# ---------------------------------------------------------------------------

def compute_mmtw_proxy(tickers: list, sample_size: int = 100) -> dict:
    """% of S&P 500 components above their 20-day SMA (batch yfinance download)."""
    import random
    random.seed(42)
    sample = random.sample(tickers, min(sample_size, len(tickers)))
    print(f"  Computing $MMTW proxy on {len(sample)} stocks (batch download)...")

    closes = batch_download(sample, period="3mo")
    if closes.empty:
        return {"mmtw_proxy": None, "error": "download failed"}

    sma20 = closes.rolling(20).mean()
    latest_close = closes.iloc[-1]
    latest_sma20 = sma20.iloc[-1]

    valid = latest_sma20.notna() & latest_close.notna()
    if valid.sum() == 0:
        return {"mmtw_proxy": None, "error": "no valid data"}

    above = (latest_close[valid] > latest_sma20[valid]).sum()
    total = valid.sum()
    mmtw = round(float(above / total * 100), 1)

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
        "above_count": int(above),
        "valid_count": int(total),
        "signal": signal,
        "note": f"Sampled {len(sample)} of {len(tickers)} S&P 500 components",
    }


# ---------------------------------------------------------------------------
# NH/NL: computed via yfinance 52-week range (replaces Barchart scrape)
# ---------------------------------------------------------------------------

def compute_nh_nl(tickers: list, sample_size: int = 100) -> dict:
    """
    Compute New Highs / New Lows by checking proximity to 52-week high/low.
    A stock is a 'New High' if it's within 1% of its 52-week high.
    A stock is a 'New Low'  if it's within 1% of its 52-week low.
    """
    import random
    random.seed(99)
    sample = random.sample(tickers, min(sample_size, len(tickers)))
    print(f"  Computing NH/NL on {len(sample)} stocks (batch download)...")

    raw = yf.download(sample, period="1y", auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return {"new_highs": None, "new_lows": None, "error": "download failed"}

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].dropna(how="all", axis=1)
        highs  = raw["High"].dropna(how="all", axis=1)
        lows   = raw["Low"].dropna(how="all", axis=1)
    else:
        return {"new_highs": None, "new_lows": None, "error": "unexpected data shape"}

    latest_close = closes.iloc[-1]
    high_52w     = highs.max()
    low_52w      = lows.min()

    valid = latest_close.notna() & high_52w.notna() & low_52w.notna()
    new_highs = int((latest_close[valid] >= high_52w[valid] * 0.99).sum())
    new_lows  = int((latest_close[valid] <= low_52w[valid]  * 1.01).sum())
    total     = int(valid.sum())

    ratio = round(new_highs / max(new_lows, 1), 2)
    if ratio >= 3:
        signal = "STRONG — NH/NL ratio bullish"
    elif ratio >= 1:
        signal = "NEUTRAL — Mixed highs/lows"
    else:
        signal = "WEAK — More new lows than highs"

    return {
        "new_highs": new_highs,
        "new_lows":  new_lows,
        "nh_nl_ratio": ratio,
        "total_checked": total,
        "signal": signal,
        "note": f"Sampled {len(sample)} tickers, within 1% of 52-week range",
        "source": "yfinance",
    }


# ---------------------------------------------------------------------------
# MCO/MCSI: McClellan Oscillator + Summation Index
# ---------------------------------------------------------------------------

def compute_mco_mcsi(tickers: list) -> dict:
    """
    Compute McClellan Oscillator (MCO) and Summation Index (MCSI)
    on the provided ticker universe (NDX 100 components).
    """
    sample = tickers[:100]
    print(f"  Computing MCO/MCSI on {len(sample)} NDX components (batch download)...")

    closes = batch_download(sample, period="6mo")
    if closes.empty or len(closes) < 40:
        return {"mco": None, "mcsi": None, "error": "insufficient data"}

    # Daily advance/decline line
    daily_changes = closes.pct_change()
    advances = (daily_changes > 0).sum(axis=1)
    declines  = (daily_changes < 0).sum(axis=1)
    net_adv   = advances - declines

    # 19-day and 39-day EMA of net advances
    ema19 = net_adv.ewm(span=19, adjust=False).mean()
    ema39 = net_adv.ewm(span=39, adjust=False).mean()
    mco_series = ema19 - ema39
    mcsi_series = mco_series.cumsum()

    mco_current  = round(float(mco_series.iloc[-1]), 2)
    mcsi_current = round(float(mcsi_series.iloc[-1]), 2)

    mco_std = float(mco_series.std())
    mco_1sp =  round(mco_std, 2)
    mco_1sm =  round(-mco_std, 2)
    mco_2sp =  round(2 * mco_std, 2)
    mco_2sm =  round(-2 * mco_std, 2)

    mcsi_ma10 = mcsi_series.rolling(10).mean()
    mcsi_above_10ma = float(mcsi_series.iloc[-1]) > float(mcsi_ma10.iloc[-1])
    mcsi_recent     = mcsi_series.iloc[-3:].values
    mcsi_ma10_recent = mcsi_ma10.iloc[-3:].values
    mcsi_curling_up   = bool(len(mcsi_recent) >= 3 and
                             mcsi_recent[-1] > mcsi_ma10_recent[-1] and
                             mcsi_recent[-2] <= mcsi_ma10_recent[-2])
    mcsi_curling_down = bool(len(mcsi_recent) >= 3 and
                             mcsi_recent[-1] < mcsi_ma10_recent[-1] and
                             mcsi_recent[-2] >= mcsi_ma10_recent[-2])

    if mco_current < mco_2sm:
        mco_signal = "DEEPLY OVERSOLD (-2\u03c3) \u2014 High-probability reversal window"
    elif mco_current < mco_1sm:
        mco_signal = "OVERSOLD (-1\u03c3) \u2014 Timing window open for pullback entries"
    elif mco_current > mco_2sp:
        mco_signal = "DEEPLY OVERBOUGHT (+2\u03c3) \u2014 Late stage, trim into strength"
    elif mco_current > mco_1sp:
        mco_signal = "OVERBOUGHT (+1\u03c3) \u2014 Stop adding new positions"
    else:
        mco_signal = "NEUTRAL \u2014 Normal conditions"

    if mcsi_curling_up:
        mcsi_signal = "CURLING UP \u2014 Press with conviction, full sizing"
    elif mcsi_curling_down:
        mcsi_signal = "CURLING DOWN \u2014 Stop adding, no new positions"
    elif mcsi_above_10ma:
        mcsi_signal = "ABOVE 10MA \u2014 Participation confirmed"
    else:
        mcsi_signal = "BELOW 10MA \u2014 Caution"

    mco_history = [round(float(v), 2) for v in mco_series.iloc[-20:].values
                   if not np.isnan(v)]

    return {
        "mco":  mco_current,
        "mcsi": mcsi_current,
        "mco_1sigma_plus":  mco_1sp,
        "mco_1sigma_minus": mco_1sm,
        "mco_2sigma_plus":  mco_2sp,
        "mco_2sigma_minus": mco_2sm,
        "mcsi_10ma":          round(float(mcsi_ma10.iloc[-1]), 2),
        "mcsi_above_10ma":    mcsi_above_10ma,
        "mcsi_curling_up":    mcsi_curling_up,
        "mcsi_curling_down":  mcsi_curling_down,
        "mco_signal":  mco_signal,
        "mcsi_signal": mcsi_signal,
        "mco_history": mco_history,
        "universe_size": len(sample),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("Fetching S&P 500 tickers...")
    sp500 = get_sp500_tickers()

    print("Fetching NDX 100 tickers...")
    ndx100 = get_ndx100_tickers()

    print("Computing $MMTW proxy...")
    mmtw = compute_mmtw_proxy(sp500, sample_size=100)

    print("Computing NH/NL (via yfinance 52-week range)...")
    nh_nl = compute_nh_nl(sp500, sample_size=100)

    print("Computing MCO/MCSI...")
    mco_mcsi = compute_mco_mcsi(ndx100 if ndx100 else sp500[:100])

    # Combined breadth score (0-100)
    breadth_score = 0
    breadth_factors = []

    mmtw_val = mmtw.get("mmtw_proxy")
    if mmtw_val is not None:
        breadth_score += min(mmtw_val, 100) * 0.4
        breadth_factors.append(f"$MMTW: {mmtw_val}%")

    mco_val = mco_mcsi.get("mco")
    if mco_val is not None:
        mco_2sp = mco_mcsi.get("mco_2sigma_plus", 1) or 1
        normalized = min(max((mco_val / (mco_2sp * 2) + 0.5) * 100, 0), 100)
        breadth_score += normalized * 0.3
        breadth_factors.append(f"MCO: {mco_val}")

    if mco_mcsi.get("mcsi_above_10ma"):
        breadth_score += 30
        breadth_factors.append("MCSI above 10MA")

    breadth_score = round(breadth_score, 1)
    breadth_summary = "BULLISH" if breadth_score >= 65 else "NEUTRAL" if breadth_score >= 45 else "BEARISH"

    output = {
        "generated_at":   datetime.now(pytz.timezone("Australia/Sydney")).isoformat(),
        "mmtw_proxy":     mmtw,
        "nh_nl":          nh_nl,
        "mco_mcsi":       mco_mcsi,
        "breadth_score":  breadth_score,
        "breadth_summary": breadth_summary,
        "breadth_factors": breadth_factors,
    }

    out_path = os.path.join(DATA_DIR, "breadth.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved -> {out_path}")
    print(f"  $MMTW: {mmtw_val}%  |  NH: {nh_nl.get('new_highs')}  NL: {nh_nl.get('new_lows')}")
    print(f"  MCO: {mco_mcsi.get('mco')}  |  MCSI: {mco_mcsi.get('mcsi')}")
    print(f"  Breadth score: {breadth_score} ({breadth_summary})")
    return output


if __name__ == "__main__":
    run()
