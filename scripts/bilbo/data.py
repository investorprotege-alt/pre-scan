"""
data.py — bars in, normalised frames out.

Two things every downstream module can then assume:

  * columns are lowercase open/high/low/close/volume
  * an intraday bar is stamped with the time it CLOSED, in US/Eastern

The second one matters more than it looks. yfinance (like most feeds) stamps a
bar with its OPEN time, so the bar labelled 09:30 is the one that closes at
10:30. The "enter only between 10:00 and 15:00 ET" rule is about the close, so
getting this wrong silently shifts every entry by one bar.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ET = "America/New_York"

_INTERVAL_MINUTES = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
    "60m": 60, "1h": 60, "90m": 90, "1d": 1440,
}


def interval_minutes(interval: str) -> int:
    try:
        return _INTERVAL_MINUTES[interval]
    except KeyError:
        raise ValueError(f"unsupported interval {interval!r}") from None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

EMPTY = ["open", "high", "low", "close", "volume"]


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull one ticker out of a (possibly multi-ticker) yfinance frame."""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in set(df.columns.get_level_values(-1)):
            df = df.xs(ticker, axis=1, level=-1)
        elif df.columns.nlevels == 2 and df.columns.get_level_values(-1).nunique() == 1:
            df.columns = df.columns.get_level_values(0)   # single-ticker download
        else:
            # a batch frame that simply has no data for this ticker
            return pd.DataFrame(columns=EMPTY)
    df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in EMPTY if c in df.columns]
    if "close" not in keep:
        return pd.DataFrame(columns=EMPTY)
    return df[keep]


def normalize(df: pd.DataFrame, ticker: str, interval: str, p_data) -> pd.DataFrame:
    """Lowercase columns, US/Eastern index, intraday bars stamped at their close."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=EMPTY)

    out = _flatten(df.copy(), ticker)
    if out.empty:
        return out
    out = out.dropna(subset=["close"]).sort_index()
    if out.empty:
        return out
    idx = pd.DatetimeIndex(out.index)

    if interval == "1d":
        out.index = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
        out.index.name = "date"
        return out

    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert(ET)

    if getattr(p_data, "index_is_bar_open", True):
        shifted = idx + timedelta(minutes=interval_minutes(interval))
        close_h, close_m = (int(x) for x in str(p_data.session_close_et).split(":")[:2])
        cap = pd.DatetimeIndex(
            [t.normalize() + timedelta(hours=close_h, minutes=close_m) for t in idx]
        )
        # The last hourly bar of a US session is a 30-minute stub; never let the
        # synthetic close time run past the bell.
        idx = pd.DatetimeIndex(np.minimum(shifted.values, cap.values)).tz_localize("UTC")
        idx = idx.tz_convert(ET)

    out.index = idx
    out.index.name = "bar_close"
    return out[~out.index.isna()]


# ---------------------------------------------------------------------------
# yfinance loaders
# ---------------------------------------------------------------------------

def _download(tickers, period_days: int, interval: str, retries: int = 3):
    import yfinance as yf

    period = f"{int(period_days)}d"
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                prepost=False,
                progress=False,
                threads=True,
                group_by="column",
            )
            if df is not None and len(df):
                return df
            last_err = "empty frame"
        except Exception as e:  # network flakiness is the norm here
            last_err = e
        if attempt < retries - 1:
            _time.sleep(2 ** attempt)
    print(f"  WARNING: download failed for {tickers} @{interval}: {last_err}")
    return pd.DataFrame()


def load_bars(tickers: list, interval: str, days: int, p_data) -> dict:
    """{ticker: normalised frame}. Missing tickers simply come back empty."""
    raw = _download(tickers, days, interval)
    return {t: normalize(raw, t, interval, p_data) for t in tickers}


def load_hourly(tickers, p_data) -> dict:
    days = min(p_data.hourly_lookback_days, p_data.max_hourly_lookback_days)
    return load_bars(tickers, "60m", days, p_data)


def load_daily(tickers, p_data) -> dict:
    return load_bars(tickers, "1d", p_data.daily_lookback_days, p_data)


def load_stop_bars(tickers, p_data, interval="5m") -> dict:
    days = min(p_data.intraday_lookback_days, p_data.max_intraday_lookback_days)
    return load_bars(tickers, interval, days, p_data)


def load_option_snapshot(ticker: str, p_option, spot: float,
                         max_spread_pct: float = 5.0, asof=None) -> dict:
    """
    Pull the live chain and pick the contract the rules ask for.

    Returns a dict with expiry/strike/bid/ask/mid, or {"ok": False, "reason": ...}
    when the chain is unusable. Never raises — a missing chain is a skipped
    signal, not a crashed scan.
    """
    from . import options as opt

    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])
        if not expiries:
            return {"ok": False, "reason": "no expiries returned"}

        pick = opt.select_expiry(expiries, asof or datetime.now().date(), p_option)
        if pick is None:
            return {"ok": False,
                    "reason": f"no expiry in the {p_option.min_dte}-{p_option.max_dte} day window"}

        chain = tk.option_chain(pick["expiry"].isoformat())
        calls = chain.calls
        if calls is None or calls.empty:
            return {"ok": False, "reason": "empty call chain"}

        strike = opt.select_strike(calls["strike"].tolist(), spot, p_option)
        if strike is None:
            return {"ok": False, "reason": "no strike above spot"}

        row = calls.loc[calls["strike"] == strike].iloc[0]
        bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
        quality = opt.quote_quality(bid, ask, max_spread_pct)
        return {
            "ok": True,
            "expiry": pick["expiry"].isoformat(),
            "dte": pick["dte"],
            "strike": float(strike),
            "bid": bid,
            "ask": ask,
            "mid": quality["mid"],
            "spread_pct": quality["spread_pct"],
            "spread_ok": quality["ok"],
            "spread_reason": quality["reason"],
            "open_interest": int(row.get("openInterest") or 0),
            "volume": int(row.get("volume") or 0),
            "implied_vol": round(float(row.get("impliedVolatility") or 0), 4),
            "symbol": opt.occ_symbol(ticker, pick["expiry"], float(strike)),
        }
    except Exception as e:
        return {"ok": False, "reason": f"chain fetch failed: {e}"}


# ---------------------------------------------------------------------------
# Offline fixtures — lets the whole pipeline run with no network at all
# ---------------------------------------------------------------------------

US_HOURLY_CLOSES_ET = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00")


def session_index(n_bars: int, start_date: str = "2026-05-01",
                  closes=US_HOURLY_CLOSES_ET) -> pd.DatetimeIndex:
    """
    A realistic US equity hourly index: bar CLOSE times, weekdays only.

    Regular-hours hourly bars run 09:30-10:30, 10:30-11:30, ... and the last one
    is a 30-minute stub closing on the bell, which is why 16:00 is in the list.
    """
    stamps, day = [], pd.Timestamp(start_date)
    while len(stamps) < n_bars:
        if day.weekday() < 5:
            for hhmm in closes:
                h, m = (int(x) for x in hhmm.split(":"))
                stamps.append(day.normalize() + timedelta(hours=h, minutes=m))
                if len(stamps) >= n_bars:
                    break
        day += timedelta(days=1)
    return pd.DatetimeIndex(stamps).tz_localize(ET)


def synthetic_series(
    n_bars: int = 420,
    start: str = "2026-05-01 09:30",
    interval: str = "60m",
    seed: int = 7,
    base: float = 100.0,
    squeeze_at: tuple = (330, 30),
    breakout_size: float = 0.04,
    post_drift: float = 0.002,
) -> pd.DataFrame:
    """
    Bars that deliberately contain a compression -> breakout -> fade episode.

    The quiet phase is *mean reverting*, not merely small-stepped, because that
    is what the Saty compression test actually keys on: the 21-bar standard
    deviation of closes falling below the 14-bar ATR. A low-volatility random
    walk does not do that (its closes keep wandering); a range-bound chop does,
    because the closes keep coming back while the bars still have range.

    Used by the tests and by `bilbo_scan.py --offline` so the whole pipeline can
    be exercised with no data feed at all.
    """
    rng = np.random.default_rng(seed)
    idx = (session_index(n_bars, str(start)[:10]) if interval in ("60m", "1h")
           else pd.date_range(start=start, periods=n_bars,
                              freq=f"{interval_minutes(interval)}min", tz=ET))
    quiet_start, quiet_len = squeeze_at
    quiet_end = quiet_start + quiet_len
    amp = base * 0.004                      # half-width of the consolidation

    closes, price, level = [], base, None
    for i in range(n_bars):
        if i == quiet_start:
            level = price
        if quiet_start <= i < quiet_end:
            # pull hard back to the level: closes cluster, ranges do not
            price = level + 0.25 * (price - level) + rng.normal(0, amp)
        elif quiet_end <= i < quiet_end + 3 and breakout_size:
            price = price + base * breakout_size / 3.0
        elif i < quiet_start:
            price = max(1.0, price + rng.normal(base * 0.0003, base * 0.005))
        else:
            # post-breakout drift with a fade, so exits have something to bite
            drift = base * post_drift if i < quiet_end + 12 else -base * post_drift * 2
            price = max(1.0, price + drift + rng.normal(0, base * 0.004))
        closes.append(price)

    close = pd.Series(closes, index=idx)
    # bar ranges stay healthy through the quiet phase — that is the whole point
    rng_width = pd.Series(
        [amp * 1.5 if quiet_start <= i < quiet_end
         else 0.35 * max(base * 0.003, abs(close.iloc[i] - close.iloc[max(0, i - 1)]))
         for i in range(n_bars)],
        index=idx,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum.reduce([close.values, open_.values]) + rng_width.values * rng.uniform(0.3, 0.9, n_bars)
    low = np.minimum.reduce([close.values, open_.values]) - rng_width.values * rng.uniform(0.3, 0.9, n_bars)
    return pd.DataFrame(
        {"open": open_.values, "high": high, "low": low, "close": close.values,
         "volume": rng.integers(1_000_000, 5_000_000, n_bars)},
        index=idx,
    )


def daily_from_intraday(bars: pd.DataFrame) -> pd.DataFrame:
    """Roll intraday bars up into daily bars (fixtures and backtest fallbacks)."""
    if bars.empty:
        return bars
    g = bars.groupby(pd.DatetimeIndex(bars.index).tz_localize(None).normalize())
    out = g.agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"))
    out.index.name = "date"
    return out
