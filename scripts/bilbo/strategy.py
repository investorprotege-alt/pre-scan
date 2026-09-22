"""
strategy.py — the gates that turn a breakout into a trade, and the exit engine.

Two ideas do all the work here:

1. A breakout is only tradeable inside a narrow set of conditions (time of day,
   trend filter, option liquidity). Everything else is a *missed* signal, which
   this system records rather than hides — the misses are how you audit whether
   the gates are earning their keep.

2. Exits are keyed to the STOCK price, never the option price. The option is
   just the instrument you expressed the view with; the thesis lives or dies on
   the stock. That single choice is what stops theta and IV noise from shaking
   you out of a thesis that is still intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import time
from typing import Optional

import pandas as pd

from . import indicators


# ---------------------------------------------------------------------------
# Entry gates
# ---------------------------------------------------------------------------

def _parse_hhmm(text: str) -> time:
    h, m = str(text).split(":")[:2]
    return time(int(h), int(m))


def in_entry_window(ts: pd.Timestamp, p) -> bool:
    """True when the bar's CLOSE falls inside the session window (default 10:00-15:00 ET)."""
    local = ts.tz_convert(p.session_timezone) if ts.tzinfo else ts
    t = local.time()
    return _parse_hhmm(p.window_start) <= t <= _parse_hhmm(p.window_end)


def daily_ema_series(daily: pd.DataFrame, p) -> pd.Series:
    """
    Daily EMA of the configured length, optionally shifted one session so an
    intraday decision only ever uses closes that were already on the tape.
    """
    e = indicators.ema(daily["close"].astype(float), p.daily_ema_length)
    return e.shift(1) if p.daily_ema_uses_prior_close else e


def align_to_index(ts, idx) -> pd.Timestamp:
    """Make a bar timestamp comparable with a (possibly tz-naive) daily index."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(idx.tz) if getattr(idx, "tz", None) is not None \
            else t.tz_convert("America/New_York").tz_localize(None)
    elif getattr(idx, "tz", None) is not None:
        t = t.tz_localize(idx.tz)
    return t.normalize()


def series_at(series: pd.Series, ts: pd.Timestamp) -> Optional[float]:
    """The most recent value of a daily series that was known at bar time `ts`."""
    if series is None or len(series) == 0:
        return None
    day = align_to_index(ts, series.index)
    prior = series.loc[:day].dropna()
    return float(prior.iloc[-1]) if len(prior) else None


def ema_at(ema_daily: pd.Series, ts: pd.Timestamp) -> Optional[float]:
    """The most recent daily EMA value known at bar time `ts`."""
    return series_at(ema_daily, ts)


def make_gate_fn(ema_daily: pd.Series, p):
    """Build the gate callback the box machine calls on every breakout bar."""
    def gate_fn(ts, bar, box) -> dict:
        gates = {"time_window": in_entry_window(ts, p)}
        if p.require_above_daily_ema:
            ema_val = ema_at(ema_daily, ts)
            gates["above_daily_ema"] = (
                ema_val is not None and float(bar["close"]) > ema_val
            )
        return gates
    return gate_fn


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@dataclass
class Position:
    id: str
    ticker: str
    entry_ts: str
    entry_stock: float
    box_high: float
    box_low: float
    daily_atr: float
    contract: dict = field(default_factory=dict)
    status: str = "open"            # open | closed
    peak_stock: float = 0.0
    trail_armed: bool = False
    exit_ts: Optional[str] = None
    exit_stock: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_seen: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Position":
        known = {f for f in Position.__dataclass_fields__}
        return Position(**{k: v for k, v in d.items() if k in known})


def position_id(ticker: str, entry_ts) -> str:
    """Stable id so re-running the scanner never double-opens a trade."""
    stamp = pd.Timestamp(entry_ts).strftime("%Y%m%dT%H%M")
    return f"{ticker}-{stamp}"


# ---------------------------------------------------------------------------
# Exit engine
# ---------------------------------------------------------------------------

@dataclass
class ExitState:
    should_exit: bool
    reason: Optional[str]
    ts: Optional[pd.Timestamp]
    price: Optional[float]
    peak_stock: float
    trail_armed: bool
    trail_level: Optional[float]
    arm_level: float
    trading_days_held: int
    bars_checked: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat() if self.ts is not None else None
        for k in ("price", "peak_stock", "trail_level", "arm_level"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 4)
        return d


def evaluate_exit(pos: Position, bars: pd.DataFrame, p) -> ExitState:
    """
    Walk the stop-interval bars that came after entry and decide.

    Priority inside a single bar, hardest rule first:
      1. close below the box low          -> the breakout was a lie, get out
      2. update the running peak
      3. arm the trail once +1 daily ATR has been earned
      4. once armed, close giving back >75% of the peak gain -> take what's left
      5. time stop after N trading days

    All four are measured on the stock. The option is never consulted.
    """
    entry = float(pos.entry_stock)
    peak = max(float(pos.peak_stock or 0.0), entry)
    armed = bool(pos.trail_armed)
    arm_level = entry + p.trail_arm_atr_multiple * float(pos.daily_atr or 0.0)
    keep = 1.0 - (p.trail_giveback_pct / 100.0)

    entry_ts = pd.Timestamp(pos.entry_ts)
    after = bars.loc[bars.index > entry_ts] if len(bars) else bars
    entry_day = entry_ts.date()
    seen_days: set = set()
    days_held = 0
    checked = 0
    trail_level = entry + keep * (peak - entry) if armed else None

    for ts, bar in after.iterrows():
        checked += 1
        close = float(bar["close"])
        day = ts.date()
        if day != entry_day:
            seen_days.add(day)
        days_held = len(seen_days)

        if close < pos.box_low:
            return ExitState(True, "box_low_invalidation", ts, close, peak, armed,
                             trail_level, arm_level, days_held, checked)

        if close > peak:
            peak = close

        if not armed and peak >= arm_level:
            armed = True

        if armed:
            trail_level = entry + keep * (peak - entry)
            if close < trail_level:
                return ExitState(True, "trail_giveback", ts, close, peak, armed,
                                 trail_level, arm_level, days_held, checked)

        if days_held >= p.time_stop_trading_days:
            return ExitState(True, "time_stop", ts, close, peak, armed,
                             trail_level, arm_level, days_held, checked)

    last_price = float(after["close"].iloc[-1]) if len(after) else entry
    return ExitState(False, None, None, last_price, peak, armed,
                     trail_level, arm_level, days_held, checked)


# ---------------------------------------------------------------------------
# Live view for the dashboard
# ---------------------------------------------------------------------------

@dataclass
class TickerView:
    ticker: str
    name: str
    status: str
    last_price: Optional[float] = None
    last_bar: Optional[str] = None
    oscillator: Optional[float] = None
    phase: Optional[str] = None
    compressed: bool = False
    compression_run: int = 0
    box: Optional[dict] = None
    distance_to_trigger_pct: Optional[float] = None
    above_daily_ema: Optional[bool] = None
    daily_ema: Optional[float] = None
    daily_atr: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_view(ticker, name, bars, osc, live_box, ema_daily, daily_atr_val, p_entry, p_box) -> TickerView:
    """Summarise where one ticker stands right now, in plain language."""
    if bars.empty:
        return TickerView(ticker=ticker, name=name, status="no data")

    last_ts = bars.index[-1]
    last = bars.iloc[-1]
    last_close = float(last["close"])
    o = osc.iloc[-1]

    run = 0
    for flag in reversed(osc["compressed"].tolist()):
        if flag:
            run += 1
        else:
            break

    ema_val = ema_at(ema_daily, last_ts)
    view = TickerView(
        ticker=ticker,
        name=name,
        status="idle",
        last_price=round(last_close, 4),
        last_bar=last_ts.isoformat(),
        oscillator=None if pd.isna(o["oscillator"]) else float(o["oscillator"]),
        phase=indicators.phase_label(o["oscillator"]),
        compressed=bool(o["compressed"]),
        compression_run=run,
        above_daily_ema=None if ema_val is None else last_close > ema_val,
        daily_ema=None if ema_val is None else round(ema_val, 4),
        daily_atr=None if daily_atr_val is None else round(float(daily_atr_val), 4),
    )

    if live_box is not None:
        view.box = live_box.to_dict()
        view.status = live_box.state
        if live_box.state == "armed":
            view.distance_to_trigger_pct = round(
                (live_box.high - last_close) / last_close * 100.0, 3
            )
            if view.compressed:
                view.note = (
                    f"box frozen at {live_box.low:.2f}-{live_box.high:.2f} but still inside "
                    f"compression — no entry until a candle closes out of the grey"
                )
            else:
                view.note = (
                    f"armed: needs an hourly close above {live_box.high:.2f} "
                    f"({view.distance_to_trigger_pct:+.2f}% away) between "
                    f"{p_entry.window_start} and {p_entry.window_end} ET, "
                    f"with price above the daily {p_entry.daily_ema_length} EMA"
                )
        else:
            view.note = (
                f"building box: {live_box.bar_count}/{p_box.max_box_bars} grey candles so far, "
                f"range {live_box.low:.2f}-{live_box.high:.2f}"
            )
    elif view.compressed:
        view.status = "compressing"
        view.note = f"{run} grey candle(s) in a row — a box is forming"
    else:
        view.note = "no compression — nothing to do"

    return view
