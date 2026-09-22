"""
boxes.py — the hourly state machine that turns compression into a tradeable box.

Life of a box:

    NO_BOX ──grey candle──▶ BUILDING ──5 grey candles, or grey run ends──▶ ARMED
                                                                            │
                  ┌───────────────────┬──────────────┬─────────────────────┤
                  ▼                   ▼              ▼                     ▼
          close > high            close < low    N bars pass        new grey run
          (gates pass)            (long-only     with nothing       starts
             TRIGGERED            invalidation)    EXPIRED           SUPERSEDED
          (gates fail)             INVALIDATED
              MISSED

The box's high/low are the extremes of the first `max_box_bars` grey candles
and never move again once frozen. The breakout candle itself is never part of
the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import pandas as pd

BUILDING = "building"
ARMED = "armed"
TRIGGERED = "triggered"
MISSED = "missed"
INVALIDATED = "invalidated"
EXPIRED = "expired"
SUPERSEDED = "superseded"
DISCARDED = "discarded"

OPEN_STATES = (BUILDING, ARMED)


@dataclass
class Box:
    ticker: str
    start_ts: pd.Timestamp
    high: float
    low: float
    bar_count: int = 1
    state: str = BUILDING
    freeze_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    end_reason: Optional[str] = None
    trigger_price: Optional[float] = None
    run_active: bool = True     # is the grey run that built this box still going?
    gates: dict = field(default_factory=dict)

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def height_pct(self) -> float:
        return (self.high - self.low) / self.low * 100.0 if self.low else 0.0

    def extend(self, bar) -> None:
        self.high = max(self.high, float(bar["high"]))
        self.low = min(self.low, float(bar["low"]))
        self.bar_count += 1

    def freeze(self, ts) -> None:
        if self.freeze_ts is None:
            self.freeze_ts = ts

    def close_out(self, ts, reason: str) -> None:
        self.state = reason
        self.end_ts = ts
        self.end_reason = reason

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start_ts", "freeze_ts", "end_ts"):
            d[k] = d[k].isoformat() if d[k] is not None else None
        d["height"] = round(self.height, 4)
        d["height_pct"] = round(self.height_pct, 3)
        return d


@dataclass
class BoxEvent:
    """One thing that happened to one box on one bar."""
    ticker: str
    ts: pd.Timestamp
    kind: str              # triggered | missed | invalidated | expired | superseded | frozen
    box: Box
    close: float
    gates: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "ts": self.ts.isoformat(),
            "kind": self.kind,
            "close": round(float(self.close), 4),
            "box": self.box.to_dict(),
            "gates": self.gates,
        }


GateFn = Callable[[pd.Timestamp, pd.Series, Box], dict]


def run_box_machine(
    bars: pd.DataFrame,
    osc: pd.DataFrame,
    p,
    ticker: str = "?",
    gate_fn: Optional[GateFn] = None,
) -> tuple[list[BoxEvent], Optional[Box]]:
    """
    Walk hourly bars once, in order, and return (events, live_box).

    `bars`: lowercase OHLC frame, oldest first, indexed by bar close time.
    `osc` : output of indicators.phase_oscillator, same index.
    `p`   : BoxParams.
    `gate_fn`: given (ts, bar, box) returns a dict of gate name -> bool for a
               breakout bar. All True means TRIGGERED, otherwise MISSED. When
               omitted, every breakout is a trigger (used by unit tests).

    The walk is stateless with respect to previous runs: feed it the same bars
    and you get the same events, which is what makes the scanner idempotent.
    """
    events: list[BoxEvent] = []
    box: Optional[Box] = None
    compressed = osc["compressed"]

    for ts, bar in bars.iterrows():
        is_grey = bool(compressed.get(ts, False))

        # ---- BUILDING ---------------------------------------------------
        if box is not None and box.state == BUILDING:
            if is_grey:
                if box.bar_count < p.max_box_bars:
                    box.extend(bar)
                if box.bar_count >= p.max_box_bars:
                    box.freeze(ts)
                    box.state = ARMED
                    events.append(BoxEvent(ticker, ts, "frozen", box, float(bar["close"])))
                continue
            # grey run ended early — freeze what we have
            box.run_active = False
            if box.bar_count >= p.min_box_bars:
                box.freeze(ts)
                box.state = ARMED
                events.append(BoxEvent(ticker, ts, "frozen", box, float(bar["close"])))
            else:
                box.close_out(ts, DISCARDED)
                box = None
            # fall through: this same bar may be the breakout

        # ---- ARMED ------------------------------------------------------
        if box is not None and box.state == ARMED:
            close = float(bar["close"])

            if is_grey:
                if box.run_active:
                    # the run that built the box is simply still going; the box
                    # stays frozen and we wait for it to end
                    continue
                if p.new_compression_supersedes:
                    box.close_out(ts, SUPERSEDED)
                    events.append(BoxEvent(ticker, ts, SUPERSEDED, box, close))
                    box = Box(ticker=ticker, start_ts=ts,
                              high=float(bar["high"]), low=float(bar["low"]))
                continue

            box.run_active = False

            if close > box.high:
                gates = gate_fn(ts, bar, box) if gate_fn else {}
                passed = all(gates.values()) if gates else True
                box.trigger_price = close
                box.gates = gates
                kind = TRIGGERED if passed else MISSED
                box.close_out(ts, kind)
                events.append(BoxEvent(ticker, ts, kind, box, close, gates))
                box = None
                continue

            if p.kill_on_close_below_low and close < box.low:
                box.close_out(ts, INVALIDATED)
                events.append(BoxEvent(ticker, ts, INVALIDATED, box, close))
                box = None
                continue

            bars_since_freeze = len(bars.loc[box.freeze_ts:ts]) - 1
            if p.box_expiry_bars and bars_since_freeze >= p.box_expiry_bars:
                box.close_out(ts, EXPIRED)
                events.append(BoxEvent(ticker, ts, EXPIRED, box, close))
                box = None
            continue

        # ---- NO BOX -----------------------------------------------------
        if box is None and is_grey:
            box = Box(ticker=ticker, start_ts=ts,
                      high=float(bar["high"]), low=float(bar["low"]))
            if p.max_box_bars <= 1:
                box.freeze(ts)
                box.state = ARMED

    return events, box
