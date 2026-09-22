"""
config.py — every tunable in one typed place.

Defaults here are the *shipped* defaults; config/bilbo_config.json overrides
them. Each field carries a provenance tag in docs/BILBO_BOX_MANUAL.md:

  [SATY]  taken verbatim from the published Saty Phase Oscillator source
  [RULE]  a published Bilbo Box rule
  [MINE]  an engineering choice this implementation had to make because the
          published rules do not pin it down
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "bilbo_config.json",
)


@dataclass
class OscillatorParams:
    pivot_ema_length: int = 21          # [SATY]
    atr_length: int = 14                # [SATY]
    atr_method: str = "wilder"          # [SATY]
    oscillator_smoothing: int = 3       # [SATY]
    oscillator_atr_multiple: float = 3.0    # [SATY]
    bband_length: int = 21              # [SATY]
    bband_stdev_mult: float = 2.0       # [SATY]
    stdev_ddof: int = 0                 # [SATY] ThinkScript StDev is population
    compression_atr_mult: float = 2.0   # [SATY]
    expansion_atr_mult: float = 1.854   # [SATY]


@dataclass
class BoxParams:
    bar_interval: str = "60m"           # [RULE] boxes are built on hourly bars
    max_box_bars: int = 5               # [RULE] box = first 5 grey candles
    min_box_bars: int = 2               # [MINE] shortest run that still counts
    box_expiry_bars: int = 30           # [MINE] armed box goes stale after this
    kill_on_close_below_low: bool = True    # [MINE] long-only housekeeping
    new_compression_supersedes: bool = True  # [MINE]


@dataclass
class EntryParams:
    session_timezone: str = "America/New_York"   # [RULE]
    window_start: str = "10:00"                  # [RULE] 10:00-15:00 ET
    window_end: str = "15:00"                    # [RULE]
    require_close_out_of_compression: bool = True    # [RULE]
    require_above_daily_ema: bool = True         # [RULE]
    daily_ema_length: int = 21                   # [RULE]
    daily_ema_uses_prior_close: bool = True      # [MINE] avoids look-ahead
    max_spread_pct_of_mid: float = 5.0           # [RULE]
    one_position_per_ticker: bool = True         # [MINE]
    max_new_positions_per_day: int = 3           # [MINE] portfolio sanity


@dataclass
class OptionParams:
    target_dte: int = 28                 # [RULE]
    min_dte: int = 21                    # [RULE]
    max_dte: int = 37                    # [RULE]
    strikes_otm: int = 1                 # [RULE] ~1 strike out of the money
    premium_pct_min: float = 3.0         # [RULE]
    premium_pct_target: float = 4.0      # [MINE] midpoint of the 3-5% band
    premium_pct_max: float = 5.0         # [RULE]
    contract_multiplier: int = 100       # [RULE]
    risk_free_rate: float = 0.042        # [MINE] backtest pricing only


@dataclass
class ExitParams:
    stop_bar_interval: str = "5m"        # [RULE] exits are judged on 5-min closes
    trail_arm_atr_multiple: float = 1.0  # [RULE] arm after +1 daily ATR
    trail_giveback_pct: float = 75.0     # [RULE] exit on >75% giveback of peak
    time_stop_trading_days: int = 10     # [RULE]
    daily_atr_length: int = 14           # [MINE] the ATR used to arm the trail


@dataclass
class AccountParams:
    equity: float = 100_000.0
    currency: str = "USD"


@dataclass
class DataParams:
    hourly_lookback_days: int = 90       # [MINE] enough to seed a 21-EMA well
    daily_lookback_days: int = 400       # [MINE]
    intraday_lookback_days: int = 5      # [MINE] 5-min bars for open positions
    max_hourly_lookback_days: int = 720  # yfinance hard limit for 60m bars
    max_intraday_lookback_days: int = 59  # yfinance hard limit for 5m bars
    index_is_bar_open: bool = True       # yfinance stamps bars with their OPEN time
    session_close_et: str = "16:00"      # clamp for the stub 15:30 hourly bar


@dataclass
class BacktestParams:
    iv_mode: str = "realized"            # "realized" | "fixed"
    fixed_iv: float = 0.32
    realized_vol_days: int = 30
    iv_premium_multiple: float = 1.15    # implied usually prints above realized
    iv_crush_on_exit: float = 0.0        # subtract this from IV at exit, in vol pts
    round_trip_spread_pct: float = 2.0   # % of mid, charged half on each side
    exit_bar_interval: str = "60m"       # 5m history is only ~60 days deep


@dataclass
class BilboConfig:
    universe: list = field(default_factory=list)
    oscillator: OscillatorParams = field(default_factory=OscillatorParams)
    box: BoxParams = field(default_factory=BoxParams)
    entry: EntryParams = field(default_factory=EntryParams)
    option: OptionParams = field(default_factory=OptionParams)
    exit: ExitParams = field(default_factory=ExitParams)
    account: AccountParams = field(default_factory=AccountParams)
    data: DataParams = field(default_factory=DataParams)
    backtest: BacktestParams = field(default_factory=BacktestParams)

    @property
    def tickers(self) -> list:
        return [u["ticker"] if isinstance(u, dict) else u for u in self.universe]

    def name_for(self, ticker: str) -> str:
        for u in self.universe:
            if isinstance(u, dict) and u.get("ticker") == ticker:
                return u.get("name", ticker)
        return ticker


def _build(cls, blob: Any):
    """Populate a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(blob, dict):
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = [k for k in blob if k not in known and not k.startswith("_")]
    if unknown:
        print(f"  NOTE: ignoring unknown {cls.__name__} keys: {', '.join(unknown)}")
    return cls(**{k: v for k, v in blob.items() if k in known})


def load_config(path: str | None = None) -> BilboConfig:
    path = path or CONFIG_PATH
    blob = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
    cfg = BilboConfig(universe=blob.get("universe", []))
    for f in fields(BilboConfig):
        if f.name == "universe":
            continue
        sub = getattr(cfg, f.name)
        if is_dataclass(sub):
            setattr(cfg, f.name, _build(type(sub), blob.get(f.name, {})))
    return cfg
