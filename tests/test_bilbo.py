"""
Tests for the Bilbo Box engine.

The oscillator tests re-derive the ThinkScript recursions with plain Python
loops rather than calling the implementation, so a refactor that changes the
maths has to fail here.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from bilbo import boxes, data as bdata, indicators, options as bopt, strategy
from bilbo.config import (BoxParams, DataParams, EntryParams, ExitParams,
                          OptionParams, OscillatorParams, load_config)

ET = "America/New_York"


def frame(closes, highs=None, lows=None, start="2026-05-01 10:30"):
    idx = pd.date_range(start=start, periods=len(closes), freq="60min", tz=ET)
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else closes + 0.5
    lows = np.asarray(lows, dtype=float) if lows is not None else closes - 0.5
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": 1_000}, index=idx)


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

def test_ema_matches_thinkscript_recursion():
    s = pd.Series([10, 11, 12, 11, 13, 14, 13], dtype=float)
    alpha = 2 / (3 + 1)
    want, prev = [], None
    for v in s:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        want.append(prev)
    assert indicators.ema(s, 3).round(10).tolist() == pytest.approx(want)


def test_wilder_rma_uses_one_over_n():
    s = pd.Series([5, 7, 6, 9], dtype=float)
    want, prev = [], None
    for v in s:
        prev = v if prev is None else prev + (v - prev) / 4
        want.append(prev)
    assert indicators.wilder_rma(s, 4).tolist() == pytest.approx(want)


def test_true_range_first_bar_has_no_prior_close():
    df = frame([100, 101], highs=[101, 102], lows=[99, 100.5])
    tr = indicators.true_range(df)
    assert tr.iloc[0] == pytest.approx(2.0)              # high - low
    assert tr.iloc[1] == pytest.approx(2.0)              # high - prior close


def test_compression_is_bollinger_inside_the_atr_envelope():
    """compression <= 0 means the 2-sigma band sits inside the 2-ATR envelope."""
    p = OscillatorParams()
    df = bdata.synthetic_series(seed=5)
    osc = indicators.phase_oscillator(df, p)
    above = df["close"] >= osc["pivot"]
    expected = np.where(above, osc["bb_up"] - osc["kc_up"], osc["kc_down"] - osc["bb_down"])
    assert osc["compression"].values == pytest.approx(expected, nan_ok=True)
    # and the tracker never fires while the bands are outside the envelope
    assert not (osc["compressed"] & (osc["compression"] > 0)).any()


def test_oscillator_is_distance_from_pivot_in_atr_units():
    p = OscillatorParams()
    df = bdata.synthetic_series(seed=11)
    osc = indicators.phase_oscillator(df, p)
    raw = ((df["close"] - osc["pivot"]) / (p.oscillator_atr_multiple * osc["atr"])) * 100
    assert osc["oscillator"].values == pytest.approx(
        indicators.ema(raw, p.oscillator_smoothing).round(2).values, nan_ok=True)
    # +/-100 is one full 3-ATR displacement from the 21 EMA
    at_100 = osc["oscillator"].abs() >= 100
    if at_100.any():
        far = (df["close"] - osc["pivot"]).abs() / osc["atr"]
        assert far[at_100].min() > 2.0


def test_compressed_flag_is_false_before_the_bollinger_window_fills():
    osc = indicators.phase_oscillator(bdata.synthetic_series(seed=2), OscillatorParams())
    assert not osc["compressed"].iloc[:20].any()


def test_phase_labels_track_the_saty_zones():
    assert indicators.phase_label(120) == "extended up"
    assert indicators.phase_label(70) == "distribution"
    assert indicators.phase_label(0) == "launch box"
    assert indicators.phase_label(-80) == "accumulation"
    assert indicators.phase_label(-140) == "extended down"


# ---------------------------------------------------------------------------
# box state machine
# ---------------------------------------------------------------------------

def grey(flags, index):
    return pd.DataFrame({"compressed": pd.Series(list(flags), index=index, dtype=bool)},
                        index=index)


def test_box_freezes_after_five_grey_candles_and_ignores_later_ones():
    closes = [100, 100, 100, 100, 100, 100, 100, 100]
    highs = [101, 102, 101, 100.5, 101, 110, 110, 110]   # bar 6 is grey but wider
    lows = [99, 98, 99, 99.5, 99, 90, 90, 90]
    df = frame(closes, highs, lows)
    osc = grey([True] * 7 + [False], df.index)
    events, live = boxes.run_box_machine(df, osc, BoxParams(), "T")
    assert live.state == boxes.ARMED
    assert live.bar_count == 5
    assert live.high == 102 and live.low == 98          # bars 6-7 did not widen it
    assert [e.kind for e in events] == ["frozen"]


def test_entry_requires_a_close_out_of_compression_above_the_box_high():
    df = frame([100, 100, 100, 100, 100, 103, 105],
               highs=[101, 101, 101, 101, 101, 104, 106],
               lows=[99, 99, 99, 99, 99, 102, 104])
    # bar 5 closes above the box high but is still grey -> no entry
    osc = grey([True] * 6 + [False], df.index)
    events, live = boxes.run_box_machine(df, osc, BoxParams(), "T")
    kinds = [e.kind for e in events]
    assert kinds == ["frozen", boxes.TRIGGERED]
    assert events[1].ts == df.index[6]                   # the first non-grey bar
    assert events[1].box.high == 101


def test_short_grey_run_still_makes_a_box_when_long_enough():
    df = frame([100, 100, 100, 104], highs=[101, 102, 101, 105], lows=[99, 99, 99, 103])
    osc = grey([True, True, True, False], df.index)
    events, _ = boxes.run_box_machine(df, osc, BoxParams(min_box_bars=2), "T")
    assert [e.kind for e in events] == ["frozen", boxes.TRIGGERED]


def test_grey_run_shorter_than_min_box_bars_is_discarded():
    df = frame([100, 104], highs=[101, 105], lows=[99, 103])
    osc = grey([True, False], df.index)
    events, live = boxes.run_box_machine(df, osc, BoxParams(min_box_bars=2), "T")
    assert events == [] and live is None


def test_close_below_box_low_kills_the_box():
    df = frame([100] * 5 + [96], highs=[101] * 5 + [97], lows=[99] * 5 + [95])
    osc = grey([True] * 5 + [False], df.index)
    events, live = boxes.run_box_machine(df, osc, BoxParams(), "T")
    assert [e.kind for e in events] == ["frozen", boxes.INVALIDATED]
    assert live is None


def test_new_compression_supersedes_a_stale_armed_box():
    df = frame([100] * 5 + [100.2, 100.1], highs=[101] * 7, lows=[99] * 7)
    osc = grey([True] * 5 + [False, True], df.index)
    events, live = boxes.run_box_machine(df, osc, BoxParams(), "T")
    assert [e.kind for e in events] == ["frozen", boxes.SUPERSEDED]
    assert live.state == boxes.BUILDING and live.bar_count == 1


def test_armed_box_expires():
    n = 5 + 8
    df = frame([100] * n, highs=[101] * n, lows=[99] * n)
    osc = grey([True] * 5 + [False] * 8, df.index)
    events, live = boxes.run_box_machine(df, osc, BoxParams(box_expiry_bars=4), "T")
    assert boxes.EXPIRED in [e.kind for e in events]
    assert live is None


def test_failed_gate_records_a_miss_not_a_trade():
    df = frame([100] * 5 + [105], highs=[101] * 5 + [106], lows=[99] * 5 + [104])
    osc = grey([True] * 5 + [False], df.index)
    events, _ = boxes.run_box_machine(df, osc, BoxParams(), "T",
                                      gate_fn=lambda ts, bar, box: {"time_window": False})
    assert events[-1].kind == boxes.MISSED
    assert events[-1].gates == {"time_window": False}


# ---------------------------------------------------------------------------
# entry gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hhmm,ok", [("10:30", True), ("14:30", True), ("15:00", True),
                                     ("09:30", False), ("15:30", False), ("16:00", False)])
def test_entry_window_is_judged_on_the_bar_close(hhmm, ok):
    ts = pd.Timestamp(f"2026-05-04 {hhmm}", tz=ET)
    assert strategy.in_entry_window(ts, EntryParams()) is ok


def test_daily_ema_gate_uses_the_prior_session_close():
    idx = pd.date_range("2026-05-01", periods=30, freq="B")
    daily = pd.DataFrame({"close": np.linspace(90, 120, 30)}, index=idx)
    p = EntryParams(daily_ema_length=5, daily_ema_uses_prior_close=True)
    shifted = strategy.daily_ema_series(daily, p)
    unshifted = strategy.daily_ema_series(daily, EntryParams(daily_ema_length=5,
                                                             daily_ema_uses_prior_close=False))
    assert shifted.iloc[10] == pytest.approx(unshifted.iloc[9])
    ts = pd.Timestamp("2026-05-20 11:30", tz=ET)
    assert strategy.ema_at(shifted, ts) == pytest.approx(shifted.loc[:"2026-05-20"].dropna().iloc[-1])


# ---------------------------------------------------------------------------
# exits
# ---------------------------------------------------------------------------

def position(entry=100.0, box_low=98.0, atr=2.0, ts="2026-05-04 11:30"):
    return strategy.Position(
        id="T-1", ticker="T", entry_ts=pd.Timestamp(ts, tz=ET).isoformat(),
        entry_stock=entry, box_high=101.0, box_low=box_low, daily_atr=atr,
        peak_stock=entry)


def bars_after(prices, start="2026-05-04 11:35", freq="5min"):
    idx = pd.date_range(start=start, periods=len(prices), freq=freq, tz=ET)
    return pd.DataFrame({"open": prices, "high": prices, "low": prices,
                         "close": prices, "volume": 1}, index=idx)


def test_exit_on_five_minute_close_below_the_box_low():
    state = strategy.evaluate_exit(position(), bars_after([100.5, 99.5, 97.9, 105]), ExitParams())
    assert state.should_exit and state.reason == "box_low_invalidation"
    assert state.price == pytest.approx(97.9)


def test_trail_arms_only_after_one_daily_atr():
    p = ExitParams()
    # peak +1.5 never reaches the +2.0 ATR arming level
    state = strategy.evaluate_exit(position(), bars_after([101.0, 101.5, 100.2]), p)
    assert not state.trail_armed and not state.should_exit
    assert state.arm_level == pytest.approx(102.0)


def test_trail_exits_after_giving_back_three_quarters_of_the_peak_gain():
    p = ExitParams()
    # peak 104 -> gain 4 -> keep 25% -> exit below 101.0
    state = strategy.evaluate_exit(position(), bars_after([102.5, 104.0, 103.0, 100.9]), p)
    assert state.should_exit and state.reason == "trail_giveback"
    assert state.trail_level == pytest.approx(101.0)
    assert state.peak_stock == pytest.approx(104.0)


def test_a_higher_close_moves_the_trail_up_and_does_not_exit():
    state = strategy.evaluate_exit(position(), bars_after([103.0, 106.0, 108.0]), ExitParams())
    assert not state.should_exit
    assert state.trail_armed
    assert state.trail_level == pytest.approx(102.0)     # 100 + 0.25 * 8


def test_time_stop_counts_trading_days_not_bars():
    p = ExitParams(time_stop_trading_days=3)
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-05-{d:02d} 12:30", tz=ET)
                            for d in (4, 5, 6, 7, 8)])
    bars = pd.DataFrame({"open": 100.5, "high": 100.5, "low": 100.5,
                         "close": [100.5] * 5, "volume": 1}, index=idx)
    state = strategy.evaluate_exit(position(ts="2026-05-04 11:30"), bars, p)
    assert state.should_exit and state.reason == "time_stop"
    assert state.ts.date() == pd.Timestamp("2026-05-07").date()   # day 3 after entry


def test_box_low_beats_the_trail_inside_the_same_bar():
    # gain, then a collapse straight through the box low
    state = strategy.evaluate_exit(position(), bars_after([104.0, 97.0]), ExitParams())
    assert state.reason == "box_low_invalidation"


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

def test_strike_selection_walks_the_real_ladder():
    p = OptionParams(strikes_otm=1)
    assert bopt.select_strike([95, 100, 102.5, 105], 101.0, p) == 102.5
    assert bopt.select_strike([95, 100, 102.5, 105], 101.0, OptionParams(strikes_otm=2)) == 105
    assert bopt.select_strike([95, 100], 101.0, p) is None


def test_expiry_picks_the_nearest_to_target_inside_the_window():
    p = OptionParams(target_dte=28, min_dte=21, max_dte=37)
    asof = pd.Timestamp("2026-05-01").date()
    expiries = ["2026-05-08", "2026-05-22", "2026-05-29", "2026-06-05", "2026-07-17"]
    pick = bopt.select_expiry(expiries, asof, p)
    assert pick["expiry"].isoformat() == "2026-05-29" and pick["dte"] == 28
    assert bopt.select_expiry(["2026-05-08"], asof, p) is None


def test_spread_gate_rejects_wide_markets():
    assert bopt.quote_quality(1.00, 1.04, 5.0)["ok"]
    bad = bopt.quote_quality(1.00, 1.20, 5.0)
    assert not bad["ok"] and "18" in bad["reason"]
    assert not bopt.quote_quality(0, 1.2, 5.0)["ok"]


def test_sizing_lands_inside_the_premium_band():
    p = OptionParams()
    s = bopt.size_position(premium=4.20, equity=100_000, p=p)
    assert p.premium_pct_min <= s["pct_of_account"] <= p.premium_pct_max
    assert s["contracts"] * 4.20 * 100 == pytest.approx(s["cost"])


def test_sizing_refuses_when_one_contract_blows_the_cap():
    s = bopt.size_position(premium=80.0, equity=100_000, p=OptionParams())
    assert s["contracts"] == 0 and "above the" in s["reason"]


def test_black_scholes_sanity():
    call = bopt.bs_call(100, 105, 28 / 365, 0.04, 0.30)
    assert 1.0 < call < 5.0
    assert bopt.bs_call(100, 105, 0, 0.04, 0.3) == 0.0             # expired OTM
    assert bopt.bs_call(110, 105, 0, 0.04, 0.3) == pytest.approx(5.0)
    deeper = bopt.bs_call(100, 105, 28 / 365, 0.04, 0.60)
    assert deeper > call                                            # vega is positive


# ---------------------------------------------------------------------------
# data normalisation
# ---------------------------------------------------------------------------

def test_bars_are_restamped_to_their_close_and_clamped_to_the_bell():
    opens = pd.DatetimeIndex([pd.Timestamp(f"2026-05-04 {t}", tz=ET)
                              for t in ("09:30", "14:30", "15:30")])
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
                       index=opens)
    out = bdata.normalize(raw, "T", "60m", DataParams())
    got = [t.strftime("%H:%M") for t in out.index]
    assert got == ["10:30", "15:30", "16:00"]       # the stub bar stops at the bell


def test_one_ticker_is_pulled_out_of_a_batch_download():
    idx = pd.date_range("2026-05-01", periods=3, freq="B")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"],
                                       ["AAPL", "MSFT"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw[("Close", "MSFT")] = 2.0
    assert bdata.normalize(raw, "MSFT", "1d", DataParams())["close"].tolist() == [2.0] * 3
    assert bdata.normalize(raw, "AAPL", "1d", DataParams())["close"].tolist() == [1.0] * 3
    # a ticker the batch has no data for comes back empty, never mangled
    assert bdata.normalize(raw, "NVDA", "1d", DataParams()).empty


def test_daily_bars_are_not_restamped():
    idx = pd.date_range("2026-05-01", periods=3, freq="B")
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}, index=idx)
    out = bdata.normalize(raw, "T", "1d", DataParams())
    assert list(out.index) == list(idx)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_session_index_only_contains_weekday_session_closes():
    idx = bdata.session_index(30)
    assert all(t.weekday() < 5 for t in idx)
    assert {t.strftime("%H:%M") for t in idx} <= set(bdata.US_HOURLY_CLOSES_ET)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_shipped_config_loads_and_is_internally_consistent():
    cfg = load_config()
    assert len(cfg.tickers) >= 5
    assert cfg.option.min_dte <= cfg.option.target_dte <= cfg.option.max_dte
    assert cfg.option.premium_pct_min <= cfg.option.premium_pct_target <= cfg.option.premium_pct_max
    assert cfg.box.min_box_bars <= cfg.box.max_box_bars
    assert 0 < cfg.exit.trail_giveback_pct < 100
    assert cfg.oscillator.expansion_atr_mult < cfg.oscillator.compression_atr_mult


# ---------------------------------------------------------------------------
# scanner + journal (the stateful part)
# ---------------------------------------------------------------------------

def _signal(ts, price=102.0):
    return {
        "id": strategy.position_id("AAPL", ts), "ticker": "AAPL", "name": "Apple",
        "ts": pd.Timestamp(ts).isoformat(), "ts_et": "x", "fresh": True,
        "status": "actionable", "stock_price": price, "box_high": 101.0,
        "box_low": 98.0, "box_bars": 5, "box_height_pct": 3.0, "gates": {},
        "daily_atr": 2.0,
        "option": {"ok": True, "symbol": "AAPL 2026-06-05 C105", "mid": 3.10},
        "sizing": {"contracts": 12, "cost": 3720.0, "pct_of_account": 3.72},
    }


def test_journal_opens_holds_and_closes_one_position(tmp_path, monkeypatch):
    import bilbo_scan

    monkeypatch.setattr(bilbo_scan, "DATA_DIR", str(tmp_path))
    cfg = load_config()

    bars = frame([100, 100, 100, 100, 100, 102, 102.5, 102.2])
    entry_ts = bars.index[5]
    result = {"signals": [_signal(entry_ts)], "hourly": {"AAPL": bars}}

    opened = bilbo_scan.update_journal(result, cfg, offline=True)
    assert [p["id"] for p in opened["open"]] == [strategy.position_id("AAPL", entry_ts)]
    assert opened["open"][0]["contract"]["symbol"] == "AAPL 2026-06-05 C105"
    assert opened["stats"]["trades"] == 0

    # re-running the same scan must not double-open
    again = bilbo_scan.update_journal(result, cfg, offline=True)
    assert len(again["open"]) == 1

    # now the stock closes under the box low
    killed = frame([100, 100, 100, 100, 100, 102, 102.5, 97.0])
    closed = bilbo_scan.update_journal({"signals": [], "hourly": {"AAPL": killed}},
                                       cfg, offline=True)
    assert closed["open"] == []
    assert closed["closed"][0]["exit_reason"] == "box_low_invalidation"
    assert closed["stats"]["trades"] == 1
    assert closed["stats"]["win_rate_pct"] == 0.0


def test_journal_respects_one_position_per_ticker(tmp_path, monkeypatch):
    import bilbo_scan

    monkeypatch.setattr(bilbo_scan, "DATA_DIR", str(tmp_path))
    cfg = load_config()
    bars = frame([100, 100, 100, 100, 100, 102, 102.5])
    first, second = bars.index[5], bars.index[6]

    bilbo_scan.update_journal({"signals": [_signal(first)], "hourly": {"AAPL": bars}},
                              cfg, offline=True)
    out = bilbo_scan.update_journal({"signals": [_signal(second, 102.5)],
                                     "hourly": {"AAPL": bars}}, cfg, offline=True)
    assert len(out["open"]) == 1


def test_scan_writes_all_three_files_offline(tmp_path, monkeypatch):
    import bilbo_scan

    monkeypatch.setattr(bilbo_scan, "DATA_DIR", str(tmp_path))
    state = bilbo_scan.run(["--offline", "--tickers", "AAPL,MSFT", "--lookback-bars", "20"])
    for name in ("bilbo_state.json", "bilbo_signals.json", "bilbo_positions.json"):
        assert (tmp_path / name).exists(), name
    assert state["data_mode"] == "offline-fixture"
    assert state["universe_size"] == 2
    assert all("note" in v for v in state["tickers"])
