#!/usr/bin/env python3
"""
bilbo_scan.py — the Bilbo Box scanner.

One pass over the universe answers four questions:

  1. Who is compressing right now?            (a box is forming)
  2. Who is armed?                            (box frozen, waiting on a breakout)
  3. Did anyone trigger on the last bar?      (act now — with the contract to buy)
  4. Do any open positions need to come off?  (stock-keyed exits on 5-min closes)

Outputs (all committed, all read by dashboard/index.html):
    data/bilbo_state.json      where every ticker stands
    data/bilbo_signals.json    triggers and misses, with contract + size
    data/bilbo_positions.json  the paper journal, open and closed

Nothing here places an order.

    python scripts/bilbo_scan.py                 # live scan
    python scripts/bilbo_scan.py --offline       # synthetic bars, no network
    python scripts/bilbo_scan.py --no-options    # skip chain fetches
    python scripts/bilbo_scan.py --equity 50000  # size against a different account
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilbo import boxes, data as bdata, indicators, options as bopt, strategy
from bilbo.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SYDNEY = pytz.timezone("Australia/Sydney")
ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def _et(ts) -> str:
    t = pd.Timestamp(ts)
    return t.tz_convert(ET).strftime("%Y-%m-%d %H:%M ET") if t.tzinfo else t.strftime("%Y-%m-%d")


def load_universe_bars(tickers, cfg, offline: bool):
    """Hourly + daily bars for the whole universe, live or synthetic."""
    if offline:
        print("  OFFLINE: generating synthetic bars (no network)")
        # a spread of end-states so every branch of the UI has something to show
        profiles = [
            {"squeeze_at": (300, 30), "breakout_size": 0.04, "post_drift": 0.002},   # long done
            {"squeeze_at": (392, 26), "breakout_size": 0.0,  "post_drift": 0.0},     # armed
            {"squeeze_at": (400, 40), "breakout_size": 0.0,  "post_drift": 0.0},     # building
            {"squeeze_at": (391, 24), "breakout_size": 0.12, "post_drift": 0.001},   # fires now
        ]
        hourly, daily = {}, {}
        for i, t in enumerate(tickers):
            h = bdata.synthetic_series(n_bars=417, seed=17 + i, base=80 + 11 * i,
                                       **profiles[i % len(profiles)])
            hourly[t] = h
            daily[t] = bdata.daily_from_intraday(h)
        return hourly, daily

    print(f"  downloading {len(tickers)} hourly series ({cfg.data.hourly_lookback_days}d)...")
    hourly = bdata.load_hourly(tickers, cfg.data)
    print(f"  downloading {len(tickers)} daily series ({cfg.data.daily_lookback_days}d)...")
    daily = bdata.load_daily(tickers, cfg.data)
    return hourly, daily


def exit_plan_for(entry_price: float, box_low: float, daily_atr: float, entry_ts, cfg) -> dict:
    arm = entry_price + cfg.exit.trail_arm_atr_multiple * daily_atr
    return {
        "hard_stop": round(box_low, 4),
        "hard_stop_rule": f"exit on a {cfg.exit.stop_bar_interval} close below the box low",
        "hard_stop_distance_pct": round((entry_price - box_low) / entry_price * 100, 2),
        "trail_arms_at": round(arm, 4),
        "trail_arms_rule": f"+{cfg.exit.trail_arm_atr_multiple:g} x daily ATR ({daily_atr:.2f})",
        "trail_giveback_pct": cfg.exit.trail_giveback_pct,
        "time_stop_trading_days": cfg.exit.time_stop_trading_days,
        "time_stop_on_or_before": str(
            (pd.Timestamp(entry_ts) + timedelta(days=cfg.exit.time_stop_trading_days * 7 / 5)).date()
        ),
    }


# ---------------------------------------------------------------------------
# main scan
# ---------------------------------------------------------------------------

def scan(cfg, offline=False, fetch_options=True, signal_lookback=6, tickers=None):
    tickers = tickers or cfg.tickers
    hourly, daily = load_universe_bars(tickers, cfg, offline)

    views, all_events, live_boxes = [], [], {}
    daily_atrs, last_prices = {}, {}

    for t in tickers:
        bars = hourly.get(t, pd.DataFrame())
        if bars is None or bars.empty or len(bars) < cfg.oscillator.bband_length + 2:
            views.append(strategy.TickerView(ticker=t, name=cfg.name_for(t),
                                             status="no data",
                                             note="not enough hourly history").to_dict())
            continue

        dbars = daily.get(t, pd.DataFrame())
        ema_daily = strategy.daily_ema_series(dbars, cfg.entry) if not dbars.empty else pd.Series(dtype=float)
        datr = (indicators.atr(dbars, cfg.exit.daily_atr_length).iloc[-1]
                if len(dbars) > cfg.exit.daily_atr_length else None)
        daily_atrs[t] = datr
        last_prices[t] = float(bars["close"].iloc[-1])

        osc = indicators.phase_oscillator(bars, cfg.oscillator)
        gate_fn = strategy.make_gate_fn(ema_daily, cfg.entry)
        events, live_box = boxes.run_box_machine(bars, osc, cfg.box, ticker=t, gate_fn=gate_fn)

        live_boxes[t] = live_box
        all_events.extend(events)
        views.append(strategy.build_view(t, cfg.name_for(t), bars, osc, live_box,
                                         ema_daily, datr, cfg.entry, cfg.box).to_dict())

    # --- which events are recent enough to matter -------------------------
    recent_cutoff = None
    if hourly:
        stamps = [b.index[-1] for b in hourly.values() if b is not None and not b.empty]
        if stamps:
            last_bar = max(stamps)
            lookback_idx = None
            for b in hourly.values():
                if b is not None and not b.empty and b.index[-1] == last_bar:
                    lookback_idx = b.index[-min(signal_lookback, len(b))]
                    break
            recent_cutoff = lookback_idx

    signals, missed = [], []
    for ev in sorted(all_events, key=lambda e: e.ts):
        if recent_cutoff is not None and ev.ts < recent_cutoff:
            continue
        if ev.kind not in (boxes.TRIGGERED, boxes.MISSED):
            continue
        t = ev.ticker
        bars = hourly.get(t)
        fresh = bool(bars is not None and not bars.empty and ev.ts == bars.index[-1])
        rec = {
            "id": strategy.position_id(t, ev.ts),
            "ticker": t,
            "name": cfg.name_for(t),
            "ts": ev.ts.isoformat(),
            "ts_et": _et(ev.ts),
            "fresh": fresh,
            "stock_price": round(float(ev.close), 4),
            "box_high": round(ev.box.high, 4),
            "box_low": round(ev.box.low, 4),
            "box_bars": ev.box.bar_count,
            "box_height_pct": round(ev.box.height_pct, 2),
            "gates": ev.gates,
            "daily_atr": None if daily_atrs.get(t) is None else round(float(daily_atrs[t]), 4),
        }
        if ev.kind == boxes.MISSED:
            rec["status"] = "missed"
            rec["block_reason"] = ", ".join(k for k, v in ev.gates.items() if not v) or "gate failed"
            missed.append(rec)
            continue

        rec["status"] = "actionable"
        rec["exit_plan"] = exit_plan_for(float(ev.close), ev.box.low,
                                         float(daily_atrs.get(t) or 0.0), ev.ts, cfg)

        if fetch_options and fresh and not offline:
            snap = bdata.load_option_snapshot(t, cfg.option, float(ev.close),
                                              cfg.entry.max_spread_pct_of_mid)
            rec["option"] = snap
            if not snap.get("ok"):
                rec["status"] = "blocked"
                rec["block_reason"] = snap.get("reason")
            elif not snap.get("spread_ok"):
                rec["status"] = "blocked"
                rec["block_reason"] = snap.get("spread_reason")
            else:
                rec["sizing"] = bopt.size_position(snap["mid"], cfg.account.equity, cfg.option)
                if rec["sizing"]["contracts"] == 0:
                    rec["status"] = "blocked"
                    rec["block_reason"] = rec["sizing"]["reason"]
        else:
            rec["option"] = {"ok": False, "reason": "chain not fetched (stale signal, "
                                                    "--no-options or --offline)"}
        signals.append(rec)

    return {
        "views": views,
        "signals": signals,
        "missed": missed,
        "events": [e.to_dict() for e in all_events],
        "hourly": hourly,
        "daily_atrs": daily_atrs,
        "last_prices": last_prices,
    }


# ---------------------------------------------------------------------------
# paper position journal
# ---------------------------------------------------------------------------

def update_journal(result, cfg, offline=False):
    path = os.path.join(DATA_DIR, "bilbo_positions.json")
    journal = _read_json(path, {"open": [], "closed": []})
    open_pos = [strategy.Position.from_dict(d) for d in journal.get("open", [])]
    closed = journal.get("closed", [])
    known = {p.id for p in open_pos} | {c.get("id") for c in closed}

    # 1. open anything the rules fired today
    opened_today = sum(1 for p in open_pos
                       if pd.Timestamp(p.entry_ts).date() == datetime.now(ET).date())
    for sig in result["signals"]:
        if sig["status"] != "actionable" or not sig.get("fresh"):
            continue
        if sig["id"] in known:
            continue
        if cfg.entry.one_position_per_ticker and any(p.ticker == sig["ticker"] for p in open_pos):
            print(f"  skip {sig['ticker']}: already holding one")
            continue
        if opened_today >= cfg.entry.max_new_positions_per_day:
            print(f"  skip {sig['ticker']}: daily new-position cap reached")
            continue
        contract = dict(sig.get("option") or {})
        contract.update(sig.get("sizing") or {})
        pos = strategy.Position(
            id=sig["id"], ticker=sig["ticker"], entry_ts=sig["ts"],
            entry_stock=sig["stock_price"], box_high=sig["box_high"],
            box_low=sig["box_low"], daily_atr=float(sig.get("daily_atr") or 0.0),
            contract=contract, peak_stock=sig["stock_price"],
            notes=[f"opened by scan at {_et(sig['ts'])}"],
        )
        open_pos.append(pos)
        opened_today += 1
        print(f"  OPENED {pos.id} @ {pos.entry_stock}")

    if not open_pos:
        payload = {"generated_at": datetime.now(SYDNEY).isoformat(),
                   "equity": cfg.account.equity, "open": [], "closed": closed,
                   "stats": journal_stats(closed)}
        _write_json(path, payload)
        return payload

    # 2. walk the stop-interval bars for everything still open
    oldest = min(pd.Timestamp(p.entry_ts) for p in open_pos)
    need_days = max(cfg.data.intraday_lookback_days,
                    (pd.Timestamp.now(tz=ET) - oldest).days + 2)
    need_days = min(need_days, cfg.data.max_intraday_lookback_days)
    tickers = sorted({p.ticker for p in open_pos})

    if offline:
        stop_bars = {t: result["hourly"].get(t, pd.DataFrame()) for t in tickers}
        print("  OFFLINE: using hourly bars for exit checks")
    else:
        print(f"  downloading {cfg.exit.stop_bar_interval} bars for {len(tickers)} open "
              f"position(s) ({need_days}d)...")
        stop_bars = bdata.load_stop_bars(tickers, cfg.data, cfg.exit.stop_bar_interval)
        for t in tickers:
            if stop_bars.get(t) is None or stop_bars[t].empty:
                stop_bars[t] = result["hourly"].get(t, pd.DataFrame())
                print(f"  WARNING: no {cfg.exit.stop_bar_interval} bars for {t} — "
                      f"falling back to hourly for exits")

    still_open = []
    for pos in open_pos:
        bars = stop_bars.get(pos.ticker, pd.DataFrame())
        if bars is None or bars.empty:
            still_open.append(pos.to_dict())
            continue
        state = strategy.evaluate_exit(pos, bars, cfg.exit)
        pos.peak_stock = round(state.peak_stock, 4)
        pos.trail_armed = state.trail_armed
        pos.bars_seen = state.bars_checked
        if state.should_exit:
            pos.status = "closed"
            pos.exit_ts = state.ts.isoformat()
            pos.exit_stock = round(float(state.price), 4)
            pos.exit_reason = state.reason
            entry = pos.entry_stock
            rec = pos.to_dict()
            rec["stock_return_pct"] = round((pos.exit_stock / entry - 1) * 100, 3)
            rec["trading_days_held"] = state.trading_days_held
            rec["exit_state"] = state.to_dict()
            closed.append(rec)
            print(f"  CLOSED {pos.id}: {state.reason} @ {pos.exit_stock} "
                  f"({rec['stock_return_pct']:+.2f}% on the stock)")
        else:
            d = pos.to_dict()
            d["live"] = state.to_dict()
            d["unrealised_stock_pct"] = round((float(state.price) / pos.entry_stock - 1) * 100, 3)
            still_open.append(d)

    payload = {
        "generated_at": datetime.now(SYDNEY).isoformat(),
        "equity": cfg.account.equity,
        "open": still_open,
        "closed": closed,
        "stats": journal_stats(closed),
    }
    _write_json(path, payload)
    return payload


def journal_stats(closed: list) -> dict:
    if not closed:
        return {"trades": 0}
    rets = [c.get("stock_return_pct", 0.0) for c in closed]
    wins = [r for r in rets if r > 0]
    by_reason = {}
    for c in closed:
        by_reason[c.get("exit_reason", "?")] = by_reason.get(c.get("exit_reason", "?"), 0) + 1
    return {
        "trades": len(closed),
        "win_rate_pct": round(len(wins) / len(rets) * 100, 1),
        "avg_stock_return_pct": round(sum(rets) / len(rets), 3),
        "best_stock_return_pct": round(max(rets), 3),
        "worst_stock_return_pct": round(min(rets), 3),
        "exit_reasons": by_reason,
        "note": "stock-leg returns; option P&L depends on the contract actually bought",
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run(argv=None):
    ap = argparse.ArgumentParser(description="Bilbo Box scanner")
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="synthetic bars, no network")
    ap.add_argument("--no-options", action="store_true", help="skip option chain fetches")
    ap.add_argument("--no-journal", action="store_true", help="do not touch the position journal")
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--tickers", default=None, help="comma-separated override of the universe")
    ap.add_argument("--lookback-bars", type=int, default=6,
                    help="how many hourly bars back to report signals from")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.equity:
        cfg.account.equity = args.equity
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print("=" * 72)
    print("BILBO BOX SCAN")
    print(f"  universe: {len(tickers or cfg.tickers)} tickers   equity: "
          f"{cfg.account.equity:,.0f} {cfg.account.currency}")
    print("=" * 72)

    result = scan(cfg, offline=args.offline, fetch_options=not args.no_options,
                  signal_lookback=args.lookback_bars, tickers=tickers)

    counts = {}
    for v in result["views"]:
        counts[v["status"]] = counts.get(v["status"], 0) + 1

    now = datetime.now(SYDNEY)
    state = {
        "generated_at": now.isoformat(),
        "generated_at_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "data_mode": "offline-fixture" if args.offline else "live",
        "universe_size": len(result["views"]),
        "counts": counts,
        "equity": cfg.account.equity,
        "params": {
            "box_bars": cfg.box.max_box_bars,
            "entry_window": f"{cfg.entry.window_start}-{cfg.entry.window_end} "
                            f"{cfg.entry.session_timezone}",
            "daily_ema": cfg.entry.daily_ema_length,
            "dte_window": f"{cfg.option.min_dte}-{cfg.option.max_dte} (target "
                          f"{cfg.option.target_dte})",
            "premium_band_pct": [cfg.option.premium_pct_min, cfg.option.premium_pct_max],
            "max_spread_pct_of_mid": cfg.entry.max_spread_pct_of_mid,
            "exits": f"{cfg.exit.stop_bar_interval} close < box low | trail after "
                     f"+{cfg.exit.trail_arm_atr_multiple:g} ATR giving back "
                     f"{cfg.exit.trail_giveback_pct:g}% | {cfg.exit.time_stop_trading_days}d time stop",
        },
        "tickers": sorted(result["views"], key=lambda v: (
            {"armed": 0, "building": 1, "compressing": 2, "idle": 3, "no data": 4}.get(v["status"], 5),
            v.get("distance_to_trigger_pct") if v.get("distance_to_trigger_pct") is not None else 99,
        )),
    }
    _write_json(os.path.join(DATA_DIR, "bilbo_state.json"), state)

    _write_json(os.path.join(DATA_DIR, "bilbo_signals.json"), {
        "generated_at": now.isoformat(),
        "lookback_bars": args.lookback_bars,
        "actionable": [s for s in result["signals"] if s["status"] == "actionable"],
        "blocked": [s for s in result["signals"] if s["status"] == "blocked"],
        "missed": result["missed"],
    })

    if not args.no_journal:
        update_journal(result, cfg, offline=args.offline)

    print("-" * 72)
    print(f"  {counts.get('armed', 0)} armed  |  {counts.get('building', 0)} building  |  "
          f"{counts.get('compressing', 0)} compressing  |  {counts.get('idle', 0)} idle")
    for s in result["signals"]:
        flag = "ACT" if s["status"] == "actionable" and s.get("fresh") else s["status"].upper()
        print(f"  [{flag}] {s['ticker']} {s['ts_et']} close {s['stock_price']} "
              f"> box {s['box_high']} {s.get('block_reason') or ''}")
    for m in result["missed"]:
        print(f"  [MISS] {m['ticker']} {m['ts_et']} — {m['block_reason']}")
    for v in state["tickers"][:5]:
        if v["status"] == "armed":
            print(f"  [ARMED] {v['ticker']} trigger {v['box']['high']} "
                  f"({v['distance_to_trigger_pct']:+.2f}% away)")
    print("=" * 72)
    return state


if __name__ == "__main__":
    run()
