#!/usr/bin/env python3
"""
bilbo_backtest.py — replay the rules over history.

Read this before you read any number it prints:

  The stock leg is real. Entries, exits, holding periods, MFE/MAE and the
  stock-move distribution all come from actual bars and are trustworthy.

  The option leg is a MODEL. No historical option quotes are available here,
  so each trade is priced with Black-Scholes using realised volatility as a
  stand-in for implied. That gets the shape of the payoff roughly right and
  the level roughly wrong: it ignores the volatility risk premium drifting,
  IV crush after events, skew, and the real bid-ask you would have paid.
  Treat option P&L here as an order-of-magnitude sanity check, never as a
  track record. The published study used real quotes; this does not.

Two more limits worth knowing:
  * hourly history from yfinance stops at ~730 days, so this cannot reach 2019
  * 5-minute history stops at ~60 days, so exits are checked on hourly closes
    by default, which makes stops *later* and therefore slightly generous

    python scripts/bilbo_backtest.py --days 720
    python scripts/bilbo_backtest.py --offline --csv trades.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilbo import boxes, data as bdata, indicators, options as bopt, strategy
from bilbo.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SYDNEY = pytz.timezone("Australia/Sydney")


# ---------------------------------------------------------------------------
# volatility stand-in
# ---------------------------------------------------------------------------

def realized_vol_series(daily: pd.DataFrame, window: int) -> pd.Series:
    """Annualised close-to-close volatility — the proxy for implied."""
    logret = np.log(daily["close"].astype(float)).diff()
    return logret.rolling(window).std(ddof=1) * np.sqrt(252)


def iv_for(trade_ts, rv: pd.Series, p_bt) -> float:
    if p_bt.iv_mode == "fixed":
        return float(p_bt.fixed_iv)
    v = strategy.series_at(rv, trade_ts)
    if v is None or not np.isfinite(v) or v <= 0:
        return float(p_bt.fixed_iv)
    return float(v) * float(p_bt.iv_premium_multiple)


# ---------------------------------------------------------------------------
# one trade
# ---------------------------------------------------------------------------

def simulate_trade(ev, bars, atr_s, rv, cfg) -> dict | None:
    entry_ts = ev.ts
    entry_stock = float(ev.close)
    datr = strategy.series_at(atr_s, entry_ts)
    if datr is None or datr <= 0:
        return None

    pos = strategy.Position(
        id=strategy.position_id(ev.ticker, entry_ts), ticker=ev.ticker,
        entry_ts=entry_ts.isoformat(), entry_stock=entry_stock,
        box_high=ev.box.high, box_low=ev.box.low, daily_atr=datr,
        peak_stock=entry_stock,
    )
    state = strategy.evaluate_exit(pos, bars, cfg.exit)

    exit_ts = state.ts if state.should_exit else bars.index[-1]
    exit_stock = float(state.price if state.price is not None else entry_stock)
    reason = state.reason or "still_open_at_data_end"

    # --- the option overlay (a model, not a quote) -----------------------
    strike = bopt.synthetic_strike(entry_stock, cfg.option)
    vol = iv_for(entry_ts, rv, cfg.backtest)
    t_entry = cfg.option.target_dte / 365.0
    held_days = max(0.0, (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 86400.0)
    t_exit = max(0.0, t_entry - held_days / 365.0)

    entry_prem = bopt.bs_call(entry_stock, strike, t_entry, cfg.option.risk_free_rate, vol)
    exit_prem = bopt.bs_call(exit_stock, strike, t_exit, cfg.option.risk_free_rate,
                             max(0.01, vol - cfg.backtest.iv_crush_on_exit))
    if entry_prem <= 0.01:
        return None

    half = cfg.backtest.round_trip_spread_pct / 200.0
    paid = entry_prem * (1 + half)
    got = exit_prem * (1 - half)

    sizing = bopt.size_position(paid, cfg.account.equity, cfg.option)
    contracts = sizing["contracts"]
    pnl = (got - paid) * contracts * cfg.option.contract_multiplier

    return {
        "id": pos.id,
        "ticker": ev.ticker,
        "entry_ts": entry_ts.isoformat(),
        "entry_date": str(pd.Timestamp(entry_ts).date()),
        "exit_ts": pd.Timestamp(exit_ts).isoformat(),
        "exit_reason": reason,
        "trading_days_held": state.trading_days_held,
        "entry_stock": round(entry_stock, 4),
        "exit_stock": round(exit_stock, 4),
        "stock_return_pct": round((exit_stock / entry_stock - 1) * 100, 3),
        "peak_stock": round(state.peak_stock, 4),
        "mfe_pct": round((state.peak_stock / entry_stock - 1) * 100, 3),
        "box_low": round(ev.box.low, 4),
        "box_high": round(ev.box.high, 4),
        "box_height_pct": round(ev.box.height_pct, 3),
        "daily_atr": round(datr, 4),
        "trail_armed": state.trail_armed,
        "model_strike": strike,
        "model_iv": round(vol, 4),
        "model_entry_premium": round(paid, 4),
        "model_exit_premium": round(got, 4),
        "model_return_on_premium_pct": round((got / paid - 1) * 100, 2),
        "model_contracts": contracts,
        "model_pnl": round(pnl, 2),
        "model_premium_at_risk": round(paid * contracts * cfg.option.contract_multiplier, 2),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def summarise(trades: list, cfg) -> dict:
    if not trades:
        return {"trades": 0, "note": "no signals in the sample"}

    df = pd.DataFrame(trades).sort_values("entry_ts")
    stock = df["stock_return_pct"]
    model = df["model_return_on_premium_pct"]
    pnl = df["model_pnl"]

    equity_curve = pnl.cumsum()
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak

    wins = model > 0
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    by_reason = (df.groupby("exit_reason")
                   .agg(n=("id", "count"),
                        avg_stock_pct=("stock_return_pct", "mean"),
                        avg_model_pct=("model_return_on_premium_pct", "mean"))
                   .round(2).reset_index().to_dict("records"))
    by_ticker = (df.groupby("ticker")
                   .agg(n=("id", "count"),
                        avg_stock_pct=("stock_return_pct", "mean"),
                        avg_model_pct=("model_return_on_premium_pct", "mean"),
                        model_pnl=("model_pnl", "sum"))
                   .round(2).sort_values("model_pnl", ascending=False)
                   .reset_index().to_dict("records"))
    df["year"] = df["entry_date"].str[:4]
    by_year = (df.groupby("year")
                 .agg(n=("id", "count"),
                      avg_model_pct=("model_return_on_premium_pct", "mean"),
                      model_pnl=("model_pnl", "sum"))
                 .round(2).reset_index().to_dict("records"))

    return {
        "trades": int(len(df)),
        "first_entry": df["entry_date"].iloc[0],
        "last_entry": df["entry_date"].iloc[-1],
        "stock_leg": {
            "win_rate_pct": round(float((stock > 0).mean() * 100), 1),
            "avg_move_pct": round(float(stock.mean()), 3),
            "median_move_pct": round(float(stock.median()), 3),
            "avg_winner_pct": round(float(stock[stock > 0].mean() or 0), 3),
            "avg_loser_pct": round(float(stock[stock <= 0].mean() or 0), 3),
            "avg_mfe_pct": round(float(df["mfe_pct"].mean()), 3),
            "avg_trading_days_held": round(float(df["trading_days_held"].mean()), 2),
            "trail_armed_pct": round(float(df["trail_armed"].mean() * 100), 1),
        },
        "option_model": {
            "win_rate_pct": round(float(wins.mean() * 100), 1),
            "avg_return_on_premium_pct": round(float(model.mean()), 2),
            "median_return_on_premium_pct": round(float(model.median()), 2),
            "avg_winner_pct": round(float(model[model > 0].mean() or 0), 2),
            "avg_loser_pct": round(float(model[model <= 0].mean() or 0), 2),
            "profit_factor": round(float(gross_win / gross_loss), 2) if gross_loss else None,
            "total_pnl": round(float(pnl.sum()), 2),
            "max_drawdown": round(float(drawdown.min()), 2),
            "expectancy_per_trade": round(float(pnl.mean()), 2),
            "premium_at_risk_per_trade": round(float(df["model_premium_at_risk"].mean()), 2),
            "WARNING": "modelled with Black-Scholes on realised vol, not real quotes",
        },
        "by_exit_reason": by_reason,
        "by_ticker": by_ticker,
        "by_year": by_year,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def backtest(cfg, days: int, offline: bool, tickers=None) -> dict:
    tickers = tickers or cfg.tickers
    cfg.data.hourly_lookback_days = min(days, cfg.data.max_hourly_lookback_days)

    if offline:
        print("  OFFLINE: synthetic bars")
        hourly = {t: bdata.synthetic_series(n_bars=900, seed=31 + i, base=60 + 9 * i,
                                            squeeze_at=(700 - 7 * (i % 6), 28))
                  for i, t in enumerate(tickers)}
        daily = {t: bdata.daily_from_intraday(h) for t, h in hourly.items()}
    else:
        print(f"  downloading {len(tickers)} hourly series ({cfg.data.hourly_lookback_days}d)...")
        hourly = bdata.load_hourly(tickers, cfg.data)
        daily = bdata.load_daily(tickers, cfg.data)

    trades, skipped = [], {"no_data": 0, "no_atr": 0}
    counts = {"triggered": 0, "missed": 0, "invalidated": 0, "expired": 0,
              "superseded": 0, "frozen": 0, "discarded": 0}

    for t in tickers:
        bars = hourly.get(t, pd.DataFrame())
        dbars = daily.get(t, pd.DataFrame())
        if bars is None or bars.empty or dbars is None or dbars.empty:
            skipped["no_data"] += 1
            continue

        osc = indicators.phase_oscillator(bars, cfg.oscillator)
        ema_daily = strategy.daily_ema_series(dbars, cfg.entry)
        atr_s = indicators.atr(dbars, cfg.exit.daily_atr_length)
        rv = realized_vol_series(dbars, cfg.backtest.realized_vol_days)

        events, _ = boxes.run_box_machine(bars, osc, cfg.box, ticker=t,
                                          gate_fn=strategy.make_gate_fn(ema_daily, cfg.entry))
        for ev in events:
            counts[ev.kind] = counts.get(ev.kind, 0) + 1
            if ev.kind != boxes.TRIGGERED:
                continue
            trade = simulate_trade(ev, bars, atr_s, rv, cfg)
            if trade is None:
                skipped["no_atr"] += 1
                continue
            trades.append(trade)

    return {"trades": trades, "event_counts": counts, "skipped": skipped,
            "universe": tickers}


def run(argv=None):
    ap = argparse.ArgumentParser(description="Bilbo Box backtest (stock leg real, option leg modelled)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--days", type=int, default=720)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--iv", type=float, default=None, help="fixed implied vol, e.g. 0.35")
    ap.add_argument("--csv", default=None, help="also write the trade list to this path")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.equity:
        cfg.account.equity = args.equity
    if args.iv:
        cfg.backtest.iv_mode = "fixed"
        cfg.backtest.fixed_iv = args.iv
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    print("=" * 72)
    print("BILBO BOX BACKTEST")
    print("  stock leg: real bars   |   option leg: Black-Scholes model, NOT quotes")
    print("=" * 72)

    out = backtest(cfg, args.days, args.offline, tickers)
    summary = summarise(out["trades"], cfg)

    payload = {
        "generated_at": datetime.now(SYDNEY).isoformat(),
        "data_mode": "offline-fixture" if args.offline else "live",
        "lookback_days": min(args.days, cfg.data.max_hourly_lookback_days),
        "universe": out["universe"],
        "event_counts": out["event_counts"],
        "summary": summary,
        "trades": out["trades"],
        "method_note": (
            "Stock entries/exits are computed from real hourly bars. Option P&L is a "
            "Black-Scholes model priced on realised volatility x "
            f"{cfg.backtest.iv_premium_multiple} with a "
            f"{cfg.backtest.round_trip_spread_pct}% round-trip spread charge. It is an "
            "approximation, not a quote-accurate result."
        ),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "bilbo_backtest.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {os.path.relpath(path, ROOT)}")

    if args.csv and out["trades"]:
        pd.DataFrame(out["trades"]).to_csv(args.csv, index=False)
        print(f"  wrote {args.csv}")

    print("-" * 72)
    print(f"  events: {out['event_counts']}")
    if summary.get("trades"):
        s, m = summary["stock_leg"], summary["option_model"]
        print(f"  {summary['trades']} trades  {summary['first_entry']} -> {summary['last_entry']}")
        print(f"  STOCK  win {s['win_rate_pct']}%  avg {s['avg_move_pct']:+.2f}%  "
              f"MFE {s['avg_mfe_pct']:+.2f}%  held {s['avg_trading_days_held']}d  "
              f"trail armed {s['trail_armed_pct']}%")
        print(f"  MODEL  win {m['win_rate_pct']}%  avg {m['avg_return_on_premium_pct']:+.1f}% "
              f"of premium  PF {m['profit_factor']}  maxDD {m['max_drawdown']:,.0f}")
        for r in summary["by_exit_reason"]:
            print(f"    {r['exit_reason']:<24} n={r['n']:<4} stock {r['avg_stock_pct']:+.2f}%  "
                  f"model {r['avg_model_pct']:+.1f}%")
    else:
        print("  no trades")
    print("=" * 72)
    return payload


if __name__ == "__main__":
    run()
