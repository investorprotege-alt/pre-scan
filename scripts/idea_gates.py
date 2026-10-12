"""
idea_gates.py — Six-gate idea filter ("Claude, my quant researcher" pattern).

Every screener hit is guilty until proven innocent. Each candidate has to clear
six gates before it reaches the focus list; a day where nothing survives is the
system working, not failing.

  G1 Regime      Is the tide with me?           QQQE 21DMA scenario 1/2 (3 = pilot size)
  G2 Structure   Is it a Stage 2 uptrend?       Trend template: price > 50 > 200, 200 rising,
                                                within 25% of 52w high, 30%+ off 52w low
  G3 Leadership  Does the cross-check agree?    RS score >= 80 and/or sector in the top group
  G4 Location    Is there a low-risk entry?     ATR multiple from 50SMA < 4, within 1 ATR of 10/21 EMA
  G5 Contrary    What's the case against it?    No earnings within 7 days, no heavy distribution,
                                                no high-volume break of the 21 EMA
  G6 Cost/Risk   Is the edge worth the friction? Liquidity floor, stop <= 2x ADR, friction <= 15% of R

Survivors get a stop, an R-based position size (risk_per_trade_pct, capped at
max_position_pct) and a score. Every evaluation is appended to
data/idea_ledger.jsonl so ledger_review.py can later test whether the gates
actually have edge (passed vs rejected forward returns).

Outputs: data/idea_gates_<session>.json, data/idea_ledger.jsonl (append)
Usage:   python scripts/idea_gates.py --session=us|asx
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.json")
os.makedirs(DATA_DIR, exist_ok=True)

AEST = pytz.timezone("Australia/Sydney")

GATES = ["regime", "structure", "leadership", "location", "contrary", "cost_risk"]

# Default thresholds — override any of these under settings.gates in watchlist.json
DEFAULTS = {
    "rs_min":                80,
    "max_pct_below_52w_high": 25.0,
    "min_pct_above_52w_low":  30.0,
    "max_atr_mult_from_50":   4.0,
    "max_atr_from_ema":       1.0,
    "earnings_buffer_days":   7,
    "max_distribution_days":  3,       # in the last 15 sessions
    "min_dollar_vol_us":      20_000_000,
    "min_dollar_vol_asx":     5_000_000,
    "max_stop_adr_mult":      2.0,
    "max_friction_pct_of_r":  15.0,
    "commission_bps_us":      1.0,     # per side (IBKR-style)
    "commission_bps_asx":     8.0,     # per side (~$8 on a $10k parcel)
}

# Sector ETF names (config) -> GICS sector names (Wikipedia/S&P 500 metadata)
SECTOR_ALIASES = {
    "technology":    "information technology",
    "healthcare":    "health care",
    "discretionary": "consumer discretionary",
}


def load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def normalise_sector(name: str) -> str:
    n = (name or "").strip().lower()
    return SECTOR_ALIASES.get(n, n)


# ---------------------------------------------------------------------------
# Metrics from one ticker's OHLCV history
# ---------------------------------------------------------------------------

def compute_metrics(hist: pd.DataFrame) -> dict | None:
    """hist: daily OHLCV with columns Open/High/Low/Close/Volume, oldest first."""
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 60:
        return None

    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"].fillna(0)
    price = float(close.iloc[-1])

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    adr_pct = float(((high / low - 1) * 100).iloc[-20:].mean())

    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma200_series = close.rolling(200).mean()
    sma200 = float(sma200_series.iloc[-1]) if len(close) >= 200 else None
    sma200_1m_ago = float(sma200_series.iloc[-22]) if len(close) >= 222 else None
    ema10  = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
    ema21  = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

    window = hist.iloc[-252:]
    high_52w = float(window["High"].max())
    low_52w  = float(window["Low"].min())

    avg_vol50 = float(vol.iloc[-50:].mean())
    dollar_vol = float((close * vol).iloc[-20:].mean())
    rvol = float(vol.iloc[-1] / avg_vol50) if avg_vol50 else None

    # Distribution days: down closes on volume > 1.5x the 50-day average
    chg = close.pct_change()
    recent = slice(-15, None)
    dist_days = int(((chg.iloc[recent] < -0.002) & (vol.iloc[recent] > 1.5 * avg_vol50)).sum())

    # Last session closed below the 21 EMA on above-average volume
    heavy_break_21 = bool(price < ema21 and vol.iloc[-1] > 1.3 * avg_vol50)

    def ret(n):
        return float(price / close.iloc[-1 - n] - 1) * 100 if len(close) > n else None

    r3m, r6m, r9m, r12m = ret(63), ret(126), ret(189), ret(252)
    rs_raw = None
    if None not in (r3m, r6m, r9m, r12m):
        rs_raw = 0.4 * r3m + 0.2 * r6m + 0.2 * r9m + 0.2 * r12m

    return {
        "price":       round(price, 4),
        "atr14":       round(atr14, 4),
        "adr_pct":     round(adr_pct, 2),
        "sma50":       round(sma50, 4),
        "sma200":      round(sma200, 4) if sma200 else None,
        "sma200_rising": (sma200 > sma200_1m_ago) if (sma200 and sma200_1m_ago) else None,
        "ema10":       round(ema10, 4),
        "ema21":       round(ema21, 4),
        "high_52w":    round(high_52w, 4),
        "low_52w":     round(low_52w, 4),
        "pct_below_52w_high": round((1 - price / high_52w) * 100, 2),
        "pct_above_52w_low":  round((price / low_52w - 1) * 100, 2),
        "atr_mult_from_50":   round(abs(price - sma50) / atr14, 2) if atr14 else None,
        "atr_from_ema10":     round(abs(price - ema10) / atr14, 2) if atr14 else None,
        "atr_from_ema21":     round(abs(price - ema21) / atr14, 2) if atr14 else None,
        "low_5d":      round(float(low.iloc[-5:].min()), 4),
        "dollar_vol":  round(dollar_vol, 0),
        "rvol":        round(rvol, 2) if rvol is not None else None,
        "distribution_days_15": dist_days,
        "heavy_break_21ema":    heavy_break_21,
        "rs_raw":      round(rs_raw, 2) if rs_raw is not None else None,
    }


# ---------------------------------------------------------------------------
# The six gates. Each returns (passed: bool, reason: str)
# ---------------------------------------------------------------------------

def gate_regime(ctx: dict) -> tuple[bool, str]:
    s = ctx.get("qqqe_scenario")
    if s in (1, 2):
        return True, f"QQQE scenario {s} — full size"
    if s == 3:
        return True, "QQQE scenario 3 — pilot size only (half R)"
    return False, f"QQQE scenario {s or 'unknown'} — no new longs"


def gate_structure(m: dict, t: dict) -> tuple[bool, str]:
    fails = []
    if m["sma200"] is None:
        fails.append("no 200SMA (history too short)")
    else:
        if not (m["price"] > m["sma50"] > m["sma200"]):
            fails.append("not price > 50 > 200")
        if m["sma200_rising"] is False:
            fails.append("200SMA falling")
    if m["pct_below_52w_high"] > t["max_pct_below_52w_high"]:
        fails.append(f"{m['pct_below_52w_high']:.0f}% off 52w high")
    if m["pct_above_52w_low"] < t["min_pct_above_52w_low"]:
        fails.append(f"only {m['pct_above_52w_low']:.0f}% above 52w low")
    return (not fails), "; ".join(fails) or "Stage 2 trend template"


def gate_leadership(m: dict, cand: dict, ctx: dict, t: dict) -> tuple[bool, str]:
    rs = ctx["rs_lookup"].get(cand["ticker"])
    in_top_sector = normalise_sector(cand.get("sector")) in ctx["top_sectors"]
    beats_bench = (m["rs_raw"] is not None and ctx.get("bench_rs_raw") is not None
                   and m["rs_raw"] > ctx["bench_rs_raw"])

    if rs is not None:
        if rs >= t["rs_min"]:
            return True, f"RS {rs}" + (" + top sector" if in_top_sector else "")
        return False, f"RS {rs} < {t['rs_min']}"
    # No usable RS rank — fall back to beating the benchmark AND sector strength
    if beats_bench and in_top_sector:
        return True, "beats benchmark RS + top sector (no RS rank)"
    if beats_bench:
        return True, "beats benchmark RS (no RS rank, sector not top)"
    return False, "lags benchmark on weighted RS"


def gate_location(m: dict, t: dict) -> tuple[bool, str]:
    if m["atr_mult_from_50"] is not None and m["atr_mult_from_50"] >= t["max_atr_mult_from_50"]:
        return False, f"extended {m['atr_mult_from_50']:.1f} ATR from 50SMA — watch for a reset"
    if m["atr_from_ema10"] is None or m["atr_from_ema21"] is None:
        return False, "no ATR — can't judge location"
    near = min(m["atr_from_ema10"], m["atr_from_ema21"])
    if near > t["max_atr_from_ema"]:
        return False, f"{near:.1f} ATR from nearest 10/21 EMA — no low-risk entry"
    which = "10 EMA" if m["atr_from_ema10"] <= m["atr_from_ema21"] else "21 EMA"
    return True, f"{near:.1f} ATR from {which}"


def gate_contrary(m: dict, cand: dict, ctx: dict, t: dict) -> tuple[bool, str]:
    fails, notes = [], []
    ed = ctx["earnings"].get(cand["ticker"])
    if ed is None:
        notes.append("earnings date unknown — check manually")
    else:
        days = (ed - ctx["today"]).days
        if 0 <= days <= t["earnings_buffer_days"]:
            fails.append(f"earnings in {days}d ({ed.isoformat()})")
    if m["distribution_days_15"] > t["max_distribution_days"]:
        fails.append(f"{m['distribution_days_15']} distribution days / 15")
    if m["heavy_break_21ema"]:
        fails.append("closed below 21 EMA on heavy volume")
    if fails:
        return False, "; ".join(fails)
    return True, "; ".join(notes) or "no red flags found"


def friction_bps(dollar_vol: float) -> float:
    """Rough one-way spread + slippage by liquidity tier."""
    if dollar_vol >= 500_000_000: return 3.0
    if dollar_vol >= 100_000_000: return 6.0
    if dollar_vol >= 25_000_000:  return 12.0
    return 25.0


def gate_cost_risk(m: dict, ctx: dict, t: dict, settings: dict) -> tuple[bool, str, dict]:
    floor = t["min_dollar_vol_asx"] if ctx["session"] == "asx" else t["min_dollar_vol_us"]
    if m["dollar_vol"] < floor:
        return False, f"${m['dollar_vol'] / 1e6:.1f}M/day < ${floor / 1e6:.0f}M liquidity floor", {}

    entry = m["price"]
    # Stop under the tighter of the 5-day low or 21 EMA, but never tighter than 0.5 ATR
    structural = max(m["low_5d"], m["ema21"]) if m["ema21"] < entry else m["low_5d"]
    stop = min(structural, entry - 0.5 * m["atr14"])
    stop_pct = (entry - stop) / entry * 100
    if stop_pct <= 0:
        return False, "no valid stop below price", {}
    if stop_pct > t["max_stop_adr_mult"] * m["adr_pct"]:
        return False, f"stop {stop_pct:.1f}% > {t['max_stop_adr_mult']:.0f}x ADR ({m['adr_pct']:.1f}%)", {}

    commission = t["commission_bps_asx"] if ctx["session"] == "asx" else t["commission_bps_us"]
    round_trip_pct = 2 * (friction_bps(m["dollar_vol"]) + commission) / 100
    friction_of_r = round_trip_pct / stop_pct * 100
    if friction_of_r > t["max_friction_pct_of_r"]:
        return False, f"friction {friction_of_r:.0f}% of R — stop too tight for the costs", {}

    risk = float(settings.get("risk_per_trade_pct", 0.75))
    if ctx.get("qqqe_scenario") == 3:
        risk /= 2
    position_pct = min(risk / stop_pct * 100, float(settings.get("max_position_pct", 20.0)))
    plan = {
        "entry_ref":      round(entry, 2),
        "stop":           round(stop, 2),
        "stop_pct":       round(stop_pct, 2),
        "risk_pct":       round(risk, 3),
        "position_pct":   round(position_pct, 1),
        "target_2r":      round(entry + 2 * (entry - stop), 2),
        "friction_pct_of_r": round(friction_of_r, 1),
    }
    return True, f"stop {stop_pct:.1f}%, size {position_pct:.1f}% of account, friction {friction_of_r:.0f}% of R", plan


def score(m: dict, rs: int | None, in_top_sector: bool) -> float:
    """Rank survivors: leadership first, then tightness of the entry and liquidity."""
    s = (rs or 70) * 1.0
    s += 10 if in_top_sector else 0
    s += max(0.0, 10 - 10 * min(m["atr_from_ema10"], m["atr_from_ema21"]))
    s += max(0.0, 10 - m["pct_below_52w_high"])
    s += 5 if (m["rvol"] or 0) < 0.8 else 0      # quiet pullback = supply drying up
    return round(s, 1)


def evaluate(cand: dict, m: dict, ctx: dict, t: dict, settings: dict) -> dict:
    results = {}
    results["regime"]     = gate_regime(ctx)
    results["structure"]  = gate_structure(m, t)
    results["leadership"] = gate_leadership(m, cand, ctx, t)
    results["location"]   = gate_location(m, t)
    results["contrary"]   = gate_contrary(m, cand, ctx, t)
    ok, why, plan         = gate_cost_risk(m, ctx, t, settings)
    results["cost_risk"]  = (ok, why)

    gates = {g: {"pass": bool(results[g][0]), "reason": results[g][1]} for g in GATES}
    first_fail = next((g for g in GATES if not gates[g]["pass"]), None)
    rs = ctx["rs_lookup"].get(cand["ticker"])
    in_top = normalise_sector(cand.get("sector")) in ctx["top_sectors"]
    return {
        "ticker":     cand["ticker"],
        "company":    cand.get("company") or cand.get("name", ""),
        "sector":     cand.get("sector", ""),
        "screeners":  cand.get("screeners", []),
        "passed":     first_fail is None,
        "killed_by":  first_fail,
        "gates_passed": sum(1 for g in GATES if gates[g]["pass"]),
        "gates":      gates,
        "rs_score":   rs,
        "score":      score(m, rs, in_top) if first_fail is None else None,
        "plan":       plan if first_fail is None else {},
        "metrics":    m,
    }


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def download_histories(tickers: list) -> dict:
    raw = yf.download(tickers, period="14mo", auto_adjust=True, progress=False,
                      threads=True, group_by="ticker")
    out = {}
    if raw.empty:
        return out
    for tk in tickers:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            out[tk] = df.dropna(how="all")
        except KeyError:
            continue
    return out


def fetch_earnings_dates(tickers: list, today) -> dict:
    """Next earnings date per ticker (date) — missing if yfinance has nothing."""
    out = {}
    for tk in tickers:
        try:
            cal = yf.Ticker(tk).calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if dates:
                upcoming = [d for d in dates if d >= today - timedelta(days=1)]
                if upcoming:
                    out[tk] = min(upcoming)
        except Exception:
            continue
    return out


def gather_candidates(session: str, cfg: dict) -> list:
    if session == "us":
        cands = load_json("finviz_screener.json").get("combined_candidates", [])
        focus = [{"ticker": f["ticker"], "screeners": ["Focus list"]} for f in cfg.get("us_focus_list", []) if f.get("ticker")]
    else:
        cands = [dict(item, screeners=["ASX leaders"]) for group in cfg["asx_leaders"].values() for item in group]
        focus = [{"ticker": f["ticker"], "screeners": ["Focus list"]} for f in cfg.get("asx_focus_list", []) if f.get("ticker")]
    seen = {c["ticker"]: c for c in cands}
    for f in focus:
        seen.setdefault(f["ticker"], f)
    return list(seen.values())


def build_context(session: str, cfg: dict) -> dict:
    scan = load_json("sector_scan.json")
    rs   = load_json("rs_rankings.json")
    key  = "asx_rankings" if session == "asx" else "us_rankings"
    rankings = rs.get(key, [])
    # A handful of rows means the universe download failed — the percentiles are meaningless
    rs_lookup = {r["ticker"]: r["rs_score"] for r in rankings} if len(rankings) >= 50 else {}

    sectors = scan.get("asx_sectors" if session == "asx" else "us_sectors", [])
    top_sectors = {normalise_sector(s.get("name")) for s in sectors if s.get("top")}
    return {
        "session":       session,
        "today":         datetime.now(AEST).date(),
        "qqqe_scenario": scan.get("qqqe_scenario", {}).get("scenario"),
        "rs_lookup":     rs_lookup,
        "rs_universe_ok": bool(rs_lookup),
        "top_sectors":   top_sectors,
        "bench_rs_raw":  None,
        "earnings":      {},
    }


def append_ledger(session: str, ctx: dict, evaluated: list) -> None:
    path = os.path.join(DATA_DIR, "idea_ledger.jsonl")
    date = ctx["today"].isoformat()
    # One row per ticker per session per day — re-runs replace nothing, they just skip
    existing = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing.add((r["date"], r["session"], r["ticker"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    with open(path, "a") as f:
        for e in evaluated:
            if (date, session, e["ticker"]) in existing:
                continue
            f.write(json.dumps({
                "date": date, "session": session, "ticker": e["ticker"],
                "price": e["metrics"]["price"], "passed": e["passed"],
                "killed_by": e["killed_by"], "gates_passed": e["gates_passed"],
                "qqqe_scenario": ctx["qqqe_scenario"],
            }) + "\n")


def run(session: str) -> dict:
    cfg = load_config()
    settings = cfg.get("settings", {})
    thresholds = {**DEFAULTS, **settings.get("gates", {})}
    ctx = build_context(session, cfg)

    cands = gather_candidates(session, cfg)
    print(f"Evaluating {len(cands)} {session.upper()} ideas through 6 gates...")
    if not cands:
        print("  No candidates — nothing to evaluate")

    bench = "^AXJO" if session == "asx" else "SPY"
    hists = download_histories([c["ticker"] for c in cands] + [bench]) if cands else {}
    bench_m = compute_metrics(hists[bench]) if bench in hists else None
    ctx["bench_rs_raw"] = bench_m["rs_raw"] if bench_m else None
    ctx["earnings"] = fetch_earnings_dates([c["ticker"] for c in cands], ctx["today"])

    evaluated, no_data = [], []
    for c in cands:
        h = hists.get(c["ticker"])
        m = compute_metrics(h) if h is not None else None
        if m is None:
            no_data.append(c["ticker"])
            continue
        evaluated.append(evaluate(c, m, ctx, thresholds, settings))

    survivors = sorted((e for e in evaluated if e["passed"]), key=lambda e: -e["score"])
    max_focus = int(settings.get("max_focus_list_size", 5))
    kill_counts = {g: sum(1 for e in evaluated if e["killed_by"] == g) for g in GATES}

    output = {
        "generated_at":  datetime.now(AEST).isoformat(),
        "session":       session,
        "qqqe_scenario": ctx["qqqe_scenario"],
        "rs_universe_ok": ctx["rs_universe_ok"],
        "evaluated":     len(evaluated),
        "survived":      len(survivors),
        "kill_counts":   kill_counts,
        "focus_list":    survivors[:max_focus],
        "bench_survivors": survivors[max_focus:],
        "near_misses":   sorted((e for e in evaluated if e["gates_passed"] == 5),
                                key=lambda e: e["ticker"]),
        "rejected":      [{"ticker": e["ticker"], "killed_by": e["killed_by"],
                           "reason": e["gates"][e["killed_by"]]["reason"]}
                          for e in evaluated if not e["passed"]],
        "no_data":       no_data,
        "thresholds":    thresholds,
    }

    with open(os.path.join(DATA_DIR, f"idea_gates_{session}.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    append_ledger(session, ctx, evaluated)

    print(f"  Survived: {len(survivors)}/{len(evaluated)}  |  killed by: "
          + ", ".join(f"{g}={n}" for g, n in kill_counts.items() if n))
    if not survivors:
        print("  Zero survivors — the gates did their job. No trade is a position.")
    for e in survivors[:max_focus]:
        p = e["plan"]
        print(f"  {e['ticker']:6s} score {e['score']:5.1f}  stop {p['stop']} ({p['stop_pct']}%)  size {p['position_pct']}%")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["asx", "us"], required=True)
    args = parser.parse_args()
    run(args.session)
