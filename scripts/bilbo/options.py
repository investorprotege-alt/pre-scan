"""
options.py — contract selection, the liquidity gate, premium-based sizing, and
a dependency-free Black-Scholes used only by the backtest proxy.

Nothing here needs scipy: the normal CDF comes from math.erf.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Pricing (backtest proxy only — live trades use real quotes)
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(spot: float, strike: float, years: float, rate: float, vol: float) -> float:
    """European call. Degenerates to intrinsic value when T or vol hits zero."""
    if years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike)
    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-rate * years) * norm_cdf(d2)


def bs_call_delta(spot: float, strike: float, years: float, rate: float, vol: float) -> float:
    if years <= 0 or vol <= 0:
        return 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
    return norm_cdf(d1)


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------

def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def select_expiry(expiries: Iterable, asof, p) -> Optional[dict]:
    """
    Nearest expiry to `target_dte` inside the [min_dte, max_dte] window.
    Returns {"expiry": date, "dte": int} or None when the window is empty.
    """
    asof = _as_date(asof)
    candidates = []
    for e in expiries:
        try:
            d = _as_date(e)
        except (ValueError, TypeError):
            continue
        dte = (d - asof).days
        if p.min_dte <= dte <= p.max_dte:
            candidates.append((abs(dte - p.target_dte), dte, d))
    if not candidates:
        return None
    candidates.sort()
    _, dte, d = candidates[0]
    return {"expiry": d, "dte": dte}


def select_strike(strikes: Iterable[float], spot: float, p) -> Optional[float]:
    """
    '~1 strike OTM' = the first listed strike above spot, then step `strikes_otm - 1`
    rungs further out. Uses the real strike ladder, so it adapts to $1/$2.50/$5 grids.
    """
    above = sorted(float(s) for s in strikes if float(s) > spot)
    if not above:
        return None
    idx = max(0, int(p.strikes_otm) - 1)
    return above[min(idx, len(above) - 1)]


def synthetic_strike(spot: float, p) -> float:
    """Strike ladder stand-in for the backtest, where no chain exists."""
    step = 1.0 if spot < 50 else 2.5 if spot < 200 else 5.0
    first_above = math.floor(spot / step) * step + step
    return round(first_above + step * max(0, int(p.strikes_otm) - 1), 2)


# ---------------------------------------------------------------------------
# Gates and sizing
# ---------------------------------------------------------------------------

def quote_quality(bid: float, ask: float, max_spread_pct: float) -> dict:
    """The liquidity gate: bid-ask must be <= max_spread_pct of the mid."""
    if bid is None or ask is None or ask <= 0 or bid <= 0 or ask < bid:
        return {"ok": False, "mid": None, "spread_pct": None, "reason": "no two-sided quote"}
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0
    return {
        "ok": spread_pct <= max_spread_pct,
        "mid": round(mid, 4),
        "spread_pct": round(spread_pct, 2),
        "reason": None if spread_pct <= max_spread_pct
                  else f"spread {spread_pct:.1f}% > {max_spread_pct:.1f}% of mid",
    }


def size_position(premium: float, equity: float, p) -> dict:
    """
    Contracts such that total premium lands inside the [min, max] % band, as
    close to target as the lot size allows.

    Premium is the *whole* risk: there is no stop-loss on the option, so this
    number is the most the trade can lose.
    """
    per_contract = premium * p.contract_multiplier
    if per_contract <= 0 or equity <= 0:
        return {"contracts": 0, "cost": 0.0, "pct_of_account": 0.0,
                "reason": "invalid premium or equity"}

    target_cash = equity * p.premium_pct_target / 100.0
    raw = target_cash / per_contract
    options = {max(1, int(math.floor(raw))), max(1, int(math.ceil(raw)))}

    best = None
    for n in sorted(options):
        pct = n * per_contract / equity * 100.0
        if p.premium_pct_min <= pct <= p.premium_pct_max:
            score = abs(pct - p.premium_pct_target)
            if best is None or score < best[0]:
                best = (score, n, pct)

    if best is None:
        one_pct = per_contract / equity * 100.0
        if one_pct > p.premium_pct_max:
            return {"contracts": 0, "cost": 0.0, "pct_of_account": round(one_pct, 2),
                    "reason": f"one contract is {one_pct:.1f}% of account, "
                              f"above the {p.premium_pct_max:.0f}% cap"}
        n = max(1, int(round(raw)))
        pct = n * per_contract / equity * 100.0
        return {"contracts": n, "cost": round(n * per_contract, 2),
                "pct_of_account": round(pct, 2),
                "reason": "lot size cannot land inside the band; nearest fit"}

    _, n, pct = best
    return {"contracts": n, "cost": round(n * per_contract, 2),
            "pct_of_account": round(pct, 2), "reason": None}


def occ_symbol(ticker: str, expiry, strike: float, right: str = "C") -> str:
    """Human-readable contract id, e.g. AAPL 2026-10-16 C240."""
    d = _as_date(expiry)
    strike_txt = f"{strike:g}"
    return f"{ticker} {d.isoformat()} {right}{strike_txt}"
