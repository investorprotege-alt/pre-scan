"""
indicators.py — the Saty Phase Oscillator, ported from the published ThinkScript.

Reference source (saty_phase_oscillator.tosts):

    def pivot = ExpAverage(close, 21);
    def above_pivot = close >= pivot;
    def oscillator_signal =
        Round(ExpAverage(((close - pivot) / (3.0 * ATR(14))) * 100, 3), 2);

    def bband_offset = 2.0 * STDev(close, 21);
    def bband_up   = pivot + bband_offset;
    def bband_down = pivot - bband_offset;
    def compression_threshold_up   = pivot + (2.0   * ATR(14));
    def compression_threshold_down = pivot - (2.0   * ATR(14));
    def expansion_threshold_up     = pivot + (1.854 * ATR(14));
    def expansion_threshold_down   = pivot - (1.854 * ATR(14));

    def compression    = if above_pivot then bband_up - compression_threshold_up
                                        else compression_threshold_down - bband_down;
    def expansion_zone = if above_pivot then bband_up - expansion_threshold_up
                                        else expansion_threshold_down - bband_down;
    def expansion = compression[1] <= compression[0];
    def compression_tracker = if expansion and expansion_zone > 0 then 0
                              else if compression <= 0 then 1
                              else 0;

`compression_tracker == 1` is the grey/magenta candle the strategy calls
"compression". Everything else in this repo keys off that one boolean.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    """ThinkScript ExpAverage: recursive from the first bar, alpha = 2/(n+1)."""
    return series.ewm(span=length, adjust=False).mean()


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """ThinkScript WildersAverage: recursive, alpha = 1/n."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """max(H-L, |H-Cprev|, |L-Cprev|); first bar has no prior close, so H-L."""
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1, skipna=True)


def atr(df: pd.DataFrame, length: int = 14, method: str = "wilder") -> pd.Series:
    """Average True Range. ThinkScript's ATR() defaults to Wilder smoothing."""
    tr = true_range(df)
    if method == "wilder":
        return wilder_rma(tr, length)
    if method == "sma":
        return tr.rolling(length).mean()
    if method == "ema":
        return ema(tr, length)
    raise ValueError(f"unknown atr method: {method!r}")


def rolling_stdev(series: pd.Series, length: int, ddof: int = 0) -> pd.Series:
    """ThinkScript StDev is the population deviation, hence ddof=0 by default."""
    return series.rolling(length, min_periods=length).std(ddof=ddof)


# ---------------------------------------------------------------------------
# The oscillator
# ---------------------------------------------------------------------------

def phase_oscillator(df: pd.DataFrame, p) -> pd.DataFrame:
    """
    Compute the Saty Phase Oscillator and its compression tracker.

    `df` needs lowercase open/high/low/close columns indexed by bar close time,
    oldest first. `p` is an OscillatorParams (see config.py).

    Returns a frame aligned to `df` with:
        pivot            21-EMA of close (the mean the whole system measures from)
        atr              ATR(14), the unit of distance
        oscillator       the -100..+100-ish phase reading
        bb_up/bb_down    Bollinger envelope (21, 2 sd)
        kc_up/kc_down    the ATR envelope the bands are compared against
        compression      signed slack: <= 0 means bands inside the ATR envelope
        expansion_zone   same measure against the 1.854 ATR envelope
        compressed       bool — the grey candle
    """
    close = df["close"].astype(float)

    pivot = ema(close, p.pivot_ema_length)
    above_pivot = close >= pivot
    a = atr(df, p.atr_length, p.atr_method)

    raw = ((close - pivot) / (p.oscillator_atr_multiple * a)) * 100.0
    oscillator = ema(raw, p.oscillator_smoothing).round(2)

    bband_offset = p.bband_stdev_mult * rolling_stdev(close, p.bband_length, p.stdev_ddof)
    bb_up = pivot + bband_offset
    bb_down = pivot - bband_offset

    kc_up = pivot + (p.compression_atr_mult * a)
    kc_down = pivot - (p.compression_atr_mult * a)
    ex_up = pivot + (p.expansion_atr_mult * a)
    ex_down = pivot - (p.expansion_atr_mult * a)

    compression = (bb_up - kc_up).where(above_pivot, kc_down - bb_down)
    expansion_zone = (bb_up - ex_up).where(above_pivot, ex_down - bb_down)

    # ThinkScript: expansion = compression[1] <= compression[0]
    # i.e. the slack is not shrinking — the bands are pushing outward.
    expanding = compression.shift(1) <= compression

    compressed = (~(expanding & (expansion_zone > 0))) & (compression <= 0)
    compressed = compressed.fillna(False).astype(bool)
    # Bars before the Bollinger window is full carry no verdict.
    compressed[bband_offset.isna()] = False

    return pd.DataFrame(
        {
            "pivot": pivot,
            "atr": a,
            "oscillator": oscillator,
            "bb_up": bb_up,
            "bb_down": bb_down,
            "kc_up": kc_up,
            "kc_down": kc_down,
            "compression": compression,
            "expansion_zone": expansion_zone,
            "compressed": compressed,
        },
        index=df.index,
    )


def phase_label(value: float) -> str:
    """The Saty zone the oscillator is sitting in — context, not a trade rule."""
    if value is None or pd.isna(value):
        return "unknown"
    if value >= 100:
        return "extended up"
    if value >= 61.8:
        return "distribution"
    if value > 23.6:
        return "mark up"
    if value >= -23.6:
        return "launch box"
    if value > -61.8:
        return "mark down"
    if value > -100:
        return "accumulation"
    return "extended down"
