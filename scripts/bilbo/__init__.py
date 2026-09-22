"""
bilbo — a from-scratch implementation of the Bilbo Box compression-breakout
strategy (Saty Mahajan's compression box; options rules systematised by
The Milk Man / @MrMilkTrading).

Layers, bottom up:
  indicators.py  pure maths — EMA, Wilder ATR, the Saty Phase Oscillator and
                 its compression tracker, ported line-for-line from the
                 published ThinkScript.
  boxes.py       the hourly state machine — compression runs become boxes,
                 boxes freeze, arm, trigger or die.
  options.py     contract selection, spread gate, premium-based sizing and a
                 dependency-free Black-Scholes for the backtest proxy.
  strategy.py    the gates and the stock-price-keyed exit engine.
  data.py        yfinance loaders + an offline fixture generator.
  config.py      loads config/bilbo_config.json into typed params.

Nothing in here places an order. The output is a signal file, a position
journal and a dashboard view.
"""

__all__ = ["indicators", "boxes", "options", "strategy", "data", "config"]
