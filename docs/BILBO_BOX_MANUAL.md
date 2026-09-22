# The Bilbo Box — A First-Principles Manual

*For the `pre-scan` repository. Written to be read once end-to-end, then used as
a reference.*

---

## 0. What you are holding

Two things:

1. **An explanation.** Every rule in this strategy is derived here from the
   market behaviour it is trying to exploit, not asserted. If you understand
   section 1 through 7 you can rebuild the whole system from memory, and more
   importantly you can tell when it has stopped working versus when it is just
   losing.
2. **A working implementation** inside this repo: a scanner, a backtester, a
   paper journal, a dashboard view, and a test suite that pins the maths.

The strategy is not mine. The compression box comes from **Saty Mahajan**
([@satymahajan](https://x.com/satymahajan)), built on his free Phase Oscillator.
The systematic options rules, the backtesting and the public forward test come
from **The Milk Man** ([@MrMilkTrading](https://x.com/MrMilkTrading)). What is
mine is the derivation below and the code in this repository.

### Provenance — read this before you trust a number

| Source | Status |
|---|---|
| Saty Phase Oscillator formula | **Verified verbatim.** The published ThinkScript was retrieved and ported line-for-line. See §2. |
| Bilbo Box trade rules | **From your brief.** `milkmantrades.com` and the Substack were blocked by this environment's network egress proxy, so I could not re-read the primary write-ups. The rules implemented here are exactly the ones in your summary. Where a rule is silent, I made an engineering choice and tagged it `[MINE]`. |
| The 20-ticker universe | **Not published.** The study says "20 liquid large-caps" without naming them. `config/bilbo_config.json` ships a defensible stand-in. Swap it. |
| Backtest results (1,031 trades, 2019–2026, ~41% win rate, +11.9% of premium per trade) | **Theirs, not reproduced here.** Nothing in this repo verifies those figures, and the backtester here *cannot* — see §11. |

Every parameter in the config file carries one of three tags:
`[SATY]` (from the indicator source), `[RULE]` (a published Bilbo rule), or
`[MINE]` (my choice, because the rules do not pin it down). §9 is the full table.

---

## 1. The idea, from the ground up

### 1.1 Volatility is the thing that mean-reverts, not price

Price is close to a random walk. Volatility is not. Volatility **clusters**:
quiet begets quiet, violence begets violence, and — crucially — the transition
between the two regimes is not symmetric in time. A market can stay quiet for
days and then resolve in minutes.

That asymmetry is the entire edge. If you can identify "quiet" *while it is
still quiet*, you get to buy the transition rather than chase it.

### 1.2 What a range actually is

A sideways range is not indecision. It is an **auction in balance**: for every
price in the range, a buyer and a seller agreed. The high of the range is where
supply reliably overwhelmed demand; the low is where demand reliably overwhelmed
supply. Those two levels are not arbitrary lines — they are the only two prices
in the recent past where you *know* one side ran out.

So a range hands you three gifts:

- **A trigger** — the high. Trading above it means the supply that defined the
  range is gone.
- **A falsification level** — the low. Below it, whatever you thought was
  happening is not happening.
- **A measuring stick** — the height of the range, which tells you how much
  energy was stored.

Most setups give you one of those. A range gives you all three, and it gives
them to you *before* you have to act. That is why box strategies survive.

### 1.3 Why this becomes an options trade

Two facts compound:

1. When a quiet market resolves, it resolves faster than usual.
2. Option premium is priced off *recent* volatility, which during the quiet
   phase is low.

So you are buying a fast move at a price set by a slow one. That is the
structural reason to express this with a long call rather than shares.

It is also why the trade has to be *time-boxed*. The same low premium that makes
the entry cheap turns into a bleeding wound if the move does not come. Buying an
option is renting a thesis. Rent is due daily (see §5.3 for the actual numbers).

### 1.4 The one-sentence version

> Find a stock whose closes have gone quiet relative to its own bar ranges, draw
> the box those quiet candles made, buy a short-dated call when an hourly candle
> closes above the box, and get out fast on the stock — not the option — when the
> move fails, stalls, or gives most of itself back.

---

## 2. The measuring instrument: the Saty Phase Oscillator

You cannot eyeball "quiet". You need a definition that a machine can apply the
same way twice. Saty's is good, and here is why each piece is there.

The published source (ThinkScript) is reproduced faithfully in
`scripts/bilbo/indicators.py`. The essentials:

```
pivot              = EMA(close, 21)
above_pivot        = close >= pivot
oscillator_signal  = EMA( (close - pivot) / (3.0 * ATR(14)) * 100 , 3 )

bband_offset       = 2.0   * StDev(close, 21)
compression_thresh = pivot ± 2.0   * ATR(14)
expansion_thresh   = pivot ± 1.854 * ATR(14)

compression        = above_pivot ? bb_up - ct_up : ct_down - bb_down
expansion_zone     = above_pivot ? bb_up - et_up : et_down - bb_down
expansion          = compression[1] <= compression[0]
compression_tracker = (expansion and expansion_zone > 0) ? 0
                    : (compression <= 0) ? 1 : 0
```

### 2.1 Why a 21 EMA is the centre

You need an anchor to measure "away from". A simple moving average weights a
price from 21 bars ago the same as the last one, which makes it lag through
turns. An EMA of length 21 puts ~9% weight on the newest bar and decays
smoothly. Twenty-one hourly bars is about three sessions — recent enough to
track the current auction, long enough not to whip. The same 21 appears on the
daily chart as the trend gate (§4.4), which is a nice consistency: the same
notion of "the mean" at two time scales.

### 2.2 Why ATR is the ruler

A $3 move means nothing until you know whether the stock usually moves $0.30 or
$30 a bar. Average True Range answers that, and **True Range specifically**
(rather than high−low) because it counts overnight gaps: the distance actually
travelled, not just the distance travelled during the day.

ATR(14) with Wilder smoothing — `alpha = 1/14`, not `2/15` — is the original
Wilder definition and what ThinkScript's `ATR()` defaults to. Getting this wrong
shifts every compression flag slightly. The test suite pins it
(`test_wilder_rma_uses_one_over_n`).

### 2.3 What the oscillator value means

```
oscillator = (close − 21EMA) / (3 × ATR) × 100,  then smoothed over 3 bars
```

Read it as: **how far is price from its mean, in units of one-third of an ATR,
expressed as a percentage.** So:

- `±100` = price is a full **3 ATR** from the 21 EMA. Statistically stretched.
- `±61.8` = ~1.85 ATR out — Saty's distribution / accumulation boundary.
- `±23.6` = ~0.7 ATR out — the "launch box", price effectively at the mean.

(61.8 and 23.6 are Fibonacci numbers. They are conventional, not derived. They
do not matter to the Bilbo Box — only the compression flag does. This repo
reports the zone label as context, nothing more.)

The 3-bar EMA smoothing removes single-bar flicker at zone boundaries. It also
means the oscillator lags price by a bar or so. Again: irrelevant to the box,
because the box keys off the compression flag, not the oscillator line.

### 2.4 Compression, decoded — the part worth understanding

This is the heart of it, and it is more interesting than "the bands got narrow".

```
compressed  ⟺  2 × StDev(close, 21)  ≤  2 × ATR(14)
            ⟺  StDev(close, 21)      ≤  ATR(14)
```

A Bollinger band inside a Keltner-style ATR envelope. Now read what those two
quantities actually measure:

- `StDev(close, 21)` — how much the **closes scatter** over 21 bars. A
  *positional* measure.
- `ATR(14)` — how much **each individual bar travels**. A *per-bar* measure.

So compression fires when **the closes are clustered tightly relative to how far
the bars themselves are moving.** That is not "low volatility". That is
*overlap*: bars with real range that keep ending up in the same place. Churn.
Two-sided fighting inside a fixed area.

This distinction matters enormously, and it is the reason a naive
"low-volatility random walk" does **not** trigger compression — I verified this
while building the fixtures. A drifting quiet market keeps wandering, so its
closes scatter, so StDev stays high relative to ATR. Only a genuinely *balanced*
market — one being fought over — compresses.

Which is exactly the condition §1.2 said gives you a real trigger and a real
falsification level. The indicator is not measuring calm. It is measuring
**contested balance**. That is why a break out of it means something.

### 2.5 The expansion escape clause

```
expansion = compression[1] <= compression[0]
compression_tracker = (expansion and expansion_zone > 0) ? 0 : (compression <= 0)
```

`compression` is signed slack: how far the Bollinger edge sits *inside* the
2-ATR envelope. `expansion` is true when that slack stops shrinking — the bands
are pushing back out. The second envelope at **1.854 ATR** is a slightly tighter
tripwire; once the band clears it *and* is widening, the tracker drops to 0 even
if the raw `compression <= 0` test still passes.

In plain terms: **compression turns off a beat early, on the first evidence that
the squeeze is releasing.** It is a deliberate bias toward calling the end of
the squeeze too soon rather than too late — correct for a breakout strategy,
since the cost of being told late is missing the move.

One practical consequence, which I confirmed on real fixture data: **the
breakout candle itself is often still grey.** One big bar barely moves a 21-bar
standard deviation, while it does raise ATR. So entries frequently land one bar
after the impulse candle. That is not a bug in this implementation — it follows
from the published formula, and it is why the rule is "closes *out of
compression* **and** above the box high" rather than just "above the box high".

---

## 3. The box

### 3.1 Construction

> The box is the high and the low of the **first 5 grey candles**. After the
> fifth, it is frozen and never moves again.

Three questions, three answers:

**Why the first five and not all of them?** Because a box that keeps widening as
compression drags on is not a level, it is a moving target — and it would ratchet
your trigger further away exactly as the setup matures. Freezing makes the
trigger a commitment made in advance. Five hourly bars is most of a session:
enough for both sides to show up, short enough that the level stays relevant.

**What if compression ends before five bars?** The box freezes with what it has.
`box.min_box_bars` `[MINE]` (default 2) is the floor below which a "range" is
just one candle and gets discarded.

**What if compression runs for twenty bars?** The box stays frozen at the first
five. Price can wander above the frozen high while still grey — no entry, because
the entry rule requires a close *out of* compression. When the grey finally
drops, the first close above the frozen high is the trigger.

### 3.2 The state machine

`scripts/bilbo/boxes.py` implements exactly this, and nothing else:

```
NO BOX ──grey──▶ BUILDING ──5 grey bars, or the grey run ends──▶ ARMED
                                                                  │
      ┌──────────────┬──────────────┬─────────────┬───────────────┤
      ▼              ▼              ▼             ▼               ▼
 close > high   close > high    close < low   N bars pass   a NEW grey
 gates pass     gate fails      (long-only    doing nothing  run starts
 TRIGGERED      MISSED           invalidation)  EXPIRED       SUPERSEDED
                (logged,          INVALIDATED
                 not traded)
```

Design decisions I had to make, all `[MINE]`, all configurable:

- **A breakout consumes the box even when a gate rejects it.** If the move
  happened at 15:45, you missed it; the box is not still sitting there tomorrow
  waiting to be traded at a worse price. The event is logged as `MISSED` so you
  can audit what the gates cost you (§10.3).
- **An hourly close below the box low kills an armed box.** Long-only strategy;
  a box that has already broken down is not a long setup.
- **A new grey run supersedes the old box** — but only a *new* run. Grey bars
  that are a continuation of the run that built the box do not (this was a real
  bug, caught by `test_box_freezes_after_five_grey_candles_and_ignores_later_ones`).
- **Armed boxes expire after 30 hourly bars** (~1 week). The published rules do
  not say. Without an expiry, stale boxes accumulate forever.

---

## 4. The entry, gate by gate

> An hourly candle closes **out of compression** and **above the box high**,
> between **10:00 and 15:00 ET**, with the stock **above its daily 21 EMA**, and
> the chosen call's **bid-ask ≤ 5% of mid**. Long only. One position per signal.

Each gate removes a specific, nameable way this trade loses money.

### 4.1 "Closes above", not "trades above"

A wick through the box high is a probe — someone testing whether supply is still
there. A *close* above it is the hour's verdict after everyone who wanted to
fight it had sixty minutes to do so. Stops sit above obvious highs; touching
them is what a stop run does, and a stop run reverses. Requiring the close
throws away the fastest entries and almost all of the fake ones.

Cost: you enter perhaps 0.3–0.8% worse than the breakout price. Benefit: you
stop paying for every liquidity sweep. This is the single highest-value gate in
the system.

### 4.2 "Out of compression"

See §2.5. The grey flag must be gone. Without this you would buy the
mid-compression probe — the overlap bar that goes nowhere.

### 4.3 The 10:00–15:00 ET window

Both ends are about *who* is trading:

- **Before 10:00** you are in the opening auction. Overnight imbalance is being
  cleared, ranges are enormous, and a "breakout" at 09:45 is frequently just the
  first ten minutes of two-way discovery. The first hourly candle is noise
  wearing a costume.
- **After 15:00** two separate things bite. Liquidity thins into the close and
  option spreads widen right when you are trying to pay them. And a breakout
  with under an hour left gets no chance to follow through before an overnight
  gap you cannot manage — a gap-down through the box low skips your stop entirely.

In this implementation the window is tested against **the bar's close time**,
which is why `scripts/bilbo/data.py` restamps every bar: feeds label an hourly
bar with the time it *opened*. Get this wrong and every entry silently shifts by
one bar. `test_bars_are_restamped_to_their_close_and_clamped_to_the_bell` pins it.

With standard 09:30-anchored hourly bars, the eligible closes are **10:30,
11:30, 12:30, 13:30 and 14:30 ET** — five shots per day per ticker.

### 4.4 Above the daily 21 EMA

The box tells you about the last few hours. It says nothing about whether you
are buying a breakout in an uptrend or a bounce in a downtrend. Those have very
different forward distributions, and in a downtrend the same pattern usually
resolves as a lower high.

This is a *regime* filter, and it is the reason the strategy is long-only. There
is no claim that upside boxes work and downside ones do not — only that this
system trades one side, with one filter, and does not pretend to know the other.

`daily_ema_uses_prior_close` `[MINE]` defaults to **true**: the EMA is computed
through the *previous* completed session. Using today's still-forming daily bar
would let information from the future of the hourly bar leak into the decision.
Small detail, and exactly the kind of thing that flatters a backtest.

### 4.5 Bid-ask ≤ 5% of mid

The most underrated rule here. A 5% spread is a **5% round-trip haircut on the
entire position**, charged before the thesis gets a chance. On a trade whose
published average edge is about +11.9% of premium, a sloppy fill can eat most of
the year.

This gate is also what really defines the universe. "20 liquid large-caps" is not
about market cap — it is "names whose 28-DTE calls quote a penny or two wide".
That is the actual requirement. A $400bn company with a thin chain fails it.

### 4.6 One position per signal, and the daily cap

`one_position_per_ticker` and `max_new_positions_per_day` (default 3) are both
`[MINE]`. Compression is regime-driven: when the whole market goes quiet, ten
boxes arm at once and then all ten break out on the same macro print. Without a
cap you are not taking ten independent bets, you are taking one bet ten times.
Set the cap to your real tolerance for correlated exposure, not to your
enthusiasm.

---

## 5. Why a call, which call, and how many

### 5.1 The convexity trade, with real numbers

Worked example, computed with the Black-Scholes in `scripts/bilbo/options.py`
(you can rerun it yourself): stock **$200**, one strike OTM at **$205**, **28
DTE**, IV **30%**, r = 4.2%.

```
premium  $4.75        delta 0.41
```

Delta 0.41 means each contract behaves like 41 shares at the moment you buy it —
**$8,200 of stock exposure for $475 of capital, with a hard floor at zero.**
That is the deal. And because gamma is positive, the exposure grows as you are
proven right:

| What happens | Option | Change |
|---|---|---|
| +2% stock in 3 days | $6.19 | **+30%** |
| +3% stock in 3 days | $7.25 | **+53%** |
| +5% stock in 3 days | $9.66 | **+103%** |
| +1% stock in 7 days | $4.67 | **−2%** |
| flat for 3 days | $4.37 | −8% |
| flat for 7 days | $3.83 | −19% |
| flat for 14 days | $2.77 | **−42%** |
| −1.5% the next day | $3.50 | −26% |

Read the last four rows again. **Being right slowly is indistinguishable from
being wrong.** A +1% move that takes a week loses money. That single fact
generates three of this system's rules: the time stop, the trailing rule, and the
insistence on entering only after a decisive hourly close.

### 5.2 Why ~1 strike OTM

Strike choice is a dial between cost and convexity:

- **Deep ITM** (delta 0.8): expensive, barely convex. You have bought expensive
  stock.
- **Far OTM** (delta 0.15): cheap, enormous convexity, but needs a move far
  larger than a box breakout typically delivers. Most expire worthless.
- **One strike OTM** (delta ~0.40–0.45): the knee of the curve. Enough delta that
  a 2–3% move registers immediately, enough gamma that a 5% move doubles it,
  cheap enough that the whole ticket is 3–5% of the account.

"One strike" rather than "0.40 delta" is a deliberate simplification: it is
unambiguous at the terminal, and on a liquid large-cap the nearest OTM strike
lands near that delta anyway. `scripts/bilbo/options.py` walks the **real strike
ladder**, so it adapts automatically to $1, $2.50 and $5 grids.

### 5.3 Why ~28 DTE

Theta is not linear. An option loses time value roughly as √(time remaining), so
the daily bleed accelerates sharply in the final fortnight.

- **Under ~14 DTE**: theta dominates. You must be right almost immediately.
- **Over ~45 DTE**: you are paying for a month of optionality you plan to throw
  away after ten trading days, and vega exposure rises — your P&L starts
  depending on IV rather than on the stock.
- **~28 DTE**: the trade's whole intended life (≤ 10 trading days, ~14 calendar)
  sits in the *flatter* part of the decay curve. You exit with roughly two weeks
  of extrinsic value still unspent — value you sell to the next holder rather
  than donate to time.

The 21–37 day window exists because listed expiries are lumpy. The implementation
picks the expiry nearest 28 inside that window and skips the signal entirely if
none exists — rather than silently taking a 45-day contract.

### 5.4 Sizing: the premium *is* the stop

There is no stop-loss on the option. The exits are triggered by the stock (§6),
and between an exit trigger and a fill anything can happen. So size on the only
number that cannot betray you: **the premium paid is the maximum loss.**

Premium = 3–5% of account. With the example above and a $100k account:

```
8 contracts × $4.75 × 100 = $3,801  →  3.8% of the account
```

The arithmetic that actually matters:

- Max loss on one trade: **3.8%** — and that requires expiring worthless, which
  the box-low stop is designed to prevent. Realised losers typically give back
  40–60% of premium, so a *typical* loss is ~2%.
- Three concurrent positions: **~11.4%** of the account at theoretical risk.
- With the published ~41% win rate, you should expect **runs of 5–7 losers**.
  At ~2% a loser that is a 10–14% drawdown from nothing but normal variance.

If a 15% drawdown from routine variance would make you change the rules, your
size is wrong, not the strategy. Halve `premium_pct_*` in the config; the
mathematics is identical and the drawdown halves.

### 5.5 Where the money comes from — the expectancy identity

Take the published figures at face value for a moment (~41% win rate, ~+11.9% of
premium per trade). Those two numbers plus one assumption fix the third:

```
0.41 × W + 0.59 × L = +11.9%
```

| If the average loser is | then the average winner must be |
|---|---|
| −45% | **+94%** |
| −55% | **+108%** |
| −65% | **+123%** |

So the shape of this strategy is: **lose about half your premium six times out
of ten, roughly double it four times out of ten.** Every rule in §6 exists to
protect one side of that identity — the stop protects the −55%, the trail
protects the +108%.

And this is why the win rate is a distraction. A 41% win rate here is not a flaw
to be fixed; raising it by taking profits earlier would *cut the +108% winner*
and destroy the edge. The number to defend is the **ratio**, not the frequency.

---

## 6. Exits, and the one principle behind all of them

> **Every exit is judged on the stock price. None of them look at the option.**

This is the most important sentence in the system, so here is why.

The option's price contains three things: your thesis (delta), the clock (theta),
and the crowd's mood (vega). Only the first is the trade you put on. If you set
an exit on the option — "sell at −40%" — you will be stopped out by an IV crush
on a day the stock did exactly what you wanted. You will have been right and paid
for it.

The stock has no clock and no mood. Key every exit to it, and the only thing
that can take you out is the thesis being wrong, stalling, or having paid.

### 6.1 Exit 1 — invalidation: a 5-minute close below the box low

The box low is the price at which demand reliably beat supply. Below it, the
premise is void — not "unfavourable", *void*. This is not a money-management
stop, it is a **falsification test**.

Why the box low and not a percentage? Because a percentage stop is a statement
about your account and the box low is a statement about the market. Only one of
those has predictive content.

Why a **5-minute close** rather than the touch? Same logic as the entry (§4.1),
one time frame down. Stops cluster under obvious lows; the touch is frequently
the sweep. Five minutes is short enough to be fast, long enough to require
agreement.

### 6.2 Exit 2 — the trail: arm at +1 ATR, exit on a 75% giveback

Two-stage by design:

**Arming.** Nothing trails until the stock has gained **one full daily ATR**
from entry. Below that, movement is noise — a trail that tightens inside the
day's normal range is just a random exit generator. One ATR is the threshold at
which a move stops being ordinary. (Note: *daily* ATR, on an hourly-entry trade.
Deliberate: you are asking whether this is a real move on the chart that matters,
not on the chart you happened to enter on.)

**The giveback.** Once armed, exit on a 5-minute close that surrenders more than
**75% of the peak gain**:

```
exit if  close < entry + 0.25 × (peak − entry)
```

A 75% giveback is very loose by conventional standards, and that looseness is
the point. Refer back to §5.5: the edge lives in the +108% winners, which means
you must survive the pullbacks inside a trend. A tight trail converts a runner
into a scratch and the identity collapses. You are explicitly agreeing to hand
back three quarters of an open gain rather than risk being shaken out of the
one trade in four that pays for the year.

It is still a ratchet: `peak` only ever rises, so the exit level only ever rises.
A trade that goes +8% and comes back to +2% is closed at +2%, not at −1%.

### 6.3 Exit 3 — the time stop: 10 trading days

From §5.1: flat for 14 calendar days costs **42% of premium**. Ten trading days
(~14 calendar) is the point where theta stops being a rent you can afford and
becomes the trade itself.

The deeper argument: the whole thesis was that *quiet resolves fast*. If it has
been two weeks and the move has not come, the thesis did not merely fail to pay —
**it was wrong**, and you are now holding a decaying lottery ticket on a
different, slower market than the one you analysed.

### 6.4 Priority inside a single bar

Implemented in `strategy.evaluate_exit`, hardest rule first:

1. Close below the box low → out, no matter what else is true.
2. Update the peak.
3. Arm the trail if the peak has cleared +1 ATR.
4. If armed, test the giveback against the *updated* peak.
5. Test the time stop.

Step 1 before step 4 is a real decision: a bar that collapses from a high
straight through the box low is an invalidation, not a trail exit. The
classification matters for your own diagnostics later
(`test_box_low_beats_the_trail_inside_the_same_bar`).

---

## 7. Putting the whole thing on one card

| | Rule |
|---|---|
| **Universe** | ~20 large-caps whose 28-DTE calls quote inside 5% of mid |
| **Setup** | Hourly. Saty compression (grey). Box = high/low of the first 5 grey candles, then frozen |
| **Entry** | Hourly close out of compression **and** above box high, 10:00–15:00 ET, stock above daily 21 EMA, spread ≤ 5% of mid |
| **Instrument** | ~1 strike OTM call, ~28 DTE (21–37 window) |
| **Size** | Premium = 3–5% of account. Premium is the whole risk |
| **Exit 1** | 5-min close below box low |
| **Exit 2** | After +1 daily ATR, 5-min close giving back >75% of peak gain |
| **Exit 3** | 10 trading days, unconditionally |
| **Never** | Key any exit to the option price |

---

## 8. What is in this repository

### 8.1 Map

```
config/bilbo_config.json        every tunable, with provenance tags
scripts/
  bilbo/
    indicators.py               Saty Phase Oscillator, ported from the ThinkScript
    boxes.py                    the hourly state machine (§3.2)
    strategy.py                 entry gates + the stock-keyed exit engine
    options.py                  strike/expiry selection, spread gate, sizing, Black-Scholes
    data.py                     yfinance loaders, bar restamping, offline fixtures
    config.py                   typed config loading
  bilbo_scan.py                 the scanner CLI  -> data/bilbo_*.json
  bilbo_backtest.py             the replay CLI   -> data/bilbo_backtest.json
tests/test_bilbo.py             42 tests pinning the maths and the state machine
dashboard/index.html            three new views: Radar, Signals & Positions, Rules
.github/workflows/bilbo-scan.yml  hourly during the US session
docs/BILBO_BOX_MANUAL.md        this file
```

The layering is deliberate: `indicators` knows nothing about boxes, `boxes`
knows nothing about EMAs or time windows (gates are injected as a callback), and
`strategy` knows nothing about where bars came from. You can unit-test any layer
without a network, and you can replace the data source without touching the
rules.

### 8.2 Data flow

```
yfinance ─► data.py (normalise, restamp to bar CLOSE, US/Eastern)
              │
              ├─ hourly ─► indicators.phase_oscillator ─► boxes.run_box_machine ──┐
              │                                              ▲                    │
              └─ daily  ─► 21 EMA + ATR(14) ─► strategy.make_gate_fn ─────────────┘
                                                                                  │
                                                     events: TRIGGERED / MISSED / …
                                                                                  │
                        ┌─────────────────────────────────────────────────────────┤
                        ▼                                                         ▼
            live chain ─► options.select_* ─► spread gate ─► size_position   bilbo_state.json
                        │                                                     (the radar)
                        ▼
                 bilbo_signals.json ──► bilbo_positions.json (paper journal)
                                              ▲
                          5-min bars ─► strategy.evaluate_exit
```

**The scan is stateless.** Every run replays the full hourly history from
scratch, so the same bars always produce the same boxes and the same signals.
Nothing accumulates, nothing drifts. The only stateful artefact is the position
journal, and it is keyed by `TICKER-YYYYMMDDTHHMM` so re-running a scan can
never double-open a trade (`test_journal_opens_holds_and_closes_one_position`).

### 8.3 Commands

```bash
# scan the universe, price the contracts, mark the paper book
python scripts/bilbo_scan.py

# same, sized against a different account
python scripts/bilbo_scan.py --equity 25000

# just the state machine — no option chain calls
python scripts/bilbo_scan.py --no-options

# the whole pipeline on synthetic bars, no network at all
python scripts/bilbo_scan.py --offline

# one name, wider signal history
python scripts/bilbo_scan.py --tickers NVDA --lookback-bars 40

# replay the rules over the available hourly history
python scripts/bilbo_backtest.py --days 720 --csv trades.csv

# prove the engine still does what this manual says
python -m pytest tests/test_bilbo.py -q
```

### 8.4 The three output files

**`data/bilbo_state.json`** — the radar. Every ticker, its state
(`armed` / `building` / `compressing` / `idle`), the frozen box levels, distance
to the trigger in percent, the oscillator reading and zone, whether it is above
its daily 21 EMA, and a plain-English `note` saying what would have to happen
next.

**`data/bilbo_signals.json`** — three lists:
- `actionable` — the rules fired. Carries the exact contract (expiry, strike,
  bid/ask/mid, spread %), the position size, and the full exit plan (hard stop
  level, the price at which the trail arms, the time-stop date).
- `blocked` — fired, but the option was untradeable (wide spread, no expiry in
  the window, one contract over the size cap).
- `missed` — broke out, but a gate said no. **Read this list.** See §10.3.

**`data/bilbo_positions.json`** — the paper journal: `open` (with live peak,
whether the trail has armed, the current trail level, days held), `closed` (with
exit reason and the realised stock move), and aggregate `stats`.

### 8.5 The dashboard

Three views under **Bilbo Box** in the sidebar of `dashboard/index.html`:

- **Compression Radar** — the whole universe sorted with armed boxes first, so
  the top of the table is always the list of things that could trigger today.
- **Signals & Positions** — live signals with contracts and sizes, the missed-
  breakout audit trail, open paper positions with live trail levels, and closed
  trades.
- **The Rules** — the one-card summary, populated from your actual config, so
  what it shows is what the scanner is really doing.

All three read the JSON files directly and auto-refresh every 5 minutes with the
rest of the dashboard. They render correctly against the offline fixtures, which
is how the views were verified.

### 8.6 Automation

`.github/workflows/bilbo-scan.yml` runs at **:35 past each hour, 13:00–21:00 UTC,
weekdays** — five minutes after each US hourly bar closes, under both EDT and
EST. It runs the test suite first, then the scan, then commits the JSON so the
GitHub Pages dashboard updates. A `workflow_dispatch` input also runs the
backtest on demand.

Note the failure mode this design avoids: because the scan is stateless, a
missed run costs you nothing structurally. The next run rebuilds every box from
history. Only journal *entries* are time-sensitive — a signal that fired during
a skipped hour is not "fresh" on the next run and will not be opened. That is
the correct behaviour: it refuses to backfill a trade you never took.

---

## 9. Every parameter, its provenance, and what moving it does

`[SATY]` from the indicator source · `[RULE]` a published Bilbo rule ·
`[MINE]` my choice, because the rules do not pin it down.

### `oscillator` — do not touch without a reason

| Key | Default | Tag | Effect of changing it |
|---|---|---|---|
| `pivot_ema_length` | 21 | `[SATY]` | The mean everything is measured from. Shorter = more, noisier compressions. |
| `atr_length` | 14 | `[SATY]` | The ruler. Shorter reacts faster and flickers. |
| `atr_method` | wilder | `[SATY]` | `ema`/`sma` will *not* match the published indicator. |
| `oscillator_atr_multiple` | 3.0 | `[SATY]` | Rescales the oscillator only. Does not affect compression or any trade rule. |
| `oscillator_smoothing` | 3 | `[SATY]` | Display smoothing. Same — no trade impact. |
| `bband_length` / `bband_stdev_mult` | 21 / 2.0 | `[SATY]` | Raising the multiplier makes compression **rarer and stricter**. |
| `compression_atr_mult` | 2.0 | `[SATY]` | The envelope the bands must fit inside. Raising it makes compression **more common**. |
| `expansion_atr_mult` | 1.854 | `[SATY]` | The early-release tripwire. Lower = compression ends later. |
| `stdev_ddof` | 0 | `[SATY]` | ThinkScript's `StDev` is the population deviation. `1` is sample; the difference is tiny but it is not the published indicator. |

Changing anything in this block means you are no longer trading the Bilbo Box.
That may be fine — but measure it as a different strategy.

### `box`

| Key | Default | Tag | Effect |
|---|---|---|---|
| `max_box_bars` | 5 | `[RULE]` | More bars = wider box = further trigger, fewer and larger signals. |
| `min_box_bars` | 2 | `[MINE]` | Floor for a short grey run. Set to 5 to require the full five. |
| `box_expiry_bars` | 30 | `[MINE]` | How long an armed box stays live. Shorter = fewer stale triggers. |
| `kill_on_close_below_low` | true | `[MINE]` | Off = boxes survive a breakdown and can still fire long. Not recommended. |
| `new_compression_supersedes` | true | `[MINE]` | Whether a fresh grey run replaces an armed box. |

### `entry`

| Key | Default | Tag | Effect |
|---|---|---|---|
| `window_start` / `window_end` | 10:00 / 15:00 ET | `[RULE]` | Widening re-admits opening noise and the closing spread tax (§4.3). |
| `require_close_out_of_compression` | true | `[RULE]` | Off = you buy mid-squeeze probes. |
| `require_above_daily_ema` / `daily_ema_length` | true / 21 | `[RULE]` | The regime filter. Off = you will trade breakouts in downtrends. |
| `daily_ema_uses_prior_close` | true | `[MINE]` | **Leave true.** False leaks same-day information into the gate. |
| `max_spread_pct_of_mid` | 5.0 | `[RULE]` | Tightening to 3% is the single cheapest improvement if your universe supports it. |
| `one_position_per_ticker` | true | `[MINE]` | Prevents stacking the same name. |
| `max_new_positions_per_day` | 3 | `[MINE]` | Your correlated-exposure cap (§4.6). |

### `option`

| Key | Default | Tag | Effect |
|---|---|---|---|
| `target_dte` / `min_dte` / `max_dte` | 28 / 21 / 37 | `[RULE]` | Shorter = more convexity, more theta. See §5.3. |
| `strikes_otm` | 1 | `[RULE]` | 2+ is cheaper and needs a much bigger move. |
| `premium_pct_min/target/max` | 3 / 4 / 5 | `[RULE]` | **Your real risk dial.** Halve for half the drawdown. |
| `risk_free_rate` | 0.042 | `[MINE]` | Backtest pricing only. Live trades use real quotes. |

### `exit`

| Key | Default | Tag | Effect |
|---|---|---|---|
| `stop_bar_interval` | 5m | `[RULE]` | 1m = more shakeouts, 15m = more slippage past the level. |
| `trail_arm_atr_multiple` | 1.0 | `[RULE]` | Lower arms the trail inside normal noise (§6.2). |
| `trail_giveback_pct` | 75 | `[RULE]` | **Tightening this is the most tempting and most destructive change you can make.** It caps the winners the expectancy identity depends on (§5.5). |
| `time_stop_trading_days` | 10 | `[RULE]` | Longer = deep in the theta cliff (§5.3). |
| `daily_atr_length` | 14 | `[MINE]` | The ATR that arms the trail. |

### `backtest` — model assumptions, not rules

| Key | Default | Effect |
|---|---|---|
| `iv_mode` / `fixed_iv` | realized / 0.32 | Where the model's volatility comes from. |
| `iv_premium_multiple` | 1.15 | Implied usually prints above realised; this is a crude constant stand-in. |
| `round_trip_spread_pct` | 2.0 | Charged half on each side. Raise it to see how fragile the edge is to fills. |
| `exit_bar_interval` | 60m | 5-minute history only goes back ~60 days, so the backtest checks exits on hourly closes — which makes stops trigger **later** and is therefore mildly optimistic. |

---

## 10. How to actually use this in your system

### 10.1 Where it fits in `pre-scan`

Your existing scripts answer *"what is the market doing and what should be on my
list?"* — sector rotation, breadth, RS rankings, Finviz screens. They run before
the session.

Bilbo answers a different question: *"is there a mechanical trade right now?"*
It runs **during** the session, hourly, and it is the only part of the repo that
produces an instruction specific enough to act on without judgement.

Use them together, in this order:

1. **Regime first.** The dashboard's regime bar already shows breadth, $MMTW and
   MCO. The Bilbo rules contain a per-stock trend filter (daily 21 EMA) but no
   market-wide filter. In a washed-out tape — the `breadth.json` in this repo
   reads 28% and `BEARISH` as I write — a long-only breakout system is fighting
   the environment. My suggestion, and it is a suggestion because it is not part
   of the published rules: **cut `premium_pct_target` when breadth is bearish**
   rather than skip trades, so you keep the sample honest while risking less.
2. **Radar during the session.** Check the Compression Radar at the top of each
   hour. Anything `armed` with a small "to trigger" number is a candidate for
   the next bar.
3. **Signals for execution.** When something fires, `Signals & Positions` gives
   you the contract, the size, the stop and the trail arming price. Nothing is
   left to decide.
4. **Journal everything, trade nothing, at first.** See §10.2.

One caution on time zones: this is a **US-session** system, so it runs
11:30pm–6:00am AEST (12:30am–7:00am AEDT). The GitHub Action does the watching.
Do not plan to sit in front of it.

### 10.2 The paper-first protocol

The scanner writes a paper journal, not orders, and that is not a limitation to
route around — it is the first stage of the protocol:

1. **Weeks 1–4: observe only.** Let the journal fill. Every hour the Action
   runs, the book updates. Do not trade.
2. **After ~30 closed signals**, compare your journal to the published
   expectations: is the win rate anywhere near 40%? Are the stock legs behaving —
   winners running a multiple of the losers? Check `stats` in
   `bilbo_positions.json`.
3. **Then trade one contract**, regardless of what sizing says, for another 20
   trades. The gap between the journal's fill and yours *is* your real
   implementation cost. Measure it before you scale it.
4. **Only then** size to the 3–5% band — and at the bottom of it.

The one number that matters at step 3 is slippage: the journal assumes you got
the mid. You will not always.

### 10.3 Read the missed list

`bilbo_signals.json → missed` is the most useful diagnostic in the system. It
records every breakout the gates rejected and why.

- Mostly `time_window`? Your bar alignment may be wrong for your broker's chart,
  or the universe genuinely breaks out late. Investigate before widening.
- Mostly `above_daily_ema`? The market is in a downtrend and the system is
  correctly refusing to trade. That is the filter working, not failing.
- If the misses are systematically the *best* moves, the gate is miscalibrated.
  Change it deliberately, in config, and re-run the backtest — never in the
  moment, on one trade.

Without this list you would silently conclude "the strategy gave no signals",
which is a very different statement from "the strategy rejected eleven".

### 10.4 Make it yours

The highest-value edits, in order:

1. **The universe.** `config/bilbo_config.json → universe`. The shipped 20 are a
   stand-in. Replace with the names *you* watch and whose chains you have
   actually checked. This is the edit that matters most.
2. **Equity.** `account.equity`, or `--equity` per run.
3. **The daily cap.** `entry.max_new_positions_per_day` to your real tolerance.
4. **The premium band.** Start at 2–3%, not 3–5%.

To add a ticker to the radar without trading it, add it to the universe and set
`max_new_positions_per_day` to 0 — you get the full analysis and no journal
entries.

### 10.5 Hooking into the rest of the repo

If you want Bilbo signals in your existing email briefing, `bilbo_signals.json`
is a plain file: read `actionable` in `scripts/briefing_gen.py` and render the
contract line. The scanner is also importable —
`from bilbo_scan import scan; scan(cfg)` returns the whole result structure
without writing anything.

---

## 11. Limitations, honestly

### 11.1 What the backtester can and cannot tell you

`scripts/bilbo_backtest.py` prints two blocks, and they have very different
standing:

- **The stock leg is real.** Entries, exits, holding periods, MFE and the move
  distribution all come from actual bars. Trust these.
- **The option leg is a model.** No historical option quotes are available in
  this environment, so each trade is priced with Black-Scholes on *realised*
  volatility × 1.15. That ignores the volatility risk premium moving, IV crush
  after earnings, skew, and the real bid-ask. Treat modelled option P&L as an
  order-of-magnitude sanity check and nothing more. **The published 1,031-trade
  study used real quotes; this does not, and does not reproduce it.**

Three more hard limits:

- **History depth.** yfinance serves ~730 days of hourly bars. This cannot reach
  2019. Your backtest window is about two years, one regime.
- **Exit resolution.** 5-minute bars go back ~60 days, so the backtest checks
  exits on hourly closes by default. Hourly stops trigger *later* than 5-minute
  stops, which flatters the results.
- **Survivorship and selection.** A hand-picked list of today's large-caps is a
  list of things that already worked. This is not a neutral universe.

### 11.2 Things the rules do not handle

- **Earnings.** Nothing in the published rules avoids them, and nothing here
  does either. A 28-DTE call held through an earnings print is a volatility bet
  you did not intend to make, and IV crush can halve the option on a day the
  stock rises. If you add one filter of your own, make it this one.
- **Overnight gaps.** Every exit is a *close-based* rule during the session. A
  gap down through the box low fills you far below your stop. This is the
  structural risk of holding options overnight and it cannot be engineered away —
  only sized for (§5.4). It is also the strongest argument for the 15:00 cutoff.
- **Correlation.** See §4.6. The daily cap is a blunt instrument; it does not
  know that NVDA and AMD are the same bet.
- **Dividends and splits.** Bars are unadjusted (`auto_adjust=False`), which is
  correct for intraday levels but means a split during a backtest window will
  corrupt that ticker's history.

### 11.3 Things I could not verify

- The primary write-ups were **blocked by this environment's egress proxy**, so
  the trade rules implemented here are those in your brief, not re-read from
  source. If the site differs, the config file is where you reconcile it — most
  discrepancies will be one number.
- **Live data paths are untested here.** Yahoo Finance is also blocked in this
  sandbox, so `--offline` is the only mode I could execute end-to-end. The live
  loaders follow the same yfinance patterns as your existing, working
  `rs_ranker.py` and `breadth_monitor.py`, and the GitHub Action does have
  network access — but the first live run is the first live run. Watch it.
- **Option chain shape.** `yfinance`'s `option_chain` bid/ask can be stale or
  zero outside market hours. The spread gate treats a missing two-sided quote as
  a block, which is the safe direction, but expect `blocked` entries in the
  signals file when the scan runs near the bell.

### 11.4 The honest summary

This is a **long-only, momentum-biased, low-win-rate, fat-tailed** strategy. It
will lose money for weeks at a time and it will have a bad year in a bear market
by construction. Its edge, if real, is small per trade and comes from repetition
and from not interfering. The implementation here is careful, tested and
transparent about what it does not know — but implementation quality is not
evidence that the strategy works, and no amount of code changes that.

---

## 12. How this was verified

`python -m pytest tests/test_bilbo.py -q` runs the suite. What it actually pins:

- **The maths, independently.** The EMA and Wilder recursions are re-derived in
  the tests with plain Python loops, so a refactor that changes the smoothing
  fails even if it looks right.
- **The compression identity** — that `compressed` really is "Bollinger inside
  the ATR envelope", checked element-wise against a recomputation.
- **The state machine**, case by case: freezing after five, ignoring later grey
  candles, refusing entry while still grey, discarding runs that are too short,
  invalidation, supersession, expiry, and the miss-versus-trigger split.
- **Every exit rule**, including the priority order inside a single bar and the
  distinction between counting trading days and counting bars.
- **Bar restamping**, including the 30-minute stub bar that must clamp to 16:00
  rather than run past the bell.
- **The journal's idempotency** — running the same scan twice must not open two
  positions.

The dashboard views were rendered in a real browser against the offline
fixtures and checked visually; the JavaScript is syntax-checked.

What is **not** verified: anything requiring live market data, and anything
about whether the strategy is profitable.

---

## 13. Sources

- **Bilbo Box, full rules and study (v2)** — https://milkmantrades.com/bilbo-box-options-v2.html
- **Substack overview and backtesting intro** — https://playingfordoubles.substack.com/p/the-bilbo-box-breakout-strategy
- **Forward/paper signal log** — https://milkmantrades.com/bilbo-paper.html
- **Concise rules summary** — https://x.com/grok/status/2098597419256623158
- **The Milk Man** — https://x.com/MrMilkTrading (systematic rules, backtesting, public forward test)
- **Saty Mahajan** — https://x.com/satymahajan (the compression box concept and the Phase Oscillator)
- **Saty Phase Oscillator, official page** — https://www.satyland.com/phaseoscillator
- **Phase Oscillator source used for the port** — https://github.com/rishid/thinkscripts/blob/master/saty_phase_oscillator.tosts

---

*Nothing in this document or this repository is financial advice. It is a
description of a mechanical process and an implementation of it. The strategy is
long-only and loses money in adverse regimes by design. Size accordingly.*
