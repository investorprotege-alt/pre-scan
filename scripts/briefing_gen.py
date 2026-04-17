"""
briefing_gen.py — Assembles the daily pre-market HTML briefing email.

Usage:
  python briefing_gen.py --session=asx
  python briefing_gen.py --session=us

Reads from:
  data/sector_scan.json
  data/breadth.json
  data/finviz_screener.json  (US session only)
  data/rs_rankings.json       (if available)
  config/watchlist.json

Outputs: data/briefing_{session}_{date}.html
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pytz

DATA_DIR    = Path(__file__).parent.parent / "data"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "watchlist.json"
DATA_DIR.mkdir(exist_ok=True)

AEST = pytz.timezone("Australia/Sydney")


def load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def roc_badge(val) -> str:
    if val is None:
        return "<span class='badge neutral'>N/A</span>"
    color = "green" if val > 0 else "red" if val < 0 else "neutral"
    sign = "+" if val > 0 else ""
    return f"<span class='badge {color}'>{sign}{val:.1f}%</span>"


def scenario_badge(n) -> str:
    colors = {1: "green", 2: "blue", 3: "orange", 4: "red"}
    labels = {
        1: "S1: Pullback in uptrend",
        2: "S2: Reclaim &amp; backtest",
        3: "S3: Reject &amp; higher low",
        4: "S4: Reject &amp; lower low",
    }
    color = colors.get(n, "neutral")
    label = labels.get(n, f"Scenario {n}")
    return f"<span class='badge {color}'>{label}</span>"


def exposure_badge(phase: str) -> str:
    color_map = {
        "Confirmed Uptrend Pullback": "green",
        "Out of Correction": "blue",
        "Breakdown": "red",
        "Overbought": "orange",
    }
    color = color_map.get(phase, "neutral")
    return f"<span class='badge {color}'>{phase}</span>"


def fmt(val, decimals=2, suffix="") -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def build_regime_section(scan: dict, breadth: dict) -> str:
    regime   = scan.get("market_regime", {})
    qqqe     = scan.get("qqqe_scenario", {})
    mco_data = breadth.get("mco_mcsi", {})

    vix      = regime.get("vix")
    vix_lbl  = regime.get("vix_regime", "")
    qqqe_s   = qqqe.get("scenario")
    exposure = regime.get("exposure_phase", "Unknown")

    mco  = mco_data.get("mco")
    mcsi = mco_data.get("mcsi")
    mco_sig  = mco_data.get("mco_signal", "")
    mcsi_sig = mco_data.get("mcsi_signal", "")

    breadth_score   = breadth.get("breadth_score", "—")
    breadth_summary = breadth.get("breadth_summary", "—")
    mmtw = breadth.get("mmtw_proxy", {}).get("mmtw_proxy")

    return f"""
<div class="regime-panel">
  <h2>Market Regime</h2>
  <div class="regime-grid">
    <div class="regime-item">
      <div class="label">Exposure Phase</div>
      {exposure_badge(exposure)}
    </div>
    <div class="regime-item">
      <div class="label">QQQE 21DMA Scenario</div>
      {scenario_badge(qqqe_s)}
    </div>
    <div class="regime-item">
      <div class="label">VIX</div>
      <div class="value">{fmt(vix, 1)} — <em>{vix_lbl}</em></div>
    </div>
    <div class="regime-item">
      <div class="label">$MMTW Proxy</div>
      <div class="value">{fmt(mmtw, 1)}%</div>
    </div>
    <div class="regime-item">
      <div class="label">MCO</div>
      <div class="value">{fmt(mco, 1)}<br><small>{mco_sig}</small></div>
    </div>
    <div class="regime-item">
      <div class="label">MCSI</div>
      <div class="value">{fmt(mcsi, 0)}<br><small>{mcsi_sig}</small></div>
    </div>
    <div class="regime-item">
      <div class="label">Breadth Score</div>
      <div class="value">{breadth_score}/100 — <strong>{breadth_summary}</strong></div>
    </div>
  </div>
  <div class="qqqe-detail">
    <strong>QQQE:</strong> {qqqe.get('description', '—')} |
    Price: {fmt(qqqe.get('price'), 2)} |
    High band: {fmt(qqqe.get('ema21_high_band'), 2)} |
    Mid: {fmt(qqqe.get('ema21_mid_band'), 2)} |
    Low band: {fmt(qqqe.get('ema21_low_band'), 2)}
  </div>
</div>
"""


def build_us_sector_table(scan: dict) -> str:
    sectors = scan.get("us_sectors", [])
    rows = ""
    for s in sectors:
        top_cls = " class='top-sector'" if s.get("top") else ""
        rows += f"""
        <tr{top_cls}>
          <td>{'⭐ ' if s.get('top') else ''}{s['name']}</td>
          <td>{s['ticker']}</td>
          <td>{roc_badge(s.get('1w_roc'))}</td>
          <td>{roc_badge(s.get('1m_roc'))}</td>
          <td>{roc_badge(s.get('3m_roc'))}</td>
        </tr>"""
    return f"""
<div class="section">
  <h2>US Sector Rankings</h2>
  <table>
    <thead><tr><th>Sector</th><th>ETF</th><th>1W</th><th>1M</th><th>3M</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def build_asx_sector_table(scan: dict) -> str:
    sectors = scan.get("asx_sectors", [])
    rows = ""
    for s in sectors:
        top_cls = " class='top-sector'" if s.get("top") else ""
        rows += f"""
        <tr{top_cls}>
          <td>{'⭐ ' if s.get('top') else ''}{s.get('name', s['ticker'])}</td>
          <td>{s['ticker']}</td>
          <td>{roc_badge(s.get('1w_roc'))}</td>
          <td>{roc_badge(s.get('1m_roc'))}</td>
        </tr>"""
    return f"""
<div class="section">
  <h2>ASX Sector Rankings</h2>
  <table>
    <thead><tr><th>Sector</th><th>Index</th><th>1W</th><th>1M</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def build_index_health_table(scan: dict) -> str:
    indices = scan.get("index_health", [])
    rows = ""
    for idx in indices:
        above_50  = idx.get("above_50sma")
        above_200 = idx.get("above_200sma")
        ma50_cls  = "green" if above_50 else "red" if above_50 is not None else "neutral"
        ma200_cls = "green" if above_200 else "red" if above_200 is not None else "neutral"
        atr_mult  = idx.get("atr_mult_from_50sma")
        atr_warn  = " ⚠️" if atr_mult and atr_mult > 4 else ""
        rows += f"""
        <tr>
          <td>{idx['name']}</td>
          <td>{fmt(idx.get('price'), 2)}</td>
          <td>{roc_badge(idx.get('1w_roc'))}</td>
          <td><span class='badge {ma50_cls}'>{'✓' if above_50 else '✗'} 50MA</span></td>
          <td><span class='badge {ma200_cls}'>{'✓' if above_200 else '✗'} 200MA</span></td>
          <td>{fmt(atr_mult, 1)}×{atr_warn}</td>
        </tr>"""
    return f"""
<div class="section">
  <h2>Index Health</h2>
  <table>
    <thead><tr><th>Index</th><th>Price</th><th>1W</th><th>50MA</th><th>200MA</th><th>ATR×/50MA</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="note">ATR× from 50MA: caution &gt;4×, no adds &gt;6-7×</p>
</div>"""


def build_commodities_section(scan: dict) -> str:
    comms = scan.get("commodities", [])
    rows = ""
    for c in comms:
        above_50  = c.get("above_50sma")
        above_200 = c.get("above_200sma")
        ma_cls = "green" if (above_50 and above_200) else "orange" if above_50 else "red"
        rows += f"""
        <tr>
          <td>{c['name']}</td>
          <td>{fmt(c.get('price'), 2)}</td>
          <td>{roc_badge(c.get('1w_roc'))}</td>
          <td>{roc_badge(c.get('1m_roc'))}</td>
          <td><span class='badge {ma_cls}'>{'✓' if above_50 and above_200 else '~' if above_50 else '✗'} Stage</span></td>
          <td>{c.get('leveraged_etf', '—')}</td>
        </tr>"""
    return f"""
<div class="section">
  <h2>Commodities + FX</h2>
  <table>
    <thead><tr><th>Asset</th><th>Price</th><th>1W</th><th>1M</th><th>Trend</th><th>ETF</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def build_asx_leaders_section(scan: dict, rs_data: dict) -> str:
    leaders = scan.get("asx_leaders", {})
    rs_map = {}
    for row in rs_data.get("asx_rankings", []):
        rs_map[row["ticker"]] = row.get("rs_score")

    html = '<div class="section"><h2>ASX Leader Status</h2>'
    for group_name, stocks in leaders.items():
        group_label = group_name.replace("_", " ").title()
        html += f'<h3>{group_label}</h3><table>'
        html += "<thead><tr><th>Stock</th><th>Price</th><th>10EMA</th><th>21EMA</th><th>50SMA</th><th>200MA</th><th>RS</th><th>Near MA?</th></tr></thead><tbody>"
        for s in stocks:
            above_200 = s.get("above_200sma")
            ma200_cls = "green" if above_200 else "red" if above_200 is not None else "neutral"
            rs = rs_map.get(s["ticker"])
            rs_cls = "green" if rs and rs >= 80 else "orange" if rs and rs >= 60 else "red" if rs else "neutral"

            near_ema10 = s.get("near_10ema", False)
            near_ema21 = s.get("near_21ema", False)
            near_sma50 = s.get("near_50sma", False)
            near_tags = []
            if near_ema10: near_tags.append("10EMA")
            if near_ema21: near_tags.append("21EMA")
            if near_sma50: near_tags.append("50SMA")
            near_str = ", ".join(near_tags) if near_tags else "—"
            near_cls = "orange" if near_tags else "neutral"

            html += f"""
            <tr>
              <td><strong>{s['ticker']}</strong><br><small>{s.get('name', '')}</small></td>
              <td>{fmt(s.get('price'), 2)}</td>
              <td>{fmt(s.get('ema10'), 2)}</td>
              <td>{fmt(s.get('ema21'), 2)}</td>
              <td>{fmt(s.get('sma50'), 2)}</td>
              <td><span class='badge {ma200_cls}'>{'✓' if above_200 else '✗'}</span></td>
              <td><span class='badge {rs_cls}'>{rs if rs else '—'}</span></td>
              <td><span class='badge {near_cls}'>{near_str}</span></td>
            </tr>"""
        html += "</tbody></table>"
    html += "</div>"
    return html


def build_finviz_section(finviz: dict, scan: dict) -> str:
    candidates = finviz.get("combined_candidates", [])
    top_sectors = [s["name"] for s in scan.get("us_sectors", []) if s.get("top")]

    if not candidates:
        return '<div class="section"><h2>Finviz Screener Results</h2><p>No data — run finviz_screener.py first.</p></div>'

    rows = ""
    for c in candidates[:25]:
        if "error" in c:
            continue
        screeners = c.get("screeners", [])
        s_badges = " ".join(
            f"<span class='badge {'green' if i == 0 else 'blue'}'>{s[:15]}</span>"
            for i, s in enumerate(screeners)
        )
        in_top = c.get("in_top_sector", False)
        row_cls = " class='highlight'" if in_top else ""
        rows += f"""
        <tr{row_cls}>
          <td><strong>{c.get('ticker', '')}</strong></td>
          <td>{c.get('company', '')[:25]}</td>
          <td>{c.get('sector', '')[:20]}</td>
          <td>{c.get('market_cap', '')}</td>
          <td>{c.get('price', '')}</td>
          <td>{c.get('change', '')}</td>
          <td>{s_badges}</td>
          <td>{'✓' if in_top else ''}</td>
        </tr>"""

    return f"""
<div class="section">
  <h2>Finviz Screener — Jeff Sun's 4 Momentum Scans</h2>
  <p>Top sectors today: <strong>{', '.join(top_sectors)}</strong> (highlighted rows = in top sector)</p>
  <table>
    <thead><tr>
      <th>Ticker</th><th>Company</th><th>Sector</th><th>Mkt Cap</th>
      <th>Price</th><th>Chg</th><th>Screeners Hit</th><th>Top Sector</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="note">Total candidates: {len(candidates)} | High confluence (2+): {finviz.get('high_confluence_count', '?')} | In top sectors: {finviz.get('in_top_sector_count', '?')}</p>
</div>"""


def build_focus_list_section(session: str, cfg: dict) -> str:
    key = "us_focus_list" if session == "us" else "asx_focus_list"
    focus = cfg.get(key, [])

    slots = []
    for i in range(5):
        if i < len(focus):
            item = focus[i]
            ticker = item.get("ticker", "")
            slots.append(f"""
            <div class="focus-card filled">
              <div class="focus-ticker">{ticker}</div>
              <div class="focus-meta">{item.get('notes', '')}</div>
            </div>""")
        else:
            slots.append(f"""
            <div class="focus-card empty">
              <div class="focus-empty">Slot {i+1} — Empty</div>
            </div>""")

    cards_html = "".join(slots)
    edit_note = f'<p class="note">Edit <code>config/watchlist.json</code> -> <code>{key}</code> to update focus list. Max 5 names.</p>'

    return f"""
<div class="section">
  <h2>Focus List (Max 5)</h2>
  <div class="focus-grid">{cards_html}</div>
  {edit_note}
</div>"""


def build_preentry_checklist() -> str:
    checks = [
        ("Market", "QQQE in Scenario 1 or 2"),
        ("Breadth", "MCO/MCSI supportive — not curling down"),
        ("Regime", "Regime dashboard not bearish"),
        ("Extension", "Stock within 1×ATR of 21DMA structure"),
        ("Jeff Sun #1", "LoD has NOT exceeded 60% of ATR"),
        ("Jeff Sun #2", "ATR% from 50-MA < 4× multiples"),
        ("Jeff Sun #4", "RVOL present"),
        ("Jeff Sun #5", "30 min post-open have elapsed"),
        ("Jeff Sun #6", "No earnings within 7 days"),
        ("Jeff Sun #7", "Not against declining 200-MA"),
        ("Jeff Sun #9", "Fewer than 3 new positions this session"),
    ]
    items = "".join(
        f'<li><label><input type="checkbox"> <strong>{cat}:</strong> {desc}</label></li>'
        for cat, desc in checks
    )
    return f"""
<div class="section checklist-section">
  <h2>Pre-Entry Checklist</h2>
  <p class="note">Apply to <strong>every</strong> trade before executing. All 11 must be checked.</p>
  <ul class="checklist">{items}</ul>
</div>"""


def build_tradingview_actions(scan: dict, session: str, cfg: dict) -> str:
    """Generate specific TradingView actions list from today's data."""
    actions = []

    if session == "asx":
        leaders = scan.get("asx_leaders", {})
        for group in leaders.values():
            for s in group:
                if s.get("near_10ema") or s.get("near_21ema") or s.get("near_50sma"):
                    ticker = s["ticker"]
                    ema10 = s.get("ema10")
                    ema21 = s.get("ema21")
                    sma50 = s.get("sma50")
                    alerts = []
                    if ema10: alerts.append(f"10EMA: {ema10:.2f}")
                    if ema21: alerts.append(f"21EMA: {ema21:.2f}")
                    if sma50: alerts.append(f"50SMA: {sma50:.2f}")
                    actions.append(f"<li><strong>{ticker}</strong> — Set TradingView alerts at: {', '.join(alerts)}</li>")

        top_sectors = [s.get("name", s["ticker"]) for s in scan.get("asx_sectors", []) if s.get("top")]
        if top_sectors:
            actions.insert(0, f"<li>Top ASX sectors: <strong>{', '.join(top_sectors)}</strong> — review sector ETF charts (XIJ, XMJ etc)</li>")

    else:  # US session
        top_sectors = [s["name"] for s in scan.get("us_sectors", []) if s.get("top")]
        if top_sectors:
            actions.append(f"<li>Top US sectors: <strong>{', '.join(top_sectors)}</strong> — open sector ETF charts (XLK, XLE etc)</li>")

        focus = cfg.get("us_focus_list", [])
        for item in focus:
            ticker = item.get("ticker", "")
            if ticker:
                actions.append(f"<li><strong>{ticker}</strong> — Set alerts at PDL, 21EMA, key range lows on TradingView</li>")

        qqqe_s = scan.get("qqqe_scenario", {}).get("scenario")
        if qqqe_s:
            desc = scan.get("qqqe_scenario", {}).get("description", "")
            actions.append(f"<li>Check <strong>QQQE</strong> 21DMA structure on TradingView — currently {desc}</li>")

        actions.append("<li>Pull up VARS histogram on focus list names</li>")
        actions.append("<li>Check Industry Group Strength (@amphtrading) for top sectors</li>")

    if not actions:
        actions = ["<li>No specific alerts needed — no leaders near key MAs today</li>"]

    items_html = "".join(actions)
    return f"""
<div class="section">
  <h2>TradingView Actions</h2>
  <p class="note">Complete these on TradingView after reviewing this briefing.</p>
  <ul>{items_html}</ul>
</div>"""


CSS = """
<style>
  /* === Light theme — email-safe, no CSS variables, explicit colors throughout === */
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
         background: #f3f4f6; color: #111827; margin: 0; padding: 16px; }
  .container { max-width: 960px; margin: 0 auto; background: #ffffff;
               border: 1px solid #d1d5db; border-radius: 10px; padding: 24px; }

  /* --- Headings --- */
  h1 { font-size: 1.45em; color: #111827; border-bottom: 2px solid #2563eb;
       padding-bottom: 10px; margin: 0 0 6px 0; }
  h2 { font-size: 1em; font-weight: 700; color: #2563eb;
       margin: 18px 0 10px 0; text-transform: uppercase; letter-spacing: 0.04em; }
  h3 { font-size: 0.9em; font-weight: 700; color: #7c3aed; margin: 12px 0 6px 0; }

  /* --- Sections --- */
  .section { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
             padding: 14px 16px; margin-bottom: 14px; }
  .regime-panel { background: #eff6ff; border: 2px solid #2563eb;
                  border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }

  /* --- Regime grid (table fallback for email) --- */
  .regime-grid { display: table; width: 100%; border-spacing: 8px; margin-bottom: 8px; }
  .regime-item { display: inline-block; background: #ffffff; border: 1px solid #dbeafe;
                 border-radius: 6px; padding: 8px 12px; margin: 4px; min-width: 130px;
                 vertical-align: top; }
  .label { font-size: 0.7em; color: #6b7280; text-transform: uppercase;
           letter-spacing: 0.06em; margin-bottom: 3px; font-weight: 600; }
  .value { font-size: 0.9em; color: #111827; font-weight: 500; }
  .qqqe-detail { font-size: 0.78em; color: #6b7280; border-top: 1px solid #dbeafe;
                 padding-top: 8px; margin-top: 6px; }

  /* --- Tables --- */
  table { width: 100%; border-collapse: collapse; font-size: 0.83em; }
  th { background: #f3f4f6; color: #374151; text-align: left; padding: 7px 10px;
       font-weight: 700; font-size: 0.8em; text-transform: uppercase;
       letter-spacing: 0.04em; border-bottom: 2px solid #d1d5db; }
  td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; color: #1f2937; }
  tr.top-sector td { background: #f0fdf4; font-weight: 600; }
  tr.highlight td  { background: #eff6ff; }

  /* --- Badges (pill labels) --- */
  .badge { display: inline-block; padding: 2px 9px; border-radius: 20px;
           font-size: 0.75em; font-weight: 700; white-space: nowrap; }
  .badge.green   { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
  .badge.red     { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
  .badge.orange  { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
  .badge.blue    { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
  .badge.neutral { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; }

  /* --- Notes & meta --- */
  .note { font-size: 0.78em; color: #6b7280; margin: 5px 0; }
  .header-meta { font-size: 0.83em; color: #6b7280; margin-bottom: 14px; }
  code { background: #f3f4f6; color: #111827; padding: 1px 5px;
         border-radius: 3px; font-size: 0.85em; border: 1px solid #e5e7eb; }

  /* --- Checklist --- */
  .checklist-section ul.checklist { list-style: none; padding: 0; margin: 0; }
  .checklist-section ul.checklist li { padding: 7px 0; border-bottom: 1px solid #e5e7eb; }
  .checklist-section ul.checklist li:last-child { border-bottom: none; }
  .checklist-section ul.checklist li label { display: flex; gap: 10px; cursor: pointer;
                                              align-items: flex-start; color: #1f2937; }
  .checklist-section input[type=checkbox] { margin-top: 2px; width: 15px; height: 15px;
                                             accent-color: #2563eb; flex-shrink: 0; }

  /* --- Focus list cards --- */
  .focus-grid { display: table; width: 100%; }
  .focus-card { display: inline-block; border-radius: 6px; padding: 10px 8px;
                text-align: center; width: 18%; margin: 0 1%; vertical-align: top; }
  .focus-card.filled { background: #f0fdf4; border: 1px solid #86efac; }
  .focus-card.empty  { background: #f9fafb; border: 2px dashed #d1d5db; }
  .focus-ticker { font-size: 1.05em; font-weight: 800; color: #166534; }
  .focus-meta   { font-size: 0.7em; color: #6b7280; margin-top: 3px; }
  .focus-empty  { font-size: 0.78em; color: #9ca3af; padding: 8px 0; }

  /* --- Lists --- */
  ul { padding-left: 20px; line-height: 1.9; color: #1f2937; }
  li strong { color: #111827; }
  ol { padding-left: 20px; line-height: 2; color: #1f2937; }
</style>
"""


def generate_briefing(session: str) -> str:
    scan     = load_json("sector_scan.json")
    breadth  = load_json("breadth.json")
    finviz   = load_json("finviz_screener.json") if session == "us" else {}
    rs_data  = load_json("rs_rankings.json")
    cfg      = load_config()

    now_aest = datetime.now(AEST)
    session_label = "ASX Pre-Market" if session == "asx" else "US Pre-Market"
    generated_at = scan.get("generated_at", now_aest.isoformat())

    body_parts = [
        build_regime_section(scan, breadth),
    ]

    if session == "asx":
        body_parts += [
            build_asx_sector_table(scan),
            build_index_health_table(scan),
            build_commodities_section(scan),
            build_asx_leaders_section(scan, rs_data),
        ]
    else:
        body_parts += [
            build_us_sector_table(scan),
            build_index_health_table(scan),
            build_commodities_section(scan),
            build_finviz_section(finviz, scan),
        ]

    body_parts += [
        build_focus_list_section(session, cfg),
        build_preentry_checklist(),
        build_tradingview_actions(scan, session, cfg),
    ]

    body = "\n".join(body_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{session_label} Briefing — {now_aest.strftime('%d %b %Y')}</title>
  {CSS}
</head>
<body>
  <div class="container">
    <h1>{session_label} Briefing — {now_aest.strftime('%A %d %b %Y, %H:%M AEST')}</h1>
    <div class="header-meta">
      Data generated: {generated_at} |
      Regime: <strong>{scan.get('market_regime', {}).get('regime', '—')}</strong>
    </div>
    {body}
  </div>
</body>
</html>"""

    date_str = now_aest.strftime("%Y%m%d")
    out_path = DATA_DIR / f"briefing_{session}_{date_str}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["asx", "us"], required=True)
    args = parser.parse_args()
    path = generate_briefing(args.session)
    print(f"Briefing ready: {path}")
