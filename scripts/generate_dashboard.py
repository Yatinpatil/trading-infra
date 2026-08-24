"""Regenerates a local, self-contained HTML dashboard for every paper
trading account in the project's SQLite store (db/) -- current
equity/cash, open positions, recent trades, and a comparison equity chart
across accounts.

No server, no network calls, no CSP restrictions to work around (this is a
local file opened directly in a browser, not a hosted artifact) -- run it
any time to refresh, or let scripts/run_daily_paper_trading.py call it
automatically after each day's step.

    python scripts/generate_dashboard.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.accounts import ACCOUNTS, load_account_state

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "dashboard" / "index.html"


def load_account(meta: dict) -> dict:
    state = load_account_state(meta, trade_limit=10)
    state["recent_trades"] = state.pop("trades")
    return state


def render(accounts: list[dict]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_json = json.dumps(accounts, default=str)

    rows_html = []
    for a in accounts:
        if not a["started"]:
            status_badge = '<span class="badge badge-muted">not started</span>'
            equity_cell = "&mdash;"
            change_cell = "&mdash;"
        else:
            badge_class = {"current": "badge-good", "stale": "badge-warn"}.get(a["status"], "badge-muted")
            badge_text = "current" if a["status"] == "current" else f"stale ({a['age_days']}d)"
            status_badge = f'<span class="badge {badge_class}">{badge_text}</span>'
            equity_cell = f"&#8377;{a['equity']:,.0f}"
            change_cell = f"{a['today_change_pct']:+.2%}" if a["today_change_pct"] is not None else "&mdash;"

        rows_html.append(f"""
        <tr>
          <td><span class="swatch" style="background:{a['color_light']}"></span>{a['label']}</td>
          <td class="num">{equity_cell}</td>
          <td class="num">{change_cell}</td>
          <td class="num">{a['num_open_positions'] if a['started'] else '&mdash;'}</td>
          <td class="num">{a['num_trades'] if a['started'] else '&mdash;'}</td>
          <td>{a['last_run_date'] or '&mdash;'}</td>
          <td>{status_badge}</td>
        </tr>""")

    positions_html = []
    for a in accounts:
        if not a["positions"]:
            continue
        rows = "".join(
            f"""<tr><td>{p['symbol']}</td><td class="num">{p['quantity']}</td>
                <td class="num">&#8377;{p['entry_price']:,.2f}</td><td>{p['entry_date']}</td>
                <td class="num">{f"&#8377;{p['stop_price']:,.2f}" if p['stop_price'] else '&mdash;'}</td></tr>"""
            for p in a["positions"]
        )
        positions_html.append(f"""
        <div class="detail-card">
          <h3><span class="swatch" style="background:{a['color_light']}"></span>{a['label']} &mdash; open positions</h3>
          <table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Entry date</th><th>Stop</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>""")

    trades_html = []
    for a in accounts:
        if not a["recent_trades"]:
            continue
        rows = "".join(
            f"""<tr><td>{t['symbol']}</td><td>{t['entry_date']}</td><td>{t['exit_date']}</td>
                <td class="num {'pnl-pos' if t['pnl'] >= 0 else 'pnl-neg'}">&#8377;{t['pnl']:,.0f}</td>
                <td class="num {'pnl-pos' if t['pnl'] >= 0 else 'pnl-neg'}">{t['pnl_pct']:+.2%}</td>
                <td>{t['exit_reason']}</td></tr>"""
            for t in a["recent_trades"]
        )
        trades_html.append(f"""
        <div class="detail-card">
          <h3><span class="swatch" style="background:{a['color_light']}"></span>{a['label']} &mdash; recent trades</h3>
          <table><thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>P&amp;L</th><th>P&amp;L %</th><th>Reason</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Paper Trading Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --bg: #F8F5EF; --surface: #FFFFFF; --surface-2: #F1ECE2; --border: #E3DBC9;
  --ink: #201C15; --ink-2: #4A4335; --ink-muted: #857C6C; --accent: #9C7D22;
  --good: #2E8B57; --warn: #B5651D; --shadow: 0 1px 2px rgba(32,28,21,0.04), 0 8px 24px -12px rgba(32,28,21,0.14);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #161310; --surface: #1E1A15; --surface-2: #241F18; --border: #332C22;
    --ink: #F1ECE0; --ink-2: #C4BBA9; --ink-muted: #8B8271; --accent: #C7A43A;
    --good: #4CAF7D; --warn: #E0954D; --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px -14px rgba(0,0,0,0.6);
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.55;
}}
.mono {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 40px 24px 72px; }}
.eyebrow {{ font-family: ui-monospace, Consolas, monospace; font-size: 12px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-muted); display: flex; align-items: center; gap: 8px; }}
.eyebrow .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }}
h1 {{ font-size: 28px; font-weight: 700; margin: 10px 0 6px; }}
.subhead {{ color: var(--ink-2); margin: 0; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; box-shadow: var(--shadow); }}
.section-head {{ display: flex; justify-content: space-between; align-items: baseline; margin: 40px 0 14px; }}
.section-head h2 {{ font-size: 18px; margin: 0; }}
.section-head .hint {{ font-family: ui-monospace, Consolas, monospace; font-size: 12px; color: var(--ink-muted); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
th {{ font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-muted); font-weight: 500; }}
td.num, th.num {{ text-align: right; font-family: ui-monospace, Consolas, monospace; font-variant-numeric: tabular-nums; }}
tbody tr:last-child td {{ border-bottom: none; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; vertical-align: middle; }}
.badge {{ display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }}
.badge-good {{ background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }}
.badge-warn {{ background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }}
.badge-muted {{ background: var(--surface-2); color: var(--ink-muted); }}
.pnl-pos {{ color: var(--good); }}
.pnl-neg {{ color: #C0392B; }}
.detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
.detail-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; box-shadow: var(--shadow); }}
.detail-card h3 {{ font-size: 14px; margin: 0 0 10px; display: flex; align-items: center; }}
.chart-card {{ padding: 16px 18px 8px; }}
#chart-svg {{ width: 100%; overflow: visible; display: block; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px; margin-bottom: 8px; font-family: ui-monospace, Consolas, monospace; }}
.legend span.name {{ color: var(--ink-2); }}
.axis-label {{ font-family: ui-monospace, Consolas, monospace; font-size: 10.5px; fill: var(--ink-muted); }}
.gridline {{ stroke: var(--border); stroke-width: 1; }}
footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); font-family: ui-monospace, Consolas, monospace;
  font-size: 11.5px; color: var(--ink-muted); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow"><span class="dot"></span>PAPER TRADING &middot; NIFTY 50 &middot; LIVE STATE</div>
  <h1>Paper Trading Dashboard</h1>
  <p class="subhead">5 strategies, one &#8377;10L account each, refreshed after every trading day's close.</p>

  <div class="section-head"><h2>Accounts</h2><span class="hint">generated {generated_at}</span></div>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th>Strategy</th><th class="num">Equity</th><th class="num">Today</th>
        <th class="num">Open</th><th class="num">Trades</th><th>Last run</th><th>Status</th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>

  <div class="section-head"><h2>Equity curves</h2><span class="hint">indexed to 100 at each account's start</span></div>
  <div class="card chart-card">
    <div class="legend" id="legend"></div>
    <svg id="chart-svg" viewBox="0 0 1080 380"></svg>
  </div>

  <div class="section-head"><h2>Open positions</h2></div>
  <div class="detail-grid">{"".join(positions_html) if positions_html else '<p class="subhead">No open positions in any account.</p>'}</div>

  <div class="section-head"><h2>Recent trades</h2></div>
  <div class="detail-grid">{"".join(trades_html) if trades_html else '<p class="subhead">No closed trades yet in any account.</p>'}</div>

  <footer>run_daily_paper_trading.py &middot; execution/state/ &middot; regenerate with scripts/generate_dashboard.py</footer>
</div>

<script id="account-data" type="application/json">{data_json}</script>
<script>
(function () {{
  var accounts = JSON.parse(document.getElementById('account-data').textContent).filter(function (a) {{ return a.started && a.dates.length > 1; }});
  var svg = document.getElementById('chart-svg');
  var svgNS = 'http://www.w3.org/2000/svg';
  var VB_W = 1080, VB_H = 380, PAD = {{ l: 46, r: 70, t: 14, b: 26 }};

  function el(tag, attrs) {{
    var e = document.createElementNS(svgNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }}

  var legendEl = document.getElementById('legend');
  accounts.forEach(function (a) {{
    var wrap = document.createElement('span');
    var sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = a.color_light;
    var isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    sw.style.background = isDark ? a.color_dark : a.color_light;
    var name = document.createElement('span'); name.className = 'name'; name.textContent = a.label;
    wrap.appendChild(sw); wrap.appendChild(name);
    legendEl.appendChild(wrap);
  }});

  if (!accounts.length) {{
    var msg = el('text', {{ x: VB_W / 2, y: VB_H / 2, 'text-anchor': 'middle', class: 'axis-label' }});
    msg.textContent = 'No account has enough history yet.';
    svg.appendChild(msg);
    return;
  }}

  var isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  var allDates = accounts[0].dates;
  var maxLen = Math.max.apply(null, accounts.map(function (a) {{ return a.dates.length; }}));

  var indexed = accounts.map(function (a) {{
    var base = a.values[0];
    return a.values.map(function (v) {{ return (v / base) * 100; }});
  }});

  var allVals = [].concat.apply([], indexed);
  var yMin = Math.min(100, Math.min.apply(null, allVals));
  var yMax = Math.max.apply(null, allVals);
  var span = (yMax - yMin) || 1;
  yMin -= span * 0.06; yMax += span * 0.06;

  var innerW = VB_W - PAD.l - PAD.r, innerH = VB_H - PAD.t - PAD.b;
  function xAt(i, n) {{ return PAD.l + (i / (n - 1)) * innerW; }}
  function yAt(v) {{ return PAD.t + innerH - ((v - yMin) / (yMax - yMin)) * innerH; }}

  for (var s = 0; s <= 5; s++) {{
    var v = yMin + (s / 5) * (yMax - yMin);
    var gy = yAt(v);
    svg.appendChild(el('line', {{ x1: PAD.l, x2: VB_W - PAD.r, y1: gy, y2: gy, class: 'gridline' }}));
    var lbl = el('text', {{ x: 4, y: gy + 3, class: 'axis-label' }});
    lbl.textContent = Math.round(v);
    svg.appendChild(lbl);
  }}
  var baseY = yAt(100);
  svg.appendChild(el('line', {{ x1: PAD.l, x2: VB_W - PAD.r, y1: baseY, y2: baseY, class: 'gridline', style: 'stroke-dasharray:2 3' }}));

  var endYs = [];
  accounts.forEach(function (a, idx) {{
    var series = indexed[idx];
    var n = series.length;
    var d = series.map(function (v, i) {{ return (i === 0 ? 'M' : 'L') + xAt(i, n).toFixed(2) + ',' + yAt(v).toFixed(2); }}).join(' ');
    var color = isDark ? a.color_dark : a.color_light;
    svg.appendChild(el('path', {{ d: d, style: 'stroke:' + color + ';fill:none;stroke-width:2', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }}));
    var lastV = series[n - 1];
    var ex = xAt(n - 1, n), ey = yAt(lastV);
    svg.appendChild(el('circle', {{ cx: ex, cy: ey, r: 4, style: 'fill:' + color + ';stroke:var(--surface);stroke-width:2' }}));
    endYs.push({{ label: a.label, color: color, ey: ey, v: lastV }});
  }});

  endYs.sort(function (a, b) {{ return a.ey - b.ey; }});
  for (var i = 1; i < endYs.length; i++) {{
    if (endYs[i].ey - endYs[i - 1].ey < 15) endYs[i].ey = endYs[i - 1].ey + 15;
  }}
  endYs.forEach(function (item) {{
    var t = el('text', {{ x: VB_W - PAD.r + 10, y: item.ey + 4, class: 'axis-label', style: 'fill:' + item.color }});
    t.textContent = Math.round(item.v);
    svg.appendChild(t);
  }});
}})();
</script>
</body>
</html>
"""


def main():
    accounts = [load_account(meta) for meta in ACCOUNTS]
    html = render(accounts)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
