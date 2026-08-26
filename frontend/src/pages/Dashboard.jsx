import { useCallback } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import EquityComparisonChart from "../components/EquityComparisonChart";
import RunButton from "../components/RunButton";
import StatusBadge from "../components/StatusBadge";
import { formatMoney, formatNum, formatPct } from "../format";
import { usePolling } from "../hooks";

export default function Dashboard() {
  const fetcher = useCallback(() => api.listAccounts(), []);
  const { data: accounts, error, loading, refetch } = usePolling(fetcher, 30_000);

  const backtestFetcher = useCallback(() => api.getBacktestResults(), []);
  // Backtest results only change when someone re-runs the comparison script
  // by hand, not every trading day -- a slow poll is plenty.
  const { data: backtestResults } = usePolling(backtestFetcher, 5 * 60_000);

  async function handleRunAll() {
    const results = await api.runAll();
    const filled = results.reduce((n, r) => n + (r.summary?.trades_today?.length || 0), 0);
    const skipped = results.filter((r) => r.summary?.skipped).length;
    return skipped === results.length
      ? "All accounts: no trading data yet for today."
      : `Ran ${results.length} accounts, ${filled} fills.`;
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
        <p style={{ color: "var(--ink-2)", margin: 0 }}>
          {accounts ? accounts.length : "—"} strategies, one ₹10L account each, refreshed after every trading day's close.
        </p>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button className="btn" onClick={refetch}>
            Refresh
          </button>
          <RunButton label="Run All" primary onRun={handleRunAll} onDone={refetch} />
        </div>
      </div>

      {error && <ErrorBanner message={error} />}
      {loading && !accounts && <p className="mono" style={{ color: "var(--ink-muted)" }}>Loading…</p>}

      {accounts && (
        <>
          <SectionHead title="Accounts" />
          <div className="card" style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th className="num">Equity</th>
                  <th className="num">Today</th>
                  <th className="num">Open</th>
                  <th className="num">Trades</th>
                  <th>Last run</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((a) => (
                  <tr key={a.key}>
                    <td>
                      <Link to={`/accounts/${a.key}`} style={{ display: "flex", alignItems: "center", fontWeight: 500 }}>
                        <span className="swatch" style={{ background: a.color_light }} />
                        {a.label}
                      </Link>
                    </td>
                    <td className="num">{a.started ? formatMoney(a.equity) : "—"}</td>
                    <td className="num">{a.today_change_pct !== null ? formatPct(a.today_change_pct) : "—"}</td>
                    <td className="num">{a.started ? a.num_open_positions : "—"}</td>
                    <td className="num">{a.started ? a.num_trades : "—"}</td>
                    <td className="mono" style={{ fontSize: 12.5 }}>
                      {a.last_run_date || "—"}
                    </td>
                    <td>
                      <StatusBadge account={a} />
                    </td>
                    <td>
                      <RunButton label="Run" onRun={() => runOne(a.key)} onDone={refetch} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {backtestResults && (
            <>
              <SectionHead title="Backtest performance" hint={backtestHint(backtestResults)} />
              <div className="card" style={{ overflowX: "auto" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th className="num">CAGR</th>
                      <th className="num">Sharpe</th>
                      <th className="num">Max DD</th>
                      <th className="num">Win rate</th>
                      <th className="num">Trades</th>
                    </tr>
                  </thead>
                  <tbody>
                    {accounts.map((a) => {
                      const bt = backtestResults[a.key];
                      return (
                        <tr key={a.key}>
                          <td style={{ display: "flex", alignItems: "center", fontWeight: 500 }}>
                            <span className="swatch" style={{ background: a.color_light }} />
                            {a.label}
                          </td>
                          {bt ? (
                            <>
                              <td className="num">{formatPct(bt.cagr)}</td>
                              <td className="num">{formatNum(bt.sharpe)}</td>
                              <td className="num">{formatPct(bt.max_drawdown, 1)}</td>
                              <td className="num">{formatPct(bt.win_rate, 0)}</td>
                              <td className="num">{bt.num_trades.toLocaleString("en-IN")}</td>
                            </>
                          ) : (
                            <td className="num" colSpan={5} style={{ color: "var(--ink-muted)", fontSize: 12.5 }}>
                              not covered by this comparison
                            </td>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="mono" style={{ fontSize: 11.5, color: "var(--ink-muted)", margin: "8px 0 0" }}>
                A multi-year historical simulation, not the live ledger above — a strategy's live account may be only
                days old.
              </p>
            </>
          )}

          <SectionHead title="Equity curves" hint="indexed to 100 at each account's start" />
          <div className="card" style={{ padding: "16px 18px 8px" }}>
            <EquityComparisonChart accounts={accounts} />
          </div>
        </>
      )}
    </div>
  );
}

function backtestHint(backtestResults) {
  const any = Object.values(backtestResults)[0];
  return any ? `${any.start_date} to ${any.end_date}, net of costs` : "not yet run";
}

async function runOne(key) {
  const result = await api.runAccount(key);
  if (result.summary?.skipped) return `Skipped: ${result.summary.reason}`;
  const trades = result.summary?.trades_today?.length || 0;
  return `Equity ₹${Math.round(result.summary.equity).toLocaleString("en-IN")}, ${trades} fill(s) today.`;
}

function SectionHead({ title, hint }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", margin: "36px 0 14px" }}>
      <h2 style={{ fontSize: 18, margin: 0 }}>{title}</h2>
      {hint && <span className="mono" style={{ fontSize: 12, color: "var(--ink-muted)" }}>{hint}</span>}
    </div>
  );
}

function ErrorBanner({ message }) {
  return (
    <div
      className="card"
      style={{ padding: "12px 16px", marginBottom: 16, borderColor: "var(--bad)", color: "var(--bad)" }}
    >
      Couldn't reach the API: {message}
    </div>
  );
}
