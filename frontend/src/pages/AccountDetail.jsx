import { useCallback, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import AccountEquityChart from "../components/AccountEquityChart";
import RunButton from "../components/RunButton";
import StatusBadge from "../components/StatusBadge";
import { formatMoney, formatNum, formatPct } from "../format";
import { usePolling } from "../hooks";

const PAGE_SIZE = 20;

export default function AccountDetail() {
  const { key } = useParams();
  const fetcher = useCallback(() => api.getAccount(key), [key]);
  const { data: account, error, loading, refetch } = usePolling(fetcher, 30_000);
  const [page, setPage] = useState(0);

  const liveSymbols = useMemo(() => {
    if (!account) return [];
    return [...new Set([...account.positions.map((p) => p.symbol), ...account.pending_entries])].sort();
  }, [account]);
  const quotesFetcher = useCallback(() => api.getLiveQuotes(liveSymbols), [liveSymbols]);
  const { data: liveQuotes } = usePolling(quotesFetcher, 60_000);

  const pageTrades = useMemo(() => {
    if (!account) return [];
    return account.trades.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  }, [account, page]);

  if (error) return <p style={{ color: "var(--bad)" }}>Couldn't reach the API: {error}</p>;
  if (loading && !account) return <p className="mono" style={{ color: "var(--ink-muted)" }}>Loading…</p>;
  if (!account) return null;

  const m = account.metrics;
  const pageCount = Math.max(1, Math.ceil(account.trades.length / PAGE_SIZE));

  return (
    <div>
      <Link to="/" className="mono" style={{ fontSize: 12.5, color: "var(--ink-muted)" }}>
        &larr; all accounts
      </Link>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "10px 0 20px" }}>
        <h2 style={{ fontSize: 22, margin: 0, display: "flex", alignItems: "center" }}>
          <span className="swatch" style={{ width: 14, height: 14, background: account.color_light }} />
          {account.label}
        </h2>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <StatusBadge account={account} />
          <button className="btn" onClick={refetch}>
            Refresh
          </button>
          <RunButton
            label="Run Now"
            primary
            onRun={async () => {
              const result = await api.runAccount(key);
              if (result.summary?.skipped) return `Skipped: ${result.summary.reason}`;
              return `Equity ${formatMoney(result.summary.equity)}, ${result.summary.trades_today.length} fill(s) today.`;
            }}
            onDone={refetch}
          />
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 24 }}>
        <Stat label="Equity" value={account.started ? formatMoney(account.equity) : "—"} />
        <Stat label="Cash" value={account.started ? formatMoney(account.cash) : "—"} />
        <Stat label="Today" value={formatPct(account.today_change_pct)} />
        <Stat label="CAGR" value={m ? formatPct(m.cagr) : "—"} />
        <Stat label="Sharpe" value={m ? formatNum(m.sharpe) : "—"} />
        <Stat label="Max DD" value={m ? formatPct(m.max_drawdown, 1) : "—"} />
      </div>

      <div className="card" style={{ padding: "16px 18px 8px", marginBottom: 28 }}>
        <AccountEquityChart account={account} />
      </div>

      <SectionHead title={`Open positions (${account.positions.length})`} />
      {account.positions.length > 0 && (
        <p className="mono" style={{ fontSize: 11.5, color: "var(--ink-muted)", margin: "-6px 0 10px" }}>
          Live price is a delayed quote (Yahoo Finance, ~15-20 min) shown for reference only — it never affects fills,
          which always use NSE's own end-of-day data.
        </p>
      )}
      <div className="card" style={{ overflowX: "auto", marginBottom: 28 }}>
        {account.positions.length ? (
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th className="num">Qty</th>
                <th className="num">Entry</th>
                <th>Entry date</th>
                <th className="num">Stop</th>
                <th className="num">Live</th>
                <th className="num">Unrealized %</th>
              </tr>
            </thead>
            <tbody>
              {account.positions.map((p) => {
                const q = liveQuotes?.[p.symbol];
                const unrealizedPct = q ? ((q.price - p.entry_price) / p.entry_price) * 100 : null;
                return (
                  <tr key={p.symbol}>
                    <td>{p.symbol}</td>
                    <td className="num">{p.quantity}</td>
                    <td className="num">{formatMoney(p.entry_price)}</td>
                    <td>{p.entry_date}</td>
                    <td className="num">{p.stop_price ? formatMoney(p.stop_price) : "—"}</td>
                    <td className="num">{q ? formatMoney(q.price) : "—"}</td>
                    <td className={`num ${unrealizedPct == null ? "" : unrealizedPct >= 0 ? "pnl-pos" : "pnl-neg"}`}>
                      {unrealizedPct == null ? "—" : formatPct(unrealizedPct)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <p style={{ padding: 16, color: "var(--ink-muted)", margin: 0 }}>No open positions.</p>
        )}
      </div>

      {(account.pending_entries.length > 0 || account.pending_exits.length > 0) && (
        <p className="mono" style={{ fontSize: 12.5, color: "var(--ink-muted)", marginTop: -18, marginBottom: 28 }}>
          Queued for next open — entries: {account.pending_entries.join(", ") || "none"}, exits:{" "}
          {account.pending_exits.join(", ") || "none"}
        </p>
      )}

      <SectionHead title={`Trade history (${account.trades.length})`} />
      <div className="card" style={{ overflowX: "auto" }}>
        {account.trades.length ? (
          <>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th className="num">Qty</th>
                  <th className="num">P&amp;L</th>
                  <th className="num">P&amp;L %</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {pageTrades.map((t, i) => (
                  <tr key={`${t.symbol}-${t.exit_date}-${i}`}>
                    <td>{t.symbol}</td>
                    <td>{t.entry_date}</td>
                    <td>{t.exit_date}</td>
                    <td className="num">{t.quantity}</td>
                    <td className={`num ${t.pnl >= 0 ? "pnl-pos" : "pnl-neg"}`}>{formatMoney(t.pnl)}</td>
                    <td className={`num ${t.pnl >= 0 ? "pnl-pos" : "pnl-neg"}`}>{formatPct(t.pnl_pct)}</td>
                    <td className="mono" style={{ fontSize: 12 }}>
                      {t.exit_reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {pageCount > 1 && (
              <div style={{ display: "flex", justifyContent: "center", gap: 12, padding: 14, alignItems: "center" }}>
                <button className="btn" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                  Prev
                </button>
                <span className="mono" style={{ fontSize: 12.5, color: "var(--ink-muted)" }}>
                  page {page + 1} / {pageCount}
                </span>
                <button className="btn" disabled={page >= pageCount - 1} onClick={() => setPage((p) => p + 1)}>
                  Next
                </button>
              </div>
            )}
          </>
        ) : (
          <p style={{ padding: 16, color: "var(--ink-muted)", margin: 0 }}>No closed trades yet.</p>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="card" style={{ padding: "12px 16px" }}>
      <div className="mono" style={{ fontSize: 11, color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 600, marginTop: 4 }} className="mono">
        {value}
      </div>
    </div>
  );
}

function SectionHead({ title }) {
  return (
    <div style={{ margin: "0 0 12px" }}>
      <h3 style={{ fontSize: 15, margin: 0 }}>{title}</h3>
    </div>
  );
}
