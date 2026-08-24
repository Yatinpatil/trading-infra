import { useCallback, useState } from "react";
import { api } from "../api";
import { usePolling } from "../hooks";

const FILES = [
  { name: "orchestrator.log", label: "Orchestrator (daily run summary)" },
  { name: "paper_trading.log", label: "Paper trading (per-account detail)" },
];

export default function Logs() {
  const [file, setFile] = useState(FILES[0].name);
  const fetcher = useCallback(() => api.getLog(file), [file]);
  const { data, error, loading, refetch } = usePolling(fetcher, 20_000);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 10 }}>
          {FILES.map((f) => (
            <button
              key={f.name}
              className="btn"
              style={file === f.name ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
              onClick={() => setFile(f.name)}
            >
              {f.label}
            </button>
          ))}
        </div>
        <button className="btn" onClick={refetch}>
          Refresh
        </button>
      </div>

      {error && <p style={{ color: "var(--bad)" }}>Couldn't reach the API: {error}</p>}
      {loading && !data && <p className="mono" style={{ color: "var(--ink-muted)" }}>Loading…</p>}

      {data && (
        <div
          className="card mono"
          style={{
            padding: "14px 18px",
            fontSize: 12.5,
            maxHeight: 560,
            overflowY: "auto",
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {data.lines.length ? (
            data.lines.map((line, i) => (
              <div key={i} style={{ color: line.includes("ERROR") ? "var(--bad)" : line.includes("WARNING") ? "var(--warn)" : "inherit" }}>
                {line}
              </div>
            ))
          ) : (
            <span style={{ color: "var(--ink-muted)" }}>No log entries yet.</span>
          )}
        </div>
      )}
    </div>
  );
}
