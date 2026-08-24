import { useState } from "react";

/** A button that fires an async action (a real paper-trading step),
 * shows a spinner while it's in flight, and a short-lived result message
 * afterward. `onRun` should return whatever summary/result text is worth
 * showing; `onDone` is called after so the caller can refresh its data. */
export default function RunButton({ label, primary, onRun, onDone }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  async function handleClick() {
    setBusy(true);
    setResult(null);
    try {
      const message = await onRun();
      setResult({ ok: true, text: message });
    } catch (err) {
      setResult({ ok: false, text: err.message || String(err) });
    } finally {
      setBusy(false);
      onDone?.();
      setTimeout(() => setResult(null), 8000);
    }
  }

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
      <button className={`btn ${primary ? "btn-primary" : ""}`} onClick={handleClick} disabled={busy}>
        {busy ? <span className="spinner" /> : label}
      </button>
      {result && (
        <span
          className="mono"
          style={{ fontSize: 12.5, color: result.ok ? "var(--good)" : "var(--bad)", maxWidth: 360 }}
        >
          {result.text}
        </span>
      )}
    </span>
  );
}
