export default function StatusBadge({ account }) {
  if (!account.started) {
    return <span className="badge badge-muted">not started</span>;
  }
  if (account.status === "current") {
    return <span className="badge badge-good">current</span>;
  }
  return <span className="badge badge-warn">stale ({account.age_days}d)</span>;
}
