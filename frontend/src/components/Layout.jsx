import { NavLink, Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "32px 24px 72px" }}>
      <header style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 28 }}>
        <div>
          <div
            className="mono"
            style={{
              fontSize: 12,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--ink-muted)",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span
              style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", display: "inline-block" }}
            />
            PAPER TRADING &middot; NIFTY 50
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: "8px 0 0" }}>
            <NavLink to="/" style={{ color: "inherit" }}>
              Paper Trading
            </NavLink>
          </h1>
        </div>
        <nav style={{ display: "flex", gap: 18 }}>
          <NavTab to="/">Dashboard</NavTab>
          <NavTab to="/logs">Logs</NavTab>
        </nav>
      </header>
      <Outlet />
    </div>
  );
}

function NavTab({ to, children }) {
  return (
    <NavLink
      to={to}
      end
      style={({ isActive }) => ({
        fontSize: 13.5,
        fontWeight: 600,
        color: isActive ? "var(--ink)" : "var(--ink-muted)",
        borderBottom: isActive ? "2px solid var(--accent)" : "2px solid transparent",
        paddingBottom: 6,
      })}
    >
      {children}
    </NavLink>
  );
}
