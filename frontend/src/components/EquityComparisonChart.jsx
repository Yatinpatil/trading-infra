import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { accountColor, usePrefersDark } from "../theme";

function mergeIndexedSeries(accounts) {
  const dateSet = new Set();
  accounts.forEach((a) => a.dates.forEach((d) => dateSet.add(d)));
  const dates = Array.from(dateSet).sort();

  const byAccount = {};
  accounts.forEach((a) => {
    const base = a.values[0];
    const map = {};
    a.dates.forEach((d, i) => {
      map[d] = base ? (a.values[i] / base) * 100 : null;
    });
    byAccount[a.key] = map;
  });

  return dates.map((date) => {
    const point = { date };
    accounts.forEach((a) => {
      point[a.key] = byAccount[a.key][date] ?? null;
    });
    return point;
  });
}

export default function EquityComparisonChart({ accounts }) {
  const dark = usePrefersDark();
  const withHistory = accounts.filter((a) => a.started && a.dates.length > 1);

  if (!withHistory.length) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: "var(--ink-muted)" }}>
        No account has enough history yet to chart.
      </div>
    );
  }

  const data = mergeIndexedSeries(withHistory);
  const border = dark ? "#332c22" : "#e3dbc9";
  const muted = dark ? "#8b8271" : "#857c6c";
  const surface = dark ? "#1e1a15" : "#ffffff";

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 24, left: 4, bottom: 4 }}>
        <CartesianGrid stroke={border} vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: muted }} tickLine={false} axisLine={{ stroke: border }} />
        <YAxis
          tick={{ fontSize: 11, fill: muted }}
          tickLine={false}
          axisLine={{ stroke: border }}
          domain={["auto", "auto"]}
          width={40}
        />
        <Tooltip
          contentStyle={{ background: surface, border: `1px solid ${border}`, borderRadius: 8, fontSize: 12.5 }}
          formatter={(value, name) => [value?.toFixed(1), withHistory.find((a) => a.key === name)?.label ?? name]}
        />
        <Legend
          formatter={(value) => withHistory.find((a) => a.key === value)?.label ?? value}
          wrapperStyle={{ fontSize: 12.5 }}
        />
        {withHistory.map((a) => (
          <Line
            key={a.key}
            type="monotone"
            dataKey={a.key}
            stroke={accountColor(a, dark)}
            strokeWidth={2}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
