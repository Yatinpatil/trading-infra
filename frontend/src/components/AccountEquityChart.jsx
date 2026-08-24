import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { accountColor, usePrefersDark } from "../theme";
import { formatMoney } from "../format";

export default function AccountEquityChart({ account }) {
  const dark = usePrefersDark();

  if (!account.started || account.dates.length < 2) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: "var(--ink-muted)" }}>
        Not enough history to chart yet.
      </div>
    );
  }

  const color = accountColor(account, dark);
  const border = dark ? "#332c22" : "#e3dbc9";
  const muted = dark ? "#8b8271" : "#857c6c";
  const surface = dark ? "#1e1a15" : "#ffffff";
  const data = account.dates.map((date, i) => ({ date, equity: account.values[i] }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <AreaChart data={data} margin={{ top: 8, right: 24, left: 4, bottom: 4 }}>
        <defs>
          <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={border} vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: muted }} tickLine={false} axisLine={{ stroke: border }} />
        <YAxis
          tick={{ fontSize: 11, fill: muted }}
          tickLine={false}
          axisLine={{ stroke: border }}
          domain={["auto", "auto"]}
          width={70}
          tickFormatter={(v) => formatMoney(v)}
        />
        <Tooltip
          contentStyle={{ background: surface, border: `1px solid ${border}`, borderRadius: 8, fontSize: 12.5 }}
          formatter={(value) => [formatMoney(value), "Equity"]}
        />
        <Area type="monotone" dataKey="equity" stroke={color} strokeWidth={2} fill="url(#equityFill)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
