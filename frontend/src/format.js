export function formatMoney(value) {
  if (value === null || value === undefined) return "—";
  return "₹" + Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

export function formatPct(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  const pct = value * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(digits)}%`;
}

export function formatNum(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
}
