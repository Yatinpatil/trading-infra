const BASE = "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  listAccounts: () => request("/accounts"),
  getAccount: (key) => request(`/accounts/${key}`),
  runAccount: (key) => request(`/accounts/${key}/run`, { method: "POST" }),
  runAll: () => request("/run-all", { method: "POST" }),
  getLog: (name, lines = 300) => request(`/logs/${name}?lines=${lines}`),
  getLiveQuotes: (symbols) =>
    symbols.length ? request(`/live-quotes?symbols=${symbols.join(",")}`) : Promise.resolve({}),
  getBacktestResults: () => request("/backtest-results"),
};
