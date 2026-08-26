# trading-infra

A backtesting, validation, and paper-trading framework for NSE (Indian equity) strategies, built layer by layer: data → indicators → strategies → execution engine → risk → validation → reporting → live paper trading. Every layer is independently testable and none of them trust the others' output without checking it — the project's running theme is "an honest negative result is not a bug," and several real numbers in here are negative or unflattering on purpose.

## Why this exists

Most weekend backtests fool their author before they fool anyone else: look-ahead bias, un-costed trades, survivorship-biased universes, and — the two bugs this codebase actually shipped and then fixed — silently corrupted price data. This project is built so those mistakes are structurally hard to make: strategies never see prices after the bar they're deciding on, every trade is costed, and every reported backtest number has a walk-forward or out-of-sample check standing behind it before it's trusted.

## Architecture

```
db/            The single SQLite store (data/trading.db): OHLCV, corporate actions, configs,
               and every paper-trading account's positions/trades/equity/fitted model
data/          OHLCV + corporate-action fetch (jugaad-data/NSE) into db/, retry/TTL, quality checks
indicators/    Causal technical indicators (zscore, RSI, MACD, ADX, ATR, Bollinger, Donchian, ROC)
strategies/    Strategy interface + mean_reversion, momentum, breakout, buy_and_hold, ml_strategy,
               rsi_mean_reversion, bollinger_breakout, adx_trend, macd_crossover, atr_channel_breakout
engine/        Single-stock and portfolio backtest engines, transaction cost model
risk/          Position sizing, concurrency/sector/correlation limits
validation/    Train/test splits, walk-forward validation, grid search, Monte Carlo, benchmark comparison
analytics/     Performance metrics, HTML tear sheets, strategy comparison tables
universe/      Point-in-time index constituents, liquidity filtering
execution/     Paper-trading broker (db/-backed) + engine (next-bar-open timing)
ml/            Feature engineering + labeling for the ML strategy
configs/       YAML strategy configs (universe, costs, risk, strategy params) -- seeded into db/ on first load
scripts/       Data backfill, strategy comparison, ML walk-forward evaluation, hyperparameter grid
               search, the daily paper-trading orchestrator, and the dashboard generator
dashboard/     Generated (gitignored): dashboard/index.html, a static, self-contained page
               summarizing every paper-trading account -- open it directly in a browser
api/           FastAPI backend for the interactive web UI -- live account state, "Run Now" /
               "Run All" triggers, log tailing; serves frontend/dist/ once it's built
frontend/      React (Vite) app for the same data, interactively -- see "Web UI" below
```

A `Strategy` only ever sees OHLCV and returns `entry_long`/`exit_long`/`entry_short`/`exit_short` signals — it never sees costs, sizing, or fills. That separation is what lets the same strategy code run in a single-stock backtest, a multi-symbol portfolio backtest, and daily paper trading with zero changes.

Every persisted table lives in one SQLite file (`data/trading.db`, gitignored) rather than the parquet/JSON/CSV/YAML mix earlier phases used — see `db/schema.py` for the full table list. `scripts/migrate_to_sqlite.py` is the one-time, additive migration that populated it from that earlier layout; it's safe to re-run (every write is an upsert) and never touches the original files.

## Setup

```
python -m venv .venv          # Python 3.12 — 3.14 lacks prebuilt wheels for pyarrow/numpy on Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

**Run a backtest:**
```
python main.py --config mean_reversion --symbol RELIANCE --start 2020-01-01 --end 2024-01-01
python main.py --config mean_reversion --start 2020-01-01 --end 2024-01-01   # portfolio mode, universe from config
```

**Backfill historical data for a universe:**
```
python scripts/backfill_universe.py --index "NIFTY 50" --start 2021-01-01
```

**Compare all strategies on a universe:**
```
python scripts/compare_nifty50_strategies.py --start 2021-01-01 --end 2026-08-05
```

**Walk-forward evaluate the ML strategy against the rule-based ones:**
```
python scripts/evaluate_ml_strategy.py
python scripts/grid_search_ml_strategy.py   # check a hyperparameter region isn't a lucky pick
```

**Run paper trading** (invoke once per trading day, e.g. via cron/Task Scheduler, after NSE close):
```
python run_paper_trading.py --config mean_reversion --symbol RELIANCE
python run_paper_trading.py --config ml_strategy_nifty50              # portfolio mode; refits on a schedule
python run_paper_trading.py --config mean_reversion --report          # tear sheet from the ledger so far
```

**Run every configured account for the day, then rebuild the dashboard** (this is what the scheduled job calls):
```
python scripts/run_daily_paper_trading.py
```
Then open `dashboard/index.html` directly in a browser — no server required.

## Automated daily paper trading

A Windows Task Scheduler job (`NIFTY50_PaperTrading_Daily`) runs `scripts/poll_and_run_paper_trading.py` every 15 minutes from 4:00-6:45 PM IST on weekdays — after NSE's 3:30 PM close, but before NSE's own EOD data is guaranteed to be published. Each invocation is a cheap check (a single-symbol, uncached fetch) for whether today's data is actually out yet; the moment it is, that invocation runs the real step immediately rather than waiting for a fixed later time, and every other invocation that day is then a no-op (`accounts.last_run_date` already matches today). If data still isn't out by 6 PM, the last few polls run the step anyway rather than silently skip the day — the same safety margin the old fixed 6:30 PM trigger gave, just no longer the common case.

The step itself (`scripts/run_daily_paper_trading.py`) steps ten accounts (mean_reversion, momentum, breakout, buy_and_hold, ml_strategy, rsi_mean_reversion, bollinger_breakout, adx_trend, macd_crossover, atr_channel_breakout, each its own ₹10L NIFTY 50 portfolio) and rebuilds the dashboard. Each account runs in its own subprocess, so one failing account never blocks the rest. Logs: `logs/orchestrator.log` (the day's run summary) and `logs/paper_trading.log` (account-level fill/warning detail).

## Web UI

A FastAPI backend (`api/`) exposes the same account state the dashboard generator reads, plus a "Run Now" / "Run All" trigger; a React app (`frontend/`) polls it for a live, navigable view (per-account detail pages, equity comparison chart, log tailing) instead of the static dashboard's fixed snapshot.

Both the static dashboard and the web UI also show each strategy's latest full-period backtest (CAGR, Sharpe, max drawdown, win rate, trades) next to its live paper-trading numbers — kept as a visibly separate section, since a live account may only be days old while the backtest spans years. That table is populated by `scripts/compare_nifty50_strategies.py` (which persists into the `backtest_results` table on every run, not just prints to console) and is a snapshot from whenever that script was last run, not recomputed on page load. `ml_strategy` has no row there since it's evaluated separately via walk-forward validation (`scripts/evaluate_ml_strategy.py`), a different methodology than the single-pass comparison the others share.

**Running the dashboard so it survives closing your editor**: register it as a Windows Scheduled Task that starts at login and auto-restarts if it crashes, rather than running it in a terminal tied to your IDE session:
```
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c ""<repo>\.venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 >> "<repo>\logs\api_server.log" 2>&1"' -WorkingDirectory '<repo>'
$trigger = New-ScheduledTaskTrigger -AtLogOn -User '<your-username>'
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew -StartWhenAvailable
Register-ScheduledTask -TaskName 'NIFTY50_DashboardServer' -Action $action -Trigger $trigger -Settings $settings -Force
Start-ScheduledTask -TaskName 'NIFTY50_DashboardServer'
```
This starts at your next Windows logon (and immediately, via the last line above) and keeps running independent of any terminal or IDE — but it still runs under your own login, not before it; making it start even before anyone logs in requires registering it with a SYSTEM principal from an elevated (Administrator) PowerShell session.

**Day to day, one process serves both** — build the frontend once (or after changing it), then run only the backend:
```
cd frontend && npm install && npm run build && cd ..
python -m uvicorn api.main:app
```
Open `http://127.0.0.1:8000`. Triggering a run from the UI calls the same `run_paper_trading.py` path the scheduled job uses, so it's covered by the same per-account lock file and `last_run_date` guard — running it twice in a row is a no-op the second time, not a double-fill.

**Frontend dev mode** (hot reload, proxies `/api` to the backend):
```
python -m uvicorn api.main:app          # terminal 1
cd frontend && npm install && npm run dev   # terminal 2 — http://127.0.0.1:5173
```

## Testing

```
pytest              # offline suite — no network required
pytest -m network   # also exercises real NSE endpoints
```

217 tests, all passing, in the default (offline) suite.

## What's actually been found running this

- **Buy-and-hold beat every active strategy** on a 48-stock NIFTY 50 backtest, 2021–2026, net of costs (CAGR 21.6%, Sharpe 1.22). Every rule-based active strategy trailed it, ranked by CAGR: MACD crossover (6.1%, Sharpe 0.62 — the best-performing active strategy), momentum (5.8%, Sharpe 0.58), RSI mean-reversion (3.6%, Sharpe 0.62 — tied for best Sharpe and by far the smallest drawdown of any active strategy, -9.3% vs. -13% to -25% for the rest), mean reversion (3.0%, Sharpe 0.44), breakout (2.4%, Sharpe 0.30), ATR channel breakout (1.2%, Sharpe 0.19), and Bollinger breakout (1.0%, Sharpe 0.16). **ADX trend came back flat-to-negative** (CAGR -0.2%, Sharpe 0.02) — a strategy that's genuinely indistinguishable from noise net of costs, kept live anyway for the same reason the others are: an honest negative result is the point, not a bug to hide.
- **A gradient-boosted classifier overfit badly** with default hyperparameters — ~20-28% CAGR in-sample every walk-forward window, but out-of-sample Sharpe of only 0.26 and wildly unstable per-window returns. Regularizing hard (shallow trees, large leaf minimums, strong L2) raised out-of-sample Sharpe to 0.87, confirmed as a robust region rather than a lucky pick via a 12-point grid search.
- **Two real data bugs were found and fixed** while backtesting on freshly-downloaded data: `jugaad-data`'s equity-series filter is silently ignored by NSE's API, splicing bond/NCD prices into equity series; and the corporate-action parser missed the "Re 1/-" wording NSE uses for most real stock splits. Outlier days across the NIFTY 50 universe went from 440+ to 8, the remainder being real market events (not bugs). A third, live-only bug: NSE's history endpoint intermittently drops the newest day depending on the exact date range requested, and *which* range gets affected shifts from day to day — a single fixed padding wasn't enough to fully dodge it, so the fetch now retries with several genuinely different range shapes until one actually includes the target date. On days NSE never returns it at all, `data/yahoo_fallback.py` supplies that one missing bar from Yahoo Finance as a last resort — sanity-checked against the last known real NSE close (rejecting anything implying a >20% same-day move, since Yahoo's OHLC comes back split-adjusted unless fetched with `auto_adjust=False`) so a bad fallback bar degrades to "no data today" rather than silently corrupting a fill.

## Design notes worth knowing before extending this

- **No look-ahead, anywhere.** Indicators are causal by construction; the backtest engines fill signals at the *next* bar's open, never the signal bar's own close; a stop-loss checks the *current* bar's low, which is a resting order, not a forecast.
- **Costs are always on.** There is no "gross" backtest result — brokerage, STT, and slippage apply to every leg.
- **Adjustment is backward and recomputed on every call**, so a paper-trading position's execution (fills, stops, mark-to-market) uses *raw* prices, never the adjusted series — adjustment can retroactively shift a historical date's value the moment a new corporate action is registered, which is fine for a one-shot backtest but wrong for state held open across many days.
- **A model needs a training window before it needs a threshold.** `ml_strategy` is the one strategy that carries fitted state; it's evaluated with embargoed, walk-forward train/test splits, not a single split, because a single split is easy to accidentally cherry-pick.
- **A paper account's cash/positions/pending-orders only persist on an explicit `save()`**, but closed trades and daily equity marks write immediately regardless — same contract the old JSON+CSV layout had, just backed by SQLite tables now. The per-account lock file is deliberately still a plain filesystem file, not a DB row: it's advisory process coordination (stopping two overlapping invocations of the CLI from racing), a different concern from the data SQLite already protects.
