# trading-infra

A backtesting, validation, and paper-trading framework for NSE (Indian equity) strategies, built layer by layer: data → indicators → strategies → execution engine → risk → validation → reporting → live paper trading. Every layer is independently testable and none of them trust the others' output without checking it — the project's running theme is "an honest negative result is not a bug," and several real numbers in here are negative or unflattering on purpose.

## Why this exists

Most weekend backtests fool their author before they fool anyone else: look-ahead bias, un-costed trades, survivorship-biased universes, and — the two bugs this codebase actually shipped and then fixed — silently corrupted price data. This project is built so those mistakes are structurally hard to make: strategies never see prices after the bar they're deciding on, every trade is costed, and every reported backtest number has a walk-forward or out-of-sample check standing behind it before it's trusted.

## Architecture

```
data/          OHLCV + corporate-action fetch (jugaad-data/NSE), parquet cache, retry/TTL, quality checks
indicators/    Causal technical indicators (zscore, RSI, MACD, ADX, ATR, Bollinger, Donchian, ROC)
strategies/    Strategy interface + mean_reversion, momentum, breakout, buy_and_hold, ml_strategy
engine/        Single-stock and portfolio backtest engines, transaction cost model
risk/          Position sizing, concurrency/sector/correlation limits
validation/    Train/test splits, walk-forward validation, grid search, Monte Carlo, benchmark comparison
analytics/     Performance metrics, HTML tear sheets, strategy comparison tables
universe/      Point-in-time index constituents, liquidity filtering
execution/     Paper-trading broker + engine (persisted ledger, next-bar-open timing)
ml/            Feature engineering + labeling for the ML strategy
configs/       YAML strategy configs (universe, costs, risk, strategy params)
scripts/       Data backfill, strategy comparison, ML walk-forward evaluation, hyperparameter grid search
```

A `Strategy` only ever sees OHLCV and returns `entry_long`/`exit_long`/`entry_short`/`exit_short` signals — it never sees costs, sizing, or fills. That separation is what lets the same strategy code run in a single-stock backtest, a multi-symbol portfolio backtest, and daily paper trading with zero changes.

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

## Testing

```
pytest              # offline suite — no network required
pytest -m network   # also exercises real NSE endpoints
```

164 tests, all passing, in the default (offline) suite.

## What's actually been found running this

- **Buy-and-hold beat every active strategy** on a 48-stock NIFTY 50 backtest, 2021–2026, net of costs. Mean reversion had the best win rate and smallest drawdown but its edge didn't clear 1,140 round-trips of brokerage, STT, and slippage.
- **A gradient-boosted classifier overfit badly** with default hyperparameters — ~20-28% CAGR in-sample every walk-forward window, but out-of-sample Sharpe of only 0.26 and wildly unstable per-window returns. Regularizing hard (shallow trees, large leaf minimums, strong L2) raised out-of-sample Sharpe to 0.87, confirmed as a robust region rather than a lucky pick via a 12-point grid search.
- **Two real data bugs were found and fixed** while backtesting on freshly-downloaded data: `jugaad-data`'s equity-series filter is silently ignored by NSE's API, splicing bond/NCD prices into equity series; and the corporate-action parser missed the "Re 1/-" wording NSE uses for most real stock splits. Outlier days across the NIFTY 50 universe went from 440+ to 8, the remainder being real market events (not bugs).

## Design notes worth knowing before extending this

- **No look-ahead, anywhere.** Indicators are causal by construction; the backtest engines fill signals at the *next* bar's open, never the signal bar's own close; a stop-loss checks the *current* bar's low, which is a resting order, not a forecast.
- **Costs are always on.** There is no "gross" backtest result — brokerage, STT, and slippage apply to every leg.
- **Adjustment is backward and recomputed on every call**, so a paper-trading position's execution (fills, stops, mark-to-market) uses *raw* prices, never the adjusted series — adjustment can retroactively shift a historical date's value the moment a new corporate action is registered, which is fine for a one-shot backtest but wrong for state held open across many days.
- **A model needs a training window before it needs a threshold.** `ml_strategy` is the one strategy that carries fitted state; it's evaluated with embargoed, walk-forward train/test splits, not a single split, because a single split is easy to accidentally cherry-pick.
