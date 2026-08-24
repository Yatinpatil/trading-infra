"""CLI entry point: run a configured strategy as a backtest and write an HTML
tear sheet — see analytics/report.py's `generate_report`, the "one command"
result of every other layer (config, data, strategy, engine, risk, reporting).

Single-stock:
    python main.py --config mean_reversion --symbol RELIANCE --start 2020-01-01 --end 2024-01-01

Portfolio (universe/liquidity/risk settings all come from the config file):
    python main.py --config mean_reversion --start 2020-01-01 --end 2024-01-01
"""
import argparse
from datetime import date, datetime, timedelta

from analytics.report import generate_report
from configs import load_config
from data.loaders import get_ohlcv
from engine.portfolio import PortfolioBacktestResult, run_portfolio_backtest
from engine.single_stock import BacktestResult, run_single_backtest
from strategies import get_strategy_class
from universe.constituents import get_universe
from universe.liquidity_filter import filter_by_liquidity


def _as_date(d) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def build_strategy(config: dict):
    strategy_cls = get_strategy_class(config["strategy"])
    return strategy_cls(config.get("params"))


def resolve_portfolio_symbols(config: dict, as_of, lookback_days: int = 60) -> list[str]:
    """Universe constituents as of `as_of`, narrowed by the configured
    liquidity filter using only data through `as_of` — never later dates,
    which would leak future liquidity into an eligibility decision made at
    the start of the backtest (see universe/liquidity_filter.py's own
    "already sliced to end at the as-of date" contract). Shared by
    run_portfolio (backtests) and run_paper_trading.py (paper trading) so
    both agree on which symbols are eligible.
    """
    universe_cfg = config.get("universe", {})
    symbols = get_universe(universe_cfg["index"], as_of_date=as_of)

    min_avg_daily_value = universe_cfg.get("min_avg_daily_value")
    if not min_avg_daily_value:
        return symbols

    as_of_date = _as_date(as_of)
    lookback_start = as_of_date - timedelta(days=lookback_days * 2)  # calendar-day buffer for weekends/holidays
    ohlcv_by_symbol = {}
    for symbol in symbols:
        df = get_ohlcv(symbol, lookback_start, as_of_date)
        if not df.empty:
            ohlcv_by_symbol[symbol] = df

    return filter_by_liquidity(ohlcv_by_symbol, min_avg_daily_value, lookback_days)


def run_single(config: dict, symbol: str, start, end, initial_capital: float) -> BacktestResult:
    df = get_ohlcv(symbol, start, end)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol} between {start} and {end}")
    strategy = build_strategy(config)
    return run_single_backtest(strategy, df, config, initial_capital)


def run_portfolio(config: dict, start, end, initial_capital: float) -> PortfolioBacktestResult:
    symbols = resolve_portfolio_symbols(config, start)

    ohlcv_by_symbol = {}
    for symbol in symbols:
        df = get_ohlcv(symbol, start, end)
        if not df.empty:
            ohlcv_by_symbol[symbol] = df

    if not ohlcv_by_symbol:
        index_name = config.get("universe", {}).get("index")
        raise ValueError(f"No symbols in universe '{index_name}' passed the liquidity filter")

    strategy = build_strategy(config)
    return run_portfolio_backtest(strategy, ohlcv_by_symbol, config, initial_capital)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured backtest and write a tear sheet.")
    parser.add_argument("--config", default="mean_reversion", help="Config name under configs/ (without .yaml)")
    parser.add_argument("--symbol", help="Run a single-stock backtest on this symbol instead of the configured universe")
    parser.add_argument("--start", required=True, help="Backtest start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Backtest end date, YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Initial capital")
    parser.add_argument("--output-dir", default="reports", help="Directory to write the tear sheet into")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)

    if args.symbol:
        result = run_single(config, args.symbol, args.start, args.end, args.capital)
        report_name = f"{config['strategy']}_{args.symbol}"
    else:
        result = run_portfolio(config, args.start, args.end, args.capital)
        report_name = f"{config['strategy']}_portfolio"

    report_path = generate_report(report_name, result, args.output_dir)

    print(f"Final equity: {result.final_equity:,.2f}")
    print(f"Trades: {len(result.trades)}")
    print(f"Report written to {report_path}")
    return result


if __name__ == "__main__":
    main()
