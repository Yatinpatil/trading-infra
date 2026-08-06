"""Performance metrics shared by every strategy report — the point of a
common metrics module is that strategies become comparable on the same
terms.
"""
import numpy as np
import pandas as pd


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1 / years) - 1


def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


_ZERO_VOL_EPSILON = 1e-9


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2 or returns.std() < _ZERO_VOL_EPSILON:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    downside = excess[excess < 0]
    downside_std = downside.std()
    if not downside_std or np.isnan(downside_std) or downside_std < _ZERO_VOL_EPSILON:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    return float(drawdown_series(equity_curve).min())


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def avg_trade_duration(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float(trades["holding_days"].mean())


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252) -> dict:
    returns = periodic_returns(equity_curve)
    return {
        "cagr": cagr(equity_curve, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_trade_duration_days": avg_trade_duration(trades),
        "num_trades": len(trades),
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0,
    }
