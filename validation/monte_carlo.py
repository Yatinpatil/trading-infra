"""Monte Carlo resampling of a trade sequence — how much of the backtest
result is the strategy's edge vs. the luck of how trades happened to order?

Two methods:
  - "shuffle": permute the same trades into a different order (isolates
    order/sequencing luck — drawdown clustering, streaks — from the edge).
  - "bootstrap": resample trades with replacement (isolates how much the
    result depends on the specific sample of trades observed).

Simplifying assumption: each trade's pnl_pct is compounded as if the full
equity were reinvested every time (initial_capital * cumprod(1+returns)).
That's why "shuffle" leaves final_equity unchanged (multiplication commutes)
but max_drawdown varies — it isolates path risk from the return itself. It
will NOT reproduce the real backtest's final equity, since the actual engine
only risks a fraction of capital per trade (position_size_pct) and leaves
cash idle between positions; this module is deliberately measuring something
narrower — sequencing risk in the trade outcomes themselves.
"""
import numpy as np
import pandas as pd


def _equity_curve_from_returns(returns: np.ndarray, initial_capital: float) -> np.ndarray:
    return initial_capital * np.cumprod(1 + returns)


def _max_drawdown(curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(curve)
    return float((curve / running_max - 1).min())


def monte_carlo_simulation(
    trade_returns: pd.Series,
    num_simulations: int = 1000,
    method: str = "shuffle",
    initial_capital: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run `num_simulations` resampled trade sequences; returns one row per
    simulation with the resulting final equity and max drawdown.
    """
    if method not in ("shuffle", "bootstrap"):
        raise ValueError(f"unknown method: {method!r} (expected 'shuffle' or 'bootstrap')")

    rng = np.random.default_rng(seed)
    returns_array = trade_returns.to_numpy()
    n = len(returns_array)

    rows = []
    for _ in range(num_simulations):
        if n == 0:
            sim_returns = returns_array
        elif method == "shuffle":
            sim_returns = rng.permutation(returns_array)
        else:
            sim_returns = rng.choice(returns_array, size=n, replace=True)

        curve = _equity_curve_from_returns(sim_returns, initial_capital)
        final_equity = curve[-1] if len(curve) else initial_capital
        rows.append(
            {
                "final_equity": final_equity,
                "total_return": final_equity / initial_capital - 1,
                "max_drawdown": _max_drawdown(curve) if len(curve) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_monte_carlo(sim_results: pd.DataFrame, percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Percentile summary of a monte_carlo_simulation() result — how wide is
    the range of plausible outcomes, and where does the *actual* backtest
    fall within it.
    """
    return sim_results.describe(percentiles=[p / 100 for p in percentiles])
