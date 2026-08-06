import numpy as np
import pandas as pd
import pytest

from validation.monte_carlo import monte_carlo_simulation, summarize_monte_carlo


def test_shuffle_preserves_final_equity_since_multiplication_commutes():
    # compounding the same set of returns in any order gives the same product
    returns = pd.Series([0.05, -0.03, 0.02, -0.01, 0.04])
    expected_final = float(np.prod(1 + returns.to_numpy()))

    results = monte_carlo_simulation(returns, num_simulations=200, method="shuffle", initial_capital=1.0, seed=1)

    assert results["final_equity"].to_numpy() == pytest.approx(expected_final, rel=1e-9)


def test_shuffle_produces_varying_drawdowns_despite_fixed_final_equity():
    returns = pd.Series([0.10, -0.08, 0.10, -0.08, 0.10, -0.08])
    results = monte_carlo_simulation(returns, num_simulations=200, method="shuffle", initial_capital=1.0, seed=2)

    # order changes drawdown even though the final compounded value can't move
    assert results["max_drawdown"].nunique() > 1


def test_bootstrap_is_reproducible_with_same_seed():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])

    a = monte_carlo_simulation(returns, num_simulations=50, method="bootstrap", seed=42)
    b = monte_carlo_simulation(returns, num_simulations=50, method="bootstrap", seed=42)

    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_differs_across_seeds():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])

    a = monte_carlo_simulation(returns, num_simulations=50, method="bootstrap", seed=1)
    b = monte_carlo_simulation(returns, num_simulations=50, method="bootstrap", seed=2)

    assert not a["final_equity"].equals(b["final_equity"])


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        monte_carlo_simulation(pd.Series([0.01]), method="not_a_method")


def test_summarize_monte_carlo_includes_requested_percentiles():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    sim = monte_carlo_simulation(returns, num_simulations=100, seed=3)

    summary = summarize_monte_carlo(sim, percentiles=(5, 50, 95))

    assert "5%" in summary.index
    assert "50%" in summary.index
    assert "95%" in summary.index
