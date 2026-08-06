import pytest

from risk.position_sizing import fixed_fractional_size, volatility_adjusted_size


def test_fixed_fractional_size():
    assert fixed_fractional_size(equity=100_000, price=100, position_size_pct=0.1) == 100


def test_fixed_fractional_size_zero_for_invalid_inputs():
    assert fixed_fractional_size(100_000, price=0, position_size_pct=0.1) == 0
    assert fixed_fractional_size(100_000, price=100, position_size_pct=0) == 0


def test_volatility_adjusted_size_uses_risk_budget():
    # risk_amount = 100_000 * 0.01 = 1000; risk_per_share = 2 * 2 = 4 -> 250 shares by risk
    # affordability = 100_000 // 50 = 2000 -> risk budget is the binding constraint
    qty = volatility_adjusted_size(equity=100_000, price=50, atr=2, risk_per_trade_pct=0.01, atr_multiple=2.0)
    assert qty == 250


def test_volatility_adjusted_size_capped_by_affordability():
    # tiny ATR gives a huge risk-based quantity; affordability should cap it
    qty = volatility_adjusted_size(equity=1_000, price=50, atr=0.01, risk_per_trade_pct=0.5, atr_multiple=1.0)
    assert qty == 20  # 1000 // 50


def test_volatility_adjusted_size_zero_for_invalid_inputs():
    assert volatility_adjusted_size(100_000, price=0, atr=1, risk_per_trade_pct=0.01) == 0
    assert volatility_adjusted_size(100_000, price=50, atr=0, risk_per_trade_pct=0.01) == 0
