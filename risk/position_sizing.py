"""Position sizing rules. Each returns a share quantity, already floored to
an integer and capped by what the given equity can actually afford.
"""


def fixed_fractional_size(equity: float, price: float, position_size_pct: float) -> int:
    """Spend a fixed fraction of `equity` on this position."""
    if price <= 0 or position_size_pct <= 0:
        return 0
    budget = equity * position_size_pct
    return int(budget // price)


def volatility_adjusted_size(
    equity: float,
    price: float,
    atr: float,
    risk_per_trade_pct: float,
    atr_multiple: float = 2.0,
) -> int:
    """Size so that a stop placed `atr_multiple` * ATR away risks exactly
    `risk_per_trade_pct` of equity — the classic "risk the same rupee amount
    per trade regardless of the stock's volatility" rule.
    """
    if price <= 0 or atr <= 0 or risk_per_trade_pct <= 0:
        return 0

    risk_amount = equity * risk_per_trade_pct
    risk_per_share = atr * atr_multiple
    quantity_by_risk = int(risk_amount // risk_per_share)
    quantity_by_affordability = int(equity // price)

    return max(0, min(quantity_by_risk, quantity_by_affordability))
