"""Transaction cost model.

Deliberately simple: brokerage + STT + slippage as a flat percentage of trade
value, charged on every leg (entry and exit). Real Indian equity STT rules
differ between delivery/intraday and buy/sell legs, but the point of this
module — per the project plan's "costs always on" rule — is that no backtest
result is ever reported gross. A config can always set the sub-components to
match a specific broker/segment more precisely; the combination is what's
applied here.
"""


def compute_transaction_cost(trade_value: float, costs_config: dict) -> float:
    rate = (
        costs_config.get("brokerage_pct", 0.0)
        + costs_config.get("stt_pct", 0.0)
        + costs_config.get("slippage_pct", 0.0)
    )
    return trade_value * rate
