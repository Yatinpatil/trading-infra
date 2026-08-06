from engine.costs import compute_transaction_cost


def test_compute_transaction_cost_sums_all_components():
    costs = {"brokerage_pct": 0.0003, "stt_pct": 0.001, "slippage_pct": 0.0005}
    cost = compute_transaction_cost(100_000.0, costs)
    assert cost == 100_000.0 * (0.0003 + 0.001 + 0.0005)


def test_compute_transaction_cost_missing_components_default_to_zero():
    cost = compute_transaction_cost(100_000.0, {"brokerage_pct": 0.001})
    assert cost == 100.0


def test_compute_transaction_cost_empty_config_is_zero():
    assert compute_transaction_cost(100_000.0, {}) == 0.0
