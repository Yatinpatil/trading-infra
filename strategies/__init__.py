from strategies.base import Strategy
from strategies.breakout import BreakoutStrategy
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.momentum import MomentumStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "breakout": BreakoutStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "ml_strategy": MLStrategy,
}


def get_strategy_class(name: str) -> type[Strategy]:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        ) from None
