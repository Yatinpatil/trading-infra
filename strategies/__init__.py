from strategies.adx_trend import ADXTrendStrategy
from strategies.atr_channel_breakout import ATRChannelBreakoutStrategy
from strategies.base import Strategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.breakout import BreakoutStrategy
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.momentum import MomentumStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "breakout": BreakoutStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "ml_strategy": MLStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
    "adx_trend": ADXTrendStrategy,
    "macd_crossover": MACDCrossoverStrategy,
    "atr_channel_breakout": ATRChannelBreakoutStrategy,
}


def get_strategy_class(name: str) -> type[Strategy]:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        ) from None
