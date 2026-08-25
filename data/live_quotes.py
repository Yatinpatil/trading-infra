"""Best-effort live quotes from Yahoo Finance, for display only.

This is deliberately kept separate from data/loaders.py: paper trading's
fills, stops, and mark-to-market always use NSE's own EOD data, vetted
against the corruption bugs documented there. Yahoo's feed is delayed
(~15-20 min) and unvalidated against those same checks, so it must never
feed a trading decision -- it only tells a viewer of the UI what a
position is worth right now, between EOD runs.
"""
import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def _to_yahoo_symbol(symbol: str) -> str:
    return f"{symbol}.NS"


def get_live_quotes(symbols: list[str]) -> dict[str, dict]:
    """Best-effort {symbol: {price, prev_close, change_pct}}. A symbol Yahoo
    can't price (delisted, renamed, a transient fetch failure) is simply
    omitted rather than raising -- callers should treat a missing key as
    "no live price available," not an error.
    """
    symbols = sorted(set(symbols))
    if not symbols:
        return {}

    quotes = {}
    try:
        tickers = yf.Tickers(" ".join(_to_yahoo_symbol(s) for s in symbols))
    except Exception:
        logger.warning("Live quote batch fetch failed", exc_info=True)
        return {}

    for symbol in symbols:
        try:
            fast_info = tickers.tickers[_to_yahoo_symbol(symbol)].fast_info
            price = fast_info.get("lastPrice")
            prev_close = fast_info.get("previousClose")
            if price is None or prev_close is None:
                continue
            quotes[symbol] = {
                "price": float(price),
                "prev_close": float(prev_close),
                "change_pct": (float(price) - float(prev_close)) / float(prev_close) * 100.0,
            }
        except Exception:
            logger.warning("Live quote failed for %s", symbol, exc_info=True)
    return quotes
