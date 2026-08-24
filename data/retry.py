"""Retry helper for network calls to NSE (via jugaad-data, an unofficial
scraper — Phase 1's commit history already documents flaky/duplicate
responses). Retries on any exception; callers decide what's worth catching
around the retried call.
"""
import time


def with_retries(fn, attempts: int = 3, base_delay: float = 1.0):
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc
