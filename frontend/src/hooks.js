import { useCallback, useEffect, useState } from "react";

/** Polls `fetcher` on an interval, refreshing in place -- callers get a
 * refetch() they can call immediately after an action (e.g. a triggered
 * run) instead of waiting for the next tick.
 *
 * `refetch` is memoized on `fetcher` itself (not via a ref that only the
 * interval callback sees), so when the caller passes a *new* fetcher --
 * e.g. AccountDetail's fetcher closing over a different `key` after
 * navigating between accounts, or Logs' after switching which file it
 * reads -- the effect's dependency actually changes too, and it re-fires
 * immediately instead of leaving the previous fetcher's stale data on
 * screen until the next scheduled tick. */
export function usePolling(fetcher, intervalMs = 30_000) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    setLoading(true);
    refetch();
    const id = setInterval(refetch, intervalMs);
    return () => clearInterval(id);
  }, [refetch, intervalMs]);

  return { data, error, loading, refetch };
}
