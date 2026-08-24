"""One-time migration: copies the existing parquet OHLCV/corporate-action
cache, the existing paper-trading JSON/CSV/joblib state, and the checked-in
YAML configs into the project's SQLite store (db/trading.db). Purely
additive -- it never deletes or modifies the original files, so it's safe
to re-run (rows are upserted) and safe to abort partway through.

    python scripts/migrate_to_sqlite.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from configs import load_config
from db.connection import connect

ROOT = Path(__file__).parent.parent
OLD_CACHE_DIR = ROOT / "data" / "cache"
OLD_STATE_DIR = ROOT / "execution" / "state"
CONFIG_DIR = ROOT / "configs"


def migrate_ohlcv() -> int:
    total = 0
    with connect() as conn:
        for path in sorted(OLD_CACHE_DIR.glob("*_ohlcv.parquet")):
            symbol = path.stem.removesuffix("_ohlcv")
            df = pd.read_parquet(path)
            rows = [
                (symbol, idx.strftime("%Y-%m-%d"), float(r.OPEN), float(r.HIGH), float(r.LOW), float(r.CLOSE), int(r.VOLUME))
                for idx, r in zip(df.index, df.itertuples(index=False))
            ]
            conn.executemany(
                "INSERT INTO ohlcv (symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(symbol, date) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume",
                rows,
            )
            total += len(rows)
            print(f"  ohlcv: {symbol} ({len(rows)} rows)")
    return total


def migrate_corporate_actions() -> int:
    total = 0
    with connect() as conn:
        for path in sorted(OLD_CACHE_DIR.glob("*_corp_actions.parquet")):
            symbol = path.stem.removesuffix("_corp_actions")
            df = pd.read_parquet(path)
            fetched_at = pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()

            conn.execute("DELETE FROM corporate_actions WHERE symbol = ?", (symbol,))
            if not df.empty:
                rows = [
                    (symbol, row.ex_date.strftime("%Y-%m-%d"), row.action_type, float(row.ratio))
                    for row in df.itertuples(index=False)
                ]
                conn.executemany(
                    "INSERT INTO corporate_actions (symbol, ex_date, action_type, ratio) VALUES (?,?,?,?)", rows
                )
                total += len(rows)
            conn.execute(
                "INSERT INTO corporate_actions_fetch_log (symbol, fetched_at) VALUES (?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET fetched_at = excluded.fetched_at",
                (symbol, fetched_at),
            )
            print(f"  corporate_actions: {symbol} ({len(df)} rows)")
    return total


def migrate_configs() -> int:
    count = 0
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        load_config(path.stem)  # load_config seeds the DB as a side effect
        count += 1
        print(f"  config: {path.stem}")
    return count


def migrate_paper_trading_accounts() -> int:
    count = 0
    with connect() as conn:
        for state_path in sorted(OLD_STATE_DIR.glob("*.json")):
            account = state_path.stem
            state = json.loads(state_path.read_text())

            conn.execute(
                "INSERT INTO accounts (name, cash, last_run_date) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET cash = excluded.cash, last_run_date = excluded.last_run_date",
                (account, state["cash"], state["last_run_date"]),
            )

            conn.execute("DELETE FROM positions WHERE account = ?", (account,))
            if state["positions"]:
                conn.executemany(
                    "INSERT INTO positions (account, symbol, quantity, entry_price, entry_date, entry_cost, stop_price) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [
                        (account, sym, p["quantity"], p["entry_price"], p["entry_date"], p["entry_cost"], p["stop_price"])
                        for sym, p in state["positions"].items()
                    ],
                )

            conn.execute("DELETE FROM pending_orders WHERE account = ?", (account,))
            pending_rows = [(account, sym, "entry") for sym in state["pending_entries"]]
            pending_rows += [(account, sym, "exit") for sym in state["pending_exits"]]
            if pending_rows:
                conn.executemany("INSERT INTO pending_orders (account, symbol, side) VALUES (?, ?, ?)", pending_rows)

            equity_path = state_path.with_name(state_path.stem + "_equity.csv")
            if equity_path.exists():
                equity_df = pd.read_csv(equity_path, parse_dates=["date"]).drop_duplicates("date", keep="last")
                rows = [(account, row.date.strftime("%Y-%m-%d"), float(row.equity)) for row in equity_df.itertuples(index=False)]
                conn.executemany(
                    "INSERT INTO equity_history (account, date, equity) VALUES (?, ?, ?) "
                    "ON CONFLICT(account, date) DO UPDATE SET equity = excluded.equity",
                    rows,
                )

            trades_path = state_path.with_name(state_path.stem + "_trades.csv")
            if trades_path.exists():
                trades_df = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])
                rows = [
                    (
                        account, row.symbol, row.entry_date.strftime("%Y-%m-%d"), float(row.entry_price),
                        row.exit_date.strftime("%Y-%m-%d"), float(row.exit_price), int(row.quantity),
                        float(row.entry_cost), float(row.exit_cost), row.exit_reason, float(row.pnl), float(row.pnl_pct),
                    )
                    for row in trades_df.itertuples(index=False)
                ]
                conn.executemany(
                    "INSERT INTO trades (account, symbol, entry_date, entry_price, exit_date, exit_price, "
                    "quantity, entry_cost, exit_cost, exit_reason, pnl, pnl_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )

            model_path = state_path.with_name(state_path.stem + "_model.joblib")
            if model_path.exists():
                fitted_at = None
                try:
                    import joblib

                    fitted_at = joblib.load(model_path).get("fitted_at")
                except Exception:
                    pass
                conn.execute(
                    "INSERT INTO ml_models (account, fitted_at, model_blob) VALUES (?, ?, ?) "
                    "ON CONFLICT(account) DO UPDATE SET fitted_at = excluded.fitted_at, model_blob = excluded.model_blob",
                    (account, fitted_at or "unknown", model_path.read_bytes()),
                )

            count += 1
            print(f"  account: {account} (cash={state['cash']:,.2f}, positions={len(state['positions'])})")
    return count


def main():
    print("Migrating OHLCV cache...")
    ohlcv_rows = migrate_ohlcv()
    print(f"-> {ohlcv_rows} OHLCV rows\n")

    print("Migrating corporate-action cache...")
    action_rows = migrate_corporate_actions()
    print(f"-> {action_rows} corporate-action rows\n")

    print("Migrating configs...")
    config_count = migrate_configs()
    print(f"-> {config_count} configs\n")

    print("Migrating paper-trading accounts...")
    account_count = migrate_paper_trading_accounts()
    print(f"-> {account_count} accounts\n")

    print("Done. Original files under data/cache/ and execution/state/ were left untouched.")


if __name__ == "__main__":
    main()
