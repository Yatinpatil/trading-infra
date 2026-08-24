"""Schema for the project's single SQLite data store: OHLCV history,
corporate actions, strategy configs, and every paper-trading account's
positions/trades/equity/fitted model all live in one file so they're
queryable together, instead of scattered across parquet/JSON/CSV/YAML
files. `init_schema()` is idempotent -- safe to call on every connection.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol TEXT NOT NULL,
    ex_date TEXT NOT NULL,
    action_type TEXT NOT NULL,
    ratio REAL NOT NULL,
    PRIMARY KEY (symbol, ex_date, action_type)
);

-- fetch freshness for corporate_actions is tracked separately from the
-- rows themselves, since a symbol with genuinely zero actions still needs
-- its TTL tracked (an empty result set can't carry its own timestamp).
CREATE TABLE IF NOT EXISTS corporate_actions_fetch_log (
    symbol TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL
);

-- Raw YAML text, not normalized columns: strategy configs are nested
-- (params/universe/costs/risk dicts) and human-edited, so keeping the
-- YAML text intact means load_config()'s parsing is unchanged, only where
-- the text comes from. file_mtime lets a config seeded from disk pick up
-- a later edit to the YAML file without needing a manual re-seed.
CREATE TABLE IF NOT EXISTS configs (
    name TEXT PRIMARY KEY,
    yaml TEXT NOT NULL,
    file_mtime REAL
);

CREATE TABLE IF NOT EXISTS accounts (
    name TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    last_run_date TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    account TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    entry_cost REAL NOT NULL,
    stop_price REAL,
    PRIMARY KEY (account, symbol)
);

CREATE TABLE IF NOT EXISTS pending_orders (
    account TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('entry', 'exit')),
    PRIMARY KEY (account, symbol, side)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_date TEXT NOT NULL,
    exit_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    entry_cost REAL NOT NULL,
    exit_cost REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    pnl REAL NOT NULL,
    pnl_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_history (
    account TEXT NOT NULL,
    date TEXT NOT NULL,
    equity REAL NOT NULL,
    PRIMARY KEY (account, date)
);

-- MLStrategy's fitted model, serialized (see strategies/ml_strategy.py's
-- to_bytes/from_bytes) -- one row per account, replaced whole on refit.
CREATE TABLE IF NOT EXISTS ml_models (
    account TEXT PRIMARY KEY,
    fitted_at TEXT NOT NULL,
    model_blob BLOB NOT NULL
);
"""


def init_schema(conn) -> None:
    conn.executescript(SCHEMA)
