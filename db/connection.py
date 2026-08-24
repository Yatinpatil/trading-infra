"""Connection helper for the project's SQLite data store.

`connect()` is the one entry point every other module should use --
it opens a connection, makes sure the schema exists, and hands back a
context manager that commits on success and always closes. There's no
long-lived shared connection: each call opens and closes its own, since
usage here is one local process doing occasional reads/writes, not a
server handling concurrent load, and SQLite handles that pattern fine.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from db.schema import init_schema

DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"


def get_connection(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


@contextmanager
def connect(db_path=None):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
