from db.connection import connect


def test_connect_creates_schema_and_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        conn.execute("INSERT INTO accounts (name, cash, last_run_date) VALUES ('a', 100.0, NULL)")

    # a second connection to the same file must not fail re-creating tables
    # and must see what the first connection committed
    with connect(db_path) as conn:
        row = conn.execute("SELECT cash FROM accounts WHERE name = 'a'").fetchone()
    assert row["cash"] == 100.0


def test_connect_commits_on_success_and_rolls_back_is_not_implicit(tmp_path):
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        conn.execute("INSERT INTO accounts (name, cash, last_run_date) VALUES ('a', 100.0, NULL)")

    try:
        with connect(db_path) as conn:
            conn.execute("UPDATE accounts SET cash = 200.0 WHERE name = 'a'")
            raise ValueError("boom")
    except ValueError:
        pass

    with connect(db_path) as conn:
        row = conn.execute("SELECT cash FROM accounts WHERE name = 'a'").fetchone()
    assert row["cash"] == 100.0  # the update was never committed


def test_separate_db_files_are_fully_isolated(tmp_path):
    with connect(tmp_path / "a.db") as conn:
        conn.execute("INSERT INTO accounts (name, cash, last_run_date) VALUES ('x', 1.0, NULL)")

    with connect(tmp_path / "b.db") as conn:
        row = conn.execute("SELECT * FROM accounts WHERE name = 'x'").fetchone()
    assert row is None
