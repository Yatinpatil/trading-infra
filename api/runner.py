"""Wraps run_paper_trading.py so the API can trigger a real daily step from
a request handler -- a "Run Now" click and tonight's scheduled run go
through the exact same code path (locking, logging, refit logic included),
not two separate implementations of the same thing.

run_paper_trading.main() is called in-process (it's already safe to invoke
repeatedly within one process -- the test suite does exactly that), while
dashboard regeneration is shelled out to scripts/generate_dashboard.py,
matching how the nightly orchestrator already does it.
"""
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_paper_trading  # noqa: E402
from execution.accounts import ACCOUNTS, account_by_key  # noqa: E402

logger = logging.getLogger("api")


def run_account(key: str) -> dict:
    meta = account_by_key(key)
    try:
        summary = run_paper_trading.main(["--config", meta["config"]])
    except Exception as exc:
        logger.exception("Manual run failed for %s", key)
        return {"key": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"key": key, "ok": True, "summary": summary}


def run_all() -> list[dict]:
    results = [run_account(meta["key"]) for meta in ACCOUNTS]
    regenerate_dashboard()
    return results


def regenerate_dashboard() -> None:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    python = str(python) if python.exists() else sys.executable
    try:
        subprocess.run([python, str(ROOT / "scripts" / "generate_dashboard.py")], cwd=ROOT, check=True)
    except Exception:
        logger.exception("Dashboard regeneration failed after a manual run")


def tail_log(name: str, lines: int = 200) -> list[str]:
    path = ROOT / "logs" / name
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
