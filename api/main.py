"""FastAPI backend for the paper-trading dashboard.

Serves account state read live from the SQLite store, can trigger a real
daily step per account ("Run Now") or for all five at once, and — once the
frontend is built — serves that build too, so `uvicorn api.main:app` from
the repo root is the one command that runs the whole thing.

    python -m uvicorn api.main:app --reload   # dev, API only (port 8000)
    python -m uvicorn api.main:app            # production, API + built frontend
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.runner import regenerate_dashboard, run_account, run_all, tail_log
from execution.accounts import ACCOUNTS, account_by_key, load_account_state

FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Paper Trading API")

# Permissive during development (Vite's dev server runs on a different
# port than the API); harmless in production since the built frontend is
# served from this same origin and never needs to cross it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/accounts")
def list_accounts():
    return [load_account_state(meta, trade_limit=5) for meta in ACCOUNTS]


@app.get("/api/accounts/{key}")
def get_account(key: str):
    try:
        meta = account_by_key(key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return load_account_state(meta, trade_limit=None)


@app.post("/api/accounts/{key}/run")
def trigger_run(key: str):
    try:
        account_by_key(key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    result = run_account(key)
    regenerate_dashboard()
    return result


@app.post("/api/run-all")
def trigger_run_all():
    return run_all()


@app.get("/api/logs/{name}")
def get_log(name: str, lines: int = 200):
    if name not in ("paper_trading.log", "orchestrator.log"):
        raise HTTPException(status_code=404, detail="Unknown log file")
    return {"name": name, "lines": tail_log(name, lines)}


if FRONTEND_DIST.exists():
    # Vite's own asset files (JS/CSS bundles, favicon) are served as-is.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    # Everything else that isn't /api/* falls back to index.html: this is
    # a single-page app, so React Router (not FastAPI) decides what
    # /accounts/mean_reversion or /logs render -- StaticFiles alone has no
    # such fallback and would 404 on a hard refresh of a client-side route.
    # An /api/* path reaching here means no real route matched it, so it
    # must 404 -- without this check it would silently return the HTML
    # page with a 200, which a fetch() call would then fail to parse as
    # JSON with a much more confusing error than a clean 404.
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        requested = FRONTEND_DIST / full_path
        if full_path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
