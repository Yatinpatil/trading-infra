"""Reporting layer: equity/drawdown charts, a trade log, a self-contained
HTML tear sheet per strategy run, and a side-by-side comparison table across
strategies. `generate_report()` is the "one command" entry point mentioned
in the project plan — it accepts anything with `.equity_curve` and `.trades`
attributes, so the same call works for single-stock or portfolio results.
"""
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics.metrics import compute_metrics, drawdown_series


def _fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve.index, equity_curve.values, color="#2a6fdb")
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    return fig


def plot_drawdown(equity_curve: pd.Series, title: str = "Drawdown"):
    dd = drawdown_series(equity_curve) * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.35)
    ax.plot(dd.index, dd.values, color="crimson")
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    return fig


def compare_strategies(results: dict[str, dict]) -> pd.DataFrame:
    """`results`: {strategy_name: metrics_dict} (as returned by
    analytics.metrics.compute_metrics) -> one row per strategy, same columns,
    directly comparable.
    """
    comparison = pd.DataFrame(results).T
    comparison.index.name = "strategy"
    return comparison


def generate_tearsheet(
    strategy_name: str,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    output_path,
    metrics: dict | None = None,
) -> Path:
    """Writes a single self-contained HTML tear sheet (charts embedded as
    base64 PNGs, so the file has no external dependencies) for one run.
    """
    metrics = metrics if metrics is not None else compute_metrics(equity_curve, trades)

    equity_img = _fig_to_base64_png(plot_equity_curve(equity_curve, f"{strategy_name} — Equity Curve"))
    drawdown_img = _fig_to_base64_png(plot_drawdown(equity_curve, f"{strategy_name} — Drawdown"))

    metrics_html = pd.DataFrame({"value": metrics}).to_html()
    trades_html = trades.to_html(index=False) if not trades.empty else "<p>No trades.</p>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{strategy_name} — Tear Sheet</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
table {{ border-collapse: collapse; margin-bottom: 2rem; }}
td, th {{ padding: 4px 10px; border: 1px solid #ddd; font-size: 0.9rem; text-align: right; }}
th {{ text-align: left; background: #f5f5f5; }}
img {{ max-width: 100%; margin-bottom: 1.5rem; }}
</style></head>
<body>
<h1>{strategy_name} — Tear Sheet</h1>
<h2>Equity Curve</h2>
<img src="data:image/png;base64,{equity_img}" alt="equity curve">
<h2>Drawdown</h2>
<img src="data:image/png;base64,{drawdown_img}" alt="drawdown">
<h2>Metrics</h2>
{metrics_html}
<h2>Trade Log ({len(trades)} trades)</h2>
{trades_html}
</body></html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_report(strategy_name: str, backtest_result, output_dir) -> Path:
    """Full tear sheet for a single_stock or portfolio BacktestResult
    (anything exposing `.equity_curve` and `.trades`)."""
    metrics = compute_metrics(backtest_result.equity_curve, backtest_result.trades)
    output_path = Path(output_dir) / f"{strategy_name}.html"
    return generate_tearsheet(
        strategy_name, backtest_result.equity_curve, backtest_result.trades, output_path, metrics
    )
