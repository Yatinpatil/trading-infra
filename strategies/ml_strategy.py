"""Gradient-boosted classifier strategy.

Unlike every other Strategy in this package, MLStrategy carries state beyond
its params dict: a fitted model. That's a deliberate asymmetry, not an
oversight — `fit()` must be called explicitly, once, on a training slice the
caller has already separated from whatever it plans to backtest on (see
ml/features.py's `embargo_train_test_split`), before `generate_signals()`
will do anything. There is no implicit fit-on-first-use, because that would
make it too easy to fit and evaluate on overlapping data by accident.

Once fitted, the same instance is meant to be reused across every symbol in
a universe — engine/portfolio.py already calls `generate_signals(df)` once
per symbol against a single strategy instance, so a model trained once on
pooled, multi-symbol data slots in with no engine changes.
"""
import io
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.features import FEATURE_COLUMNS, make_training_frame, build_features
from strategies.base import Strategy, empty_signals


class MLStrategy(Strategy):
    # Regularized deliberately hard: an unregularized config (max_depth=4,
    # min_samples_leaf=40, l2=0.1) fit the training data well every single
    # walk-forward window (~20-28% CAGR in-sample) but that "edge" mostly
    # didn't survive out-of-sample -- a textbook overfitting signature on
    # this small, correlated a universe. These defaults, found by shrinking
    # capacity until the walk-forward gap narrowed, took NIFTY 50 walk-forward
    # OOS Sharpe from 0.26 to 0.87 -- see scripts/evaluate_ml_strategy.py.
    default_params = {
        "horizon": 5,
        "return_threshold": 0.0,
        "entry_threshold": 0.6,
        "exit_threshold": 0.45,
        "max_iter": 100,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_samples_leaf": 200,
        "l2_regularization": 2.0,
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.model: HistGradientBoostingClassifier | None = None
        self.fitted_at: str | None = None

    def fit(self, ohlcv_by_symbol: dict[str, pd.DataFrame]) -> "MLStrategy":
        """Pool features+labels across every given symbol's history and fit
        one classifier. Every symbol's data should already be sliced to the
        intended training window — this does not know or enforce a
        train/test boundary itself.
        """
        horizon = self.params["horizon"]
        return_threshold = self.params["return_threshold"]
        frames = [make_training_frame(df, horizon, return_threshold) for df in ohlcv_by_symbol.values()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise ValueError("No training rows: every symbol's history was shorter than `horizon`")
        training = pd.concat(frames, ignore_index=True)

        self.model = HistGradientBoostingClassifier(
            max_iter=self.params["max_iter"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            min_samples_leaf=self.params["min_samples_leaf"],
            l2_regularization=self.params["l2_regularization"],
            random_state=0,
        )
        self.model.fit(training[FEATURE_COLUMNS], training["label"])
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("MLStrategy.generate_signals() called before fit()")

        signals = empty_signals(df.index)
        features = build_features(df)
        proba = pd.Series(self.model.predict_proba(features[FEATURE_COLUMNS])[:, 1], index=df.index)

        signals["entry_long"] = proba >= self.params["entry_threshold"]
        signals["exit_long"] = proba <= self.params["exit_threshold"]
        return signals

    def to_bytes(self) -> bytes:
        """Serialize params + the fitted model (and `fitted_at`, if the
        caller set one) so a long-running process — paper trading in
        particular — doesn't have to refit from scratch on every
        invocation. Returns raw bytes so the caller can store them
        anywhere (a file, a database BLOB column, ...); `save`/`load`
        below are just a file-based convenience over this."""
        if self.model is None:
            raise RuntimeError("MLStrategy.to_bytes() called before fit()")
        buf = io.BytesIO()
        joblib.dump({"params": self.params, "model": self.model, "fitted_at": self.fitted_at}, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "MLStrategy":
        state = joblib.load(io.BytesIO(data))
        strategy = cls(state["params"])
        strategy.model = state["model"]
        strategy.fitted_at = state.get("fitted_at")
        return strategy

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path) -> "MLStrategy":
        return cls.from_bytes(Path(path).read_bytes())
