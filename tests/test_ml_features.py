import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, build_features, build_labels, embargo_train_test_split, make_training_frame


def _ohlcv(n=80, seed=1, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    close = 100 + rng.normal(0, 1, n).cumsum()
    close = np.abs(close) + 50
    return pd.DataFrame(
        {
            "OPEN": close * (1 + rng.normal(0, 0.001, n)),
            "HIGH": close * 1.01,
            "LOW": close * 0.99,
            "CLOSE": close,
            "VOLUME": rng.integers(10_000, 50_000, n),
        },
        index=dates,
    )


def test_build_features_has_every_declared_column():
    df = _ohlcv()
    features = build_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    assert list(features.index) == list(df.index)


def test_build_features_is_causal():
    # row t's feature value must not change when future rows are appended --
    # the same causality property every rule-based strategy depends on.
    df = _ohlcv(n=60)
    truncated = df.iloc[:40]

    full_features = build_features(df)
    truncated_features = build_features(truncated)

    pd.testing.assert_frame_equal(
        full_features.iloc[:40], truncated_features, check_exact=False, rtol=1e-9
    )


def test_build_labels_matches_manual_forward_return():
    df = _ohlcv(n=30)
    labels = build_labels(df, horizon=5, return_threshold=0.0)

    i = 10
    expected_return = df["CLOSE"].iloc[i + 5] / df["CLOSE"].iloc[i] - 1.0
    assert labels.iloc[i] == float(expected_return > 0.0)


def test_build_labels_is_nan_for_the_last_horizon_rows():
    df = _ohlcv(n=20)
    labels = build_labels(df, horizon=5)
    assert labels.iloc[-5:].isna().all()
    assert labels.iloc[:-5].notna().all()


def test_build_labels_respects_return_threshold():
    # a flat-then-jump series: only a large enough forward move should count
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = pd.Series([100, 100, 100, 100, 100, 101, 100, 100, 100, 100], index=dates)
    df = pd.DataFrame({"OPEN": close, "HIGH": close, "LOW": close, "CLOSE": close, "VOLUME": 1000})

    labels_loose = build_labels(df, horizon=1, return_threshold=0.0)
    labels_strict = build_labels(df, horizon=1, return_threshold=0.05)

    assert labels_loose.iloc[4] == 1.0  # +1% clears a 0% threshold
    assert labels_strict.iloc[4] == 0.0  # +1% does not clear a 5% threshold


def test_make_training_frame_drops_unlabeled_tail_but_keeps_feature_warmup_nans():
    df = _ohlcv(n=50)
    frame = make_training_frame(df, horizon=5)

    assert "label" in frame.columns
    assert len(frame) == len(df) - 5  # only the unlabeled tail is dropped
    # early rows (indicator warmup) may still have NaN features -- the model
    # handles those natively, so they aren't dropped here
    assert frame[FEATURE_COLUMNS].isna().any().any()


def test_embargo_train_test_split_purges_a_gap_between_train_and_test():
    df = _ohlcv(n=100)
    train, test = embargo_train_test_split(df, train_frac=0.7, horizon=5)

    split_idx = int(len(df) * 0.7)
    assert train.index[-1] == df.index[split_idx - 5 - 1]
    assert test.index[0] == df.index[split_idx]
    # the horizon rows between them belong to neither split
    gap = df.loc[(df.index > train.index[-1]) & (df.index < test.index[0])]
    assert len(gap) == 5


def test_embargo_train_test_split_train_and_test_never_overlap():
    df = _ohlcv(n=100)
    train, test = embargo_train_test_split(df, train_frac=0.6, horizon=10)
    assert train.index[-1] < test.index[0]
    assert set(train.index).isdisjoint(set(test.index))
