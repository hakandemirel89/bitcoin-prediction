"""Tests for feature engineering — leakage checks and correctness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_features, get_feature_columns, prepare_dataset


@pytest.fixture
def sample_df():
    """Create minimal merged dataframe for testing."""
    n = 100
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-11", periods=n)
    return pd.DataFrame({
        "date": dates,
        "flow_usd": np.random.normal(200, 150, n),
        "btc_open": 40000 + np.cumsum(np.random.normal(0, 500, n)),
        "btc_high": 41000 + np.cumsum(np.random.normal(0, 500, n)),
        "btc_low": 39000 + np.cumsum(np.random.normal(0, 500, n)),
        "btc_close": 40000 + np.cumsum(np.random.normal(0, 500, n)),
        "btc_volume": np.random.uniform(20e9, 50e9, n),
    })


def test_build_features_columns(sample_df):
    """Check that expected feature columns are created."""
    result = build_features(sample_df)
    assert "flow_lag_1" in result.columns
    assert "flow_lag_7" in result.columns
    assert "flow_roll_sum_3" in result.columns
    assert "flow_roll_sum_30" in result.columns
    assert "flow_roll_mean_7" in result.columns
    assert "flow_roll_std_30" in result.columns
    assert "feat_cum_30d" in result.columns
    assert "feat_cum_60d" in result.columns
    assert "flow_zscore_30d" in result.columns
    assert "btc_log_ret_1d" in result.columns
    assert "btc_log_ret_5d" in result.columns
    assert "btc_volatility_7d" in result.columns
    assert "target_48h_ret" in result.columns


def test_target_is_forward_looking(sample_df):
    """Target should use future prices (shifted negative)."""
    result = build_features(sample_df)
    # Last 2 rows should have NaN target (no future data)
    assert pd.isna(result["target_48h_ret"].iloc[-1])
    assert pd.isna(result["target_48h_ret"].iloc[-2])
    # Earlier rows should have valid targets
    assert not pd.isna(result["target_48h_ret"].iloc[0])


def test_no_leakage_in_features(sample_df):
    """Features at time t should not use data from time t+1 or later."""
    result = build_features(sample_df)

    # flow_lag_1 at row i should equal flow_usd at row i-1
    for i in range(1, 10):
        expected = sample_df["flow_usd"].iloc[i - 1]
        actual = result["flow_lag_1"].iloc[i]
        assert np.isclose(actual, expected), f"Lag 1 leakage at row {i}"

    # flow_lag_1 at row 0 should be NaN (no previous data)
    assert pd.isna(result["flow_lag_1"].iloc[0])


def test_rolling_features_backward_only(sample_df):
    """Rolling features use shift(1) so they don't include today's value."""
    result = build_features(sample_df)
    # flow_roll_sum_3 at row i should be sum of flow_usd[i-3:i] (exclusive of i)
    # i.e., shifted by 1 then rolling 3
    for i in range(4, 10):
        expected = sample_df["flow_usd"].iloc[i - 3:i].sum()
        actual = result["flow_roll_sum_3"].iloc[i]
        assert np.isclose(actual, expected, rtol=1e-5), f"Rolling sum leakage at row {i}"


def test_prepare_dataset_shapes(sample_df):
    """prepare_dataset should return matching shapes with no NaN."""
    X, y, dates, feature_cols, clean = prepare_dataset(sample_df)
    assert X.shape[0] == y.shape[0] == len(dates)
    assert X.shape[1] == len(feature_cols)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))


def test_feature_count(sample_df):
    """Should produce ~20 features."""
    result = build_features(sample_df)
    cols = get_feature_columns(result)
    assert 15 <= len(cols) <= 25, f"Expected ~20 features, got {len(cols)}"


def test_target_values_reasonable(sample_df):
    """48h log returns should be small (not wildly off)."""
    result = build_features(sample_df)
    target = result["target_48h_ret"].dropna()
    # Daily moves are typically < 20% for BTC, 48h should be < 40%
    assert target.abs().max() < 0.5, "Target values unreasonably large"
