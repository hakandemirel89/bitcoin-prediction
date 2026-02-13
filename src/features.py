"""
Feature engineering for IBIT-flow-based BTC prediction.
All features are backward-looking (no leakage).
"""

import numpy as np
import pandas as pd


# Default feature configuration
FLOW_LAGS = list(range(1, 8))  # 1..7 days
FLOW_ROLL_WINDOWS = [3, 7, 30]
TARGET_HORIZON = 2  # 48h ≈ 2 trading days


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from merged IBIT + BTC dataframe.

    Input must contain: date, flow_usd, btc_close
    Returns a copy with new feature columns and target column.
    All features are backward-looking only (no data leakage).
    """
    out = df.copy()

    # ------------------------------------------------------------------
    # Flow-based features
    # ------------------------------------------------------------------

    # Lagged flows
    for lag in FLOW_LAGS:
        out[f"flow_lag_{lag}"] = out["flow_usd"].shift(lag)

    # Rolling sums
    for w in FLOW_ROLL_WINDOWS:
        out[f"flow_roll_sum_{w}"] = out["flow_usd"].shift(1).rolling(w, min_periods=1).sum()

    # Rolling mean
    for w in [7, 30]:
        out[f"flow_roll_mean_{w}"] = out["flow_usd"].shift(1).rolling(w, min_periods=1).mean()

    # Rolling std
    for w in [7, 30]:
        out[f"flow_roll_std_{w}"] = out["flow_usd"].shift(1).rolling(w, min_periods=max(2, w)).std()

    # Cumulative 30d/60d (already in merged df but re-derive shifted to avoid leakage)
    out["feat_cum_30d"] = out["flow_usd"].shift(1).rolling(30, min_periods=1).sum()
    out["feat_cum_60d"] = out["flow_usd"].shift(1).rolling(60, min_periods=1).sum()

    # Flow z-score (30d)
    roll_mean_30 = out["flow_usd"].shift(1).rolling(30, min_periods=5).mean()
    roll_std_30 = out["flow_usd"].shift(1).rolling(30, min_periods=5).std()
    out["flow_zscore_30d"] = (out["flow_usd"].shift(1) - roll_mean_30) / roll_std_30.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Price-derived features (secondary)
    # ------------------------------------------------------------------

    # Log returns (backward-looking)
    out["btc_log_ret_1d"] = np.log(out["btc_close"] / out["btc_close"].shift(1))
    out["btc_log_ret_5d"] = np.log(out["btc_close"] / out["btc_close"].shift(5))

    # Volatility (7d rolling std of daily log returns)
    out["btc_volatility_7d"] = out["btc_log_ret_1d"].shift(1).rolling(7, min_periods=3).std()

    # BTC bought/sold (shifted by 1 to avoid using today's data)
    out["feat_btc_bought_sold"] = (out["flow_usd"].shift(1) * 1e6) / out["btc_close"].shift(1)

    # ------------------------------------------------------------------
    # Target: 48h forward log return (2 trading days)
    # ------------------------------------------------------------------

    out["target_48h_ret"] = np.log(
        out["btc_close"].shift(-TARGET_HORIZON) / out["btc_close"]
    )

    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes target and metadata)."""
    exclude = {
        "date", "flow_usd", "btc_open", "btc_high", "btc_low", "btc_close",
        "btc_volume", "flow_cum_30d", "flow_cum_60d", "btc_bought_sold",
        "target_48h_ret",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def prepare_dataset(df: pd.DataFrame):
    """
    Build features, drop NaN rows, return (X, y, dates, feature_cols, clean_df).

    Returns:
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        dates: pd.Series of dates for each sample
        feature_cols: list of feature column names
        clean_df: the full DataFrame after dropna
    """
    feat_df = build_features(df)
    feature_cols = get_feature_columns(feat_df)

    # Drop rows with any NaN in features or target
    required_cols = feature_cols + ["target_48h_ret"]
    clean = feat_df.dropna(subset=required_cols).reset_index(drop=True)

    X = clean[feature_cols].values.astype(np.float32)
    y = clean["target_48h_ret"].values.astype(np.float32)
    dates = clean["date"]

    return X, y, dates, feature_cols, clean
