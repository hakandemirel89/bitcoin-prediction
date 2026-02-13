"""Tests for backtest engine — accounting correctness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest, compute_metrics


@pytest.fixture
def simple_data():
    """Simple 10-day scenario with known prices."""
    dates = pd.bdate_range("2024-06-01", periods=10)
    prices = np.array([
        100.0, 105.0, 110.0, 108.0, 112.0,
        115.0, 113.0, 118.0, 120.0, 125.0,
    ])
    return dates, prices


def test_buy_and_hold_long(simple_data):
    """All-long should match buy-and-hold (minus initial tx cost)."""
    dates, prices = simple_data
    signals = np.array(["Long"] * 10)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=0.0,  # No cost for clean comparison
    )

    # USD equity should track BTC price (started with 1 BTC)
    expected_final_usd = prices[-1]
    actual_final_usd = result["equity_df"]["equity_usd"].iloc[-1]
    assert np.isclose(actual_final_usd, expected_final_usd, rtol=0.01), \
        f"Expected ~{expected_final_usd}, got {actual_final_usd}"


def test_flat_holds_usd(simple_data):
    """All-flat should maintain initial USD value (minus initial tx cost)."""
    dates, prices = simple_data
    signals = np.array(["Flat"] * 10)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=0.0,
    )

    # Equity should stay at initial USD (100) throughout since we start Flat
    initial_usd = prices[0]  # 1 BTC * 100
    final_usd = result["equity_df"]["equity_usd"].iloc[-1]
    assert np.isclose(final_usd, initial_usd, rtol=0.01), \
        f"Flat should hold USD, expected ~{initial_usd}, got {final_usd}"


def test_short_profits_on_decline():
    """Short position should profit when BTC price drops."""
    dates = pd.bdate_range("2024-06-01", periods=5)
    prices = np.array([100.0, 95.0, 90.0, 85.0, 80.0])  # Steady decline
    signals = np.array(["Short"] * 5)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=0.0,
    )

    initial_usd = prices[0]
    final_usd = result["equity_df"]["equity_usd"].iloc[-1]
    # Short should profit as price drops
    assert final_usd > initial_usd, \
        f"Short should profit on decline, got {final_usd} vs initial {initial_usd}"


def test_short_loses_on_rise():
    """Short position should lose when BTC price rises."""
    dates = pd.bdate_range("2024-06-01", periods=5)
    prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0])  # Steady rise
    signals = np.array(["Short"] * 5)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=0.0,
    )

    initial_usd = prices[0]
    final_usd = result["equity_df"]["equity_usd"].iloc[-1]
    assert final_usd < initial_usd, \
        f"Short should lose on rise, got {final_usd} vs initial {initial_usd}"


def test_transaction_costs_applied(simple_data):
    """Switching positions should reduce equity by tx cost."""
    dates, prices = simple_data
    # Switch every day to maximize tx costs
    signals = np.array(["Long", "Flat", "Long", "Flat", "Long",
                         "Flat", "Long", "Flat", "Long", "Flat"])

    result_with_cost = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=100.0,  # 1% cost (high to see effect)
    )

    result_no_cost = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=0.0,
    )

    # With costs should be strictly less
    assert result_with_cost["metrics"]["final_usd"] < result_no_cost["metrics"]["final_usd"]


def test_btc_equity_consistency(simple_data):
    """equity_btc should equal equity_usd / btc_close at every point."""
    dates, prices = simple_data
    signals = np.array(["Long", "Long", "Short", "Short", "Flat",
                         "Flat", "Long", "Long", "Short", "Long"])

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        initial_btc=1.0,
        tx_cost_bps=10.0,
    )

    eq = result["equity_df"]
    expected_btc = eq["equity_usd"] / eq["btc_close"]
    np.testing.assert_allclose(eq["equity_btc"].values, expected_btc.values, rtol=1e-6)


def test_metrics_keys(simple_data):
    """Metrics dict should contain all required keys."""
    dates, prices = simple_data
    signals = np.array(["Long"] * 10)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
    )

    required = [
        "initial_usd", "initial_btc", "final_usd", "final_btc",
        "total_return_usd", "total_return_btc", "buy_hold_return_usd",
        "max_drawdown_usd", "max_drawdown_btc", "sharpe_ratio",
        "hit_rate", "n_trades", "exposure_pct", "n_days",
    ]
    for key in required:
        assert key in result["metrics"], f"Missing metric: {key}"


def test_exposure_calculation(simple_data):
    """Exposure should reflect fraction of non-flat days."""
    dates, prices = simple_data
    # 5 Long, 5 Flat → 50% exposure
    signals = np.array(["Long"] * 5 + ["Flat"] * 5)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
        tx_cost_bps=0.0,
    )

    assert np.isclose(result["metrics"]["exposure_pct"], 0.5, atol=0.05)


def test_max_drawdown_not_positive(simple_data):
    """Max drawdown should be zero or negative."""
    dates, prices = simple_data
    signals = np.array(["Long"] * 10)

    result = run_backtest(
        dates=pd.Series(dates),
        btc_close=prices,
        signals=signals,
    )

    assert result["metrics"]["max_drawdown_usd"] <= 0
    assert result["metrics"]["max_drawdown_btc"] <= 0
