"""
Backtest engine with USD and BTC equity tracking.
Supports Long / Short / Flat positions, no leverage.

Short accounting: inverse-tracking model (daily inverse return on notional).
See plan.md Section 7.1 for full explanation.
"""

import numpy as np
import pandas as pd


def run_backtest(
    dates: pd.Series,
    btc_close: np.ndarray,
    signals: np.ndarray,
    initial_btc: float = 1.0,
    tx_cost_bps: float = 10.0,
) -> dict:
    """
    Run backtest with given signals.

    Parameters:
        dates:        pd.Series of datetime dates
        btc_close:    array of BTC close prices
        signals:      array of 'Long', 'Short', 'Flat' strings
        initial_btc:  starting BTC amount (default 1.0)
        tx_cost_bps:  transaction cost in basis points per position change

    Returns dict with:
        equity_df:    DataFrame (date, position, equity_usd, equity_btc, btc_close)
        trades_df:    DataFrame (date, signal, entry_price, exit_price, pnl_usd, pnl_btc)
        metrics:      dict of performance metrics
    """
    n = len(dates)
    assert len(btc_close) == n == len(signals), "Input length mismatch"

    tx_cost = tx_cost_bps / 10_000

    # Initialize
    initial_usd = initial_btc * btc_close[0]
    equity_usd = np.zeros(n)
    equity_btc = np.zeros(n)
    positions = []

    current_pos = "Flat"  # Start flat, will enter on first signal
    current_equity_usd = initial_usd

    trades = []
    trade_entry_price = None
    trade_entry_usd = None
    trade_entry_date = None

    for i in range(n):
        price = btc_close[i]
        new_signal = signals[i]

        # Apply daily return based on current position
        if i > 0:
            daily_ret = (btc_close[i] - btc_close[i - 1]) / btc_close[i - 1]

            if current_pos == "Long":
                current_equity_usd *= (1 + daily_ret)
            elif current_pos == "Short":
                # Inverse tracking: profit when BTC drops
                inv_ret = -daily_ret
                # Cap loss at 100% of position
                inv_ret = max(inv_ret, -0.99)
                current_equity_usd *= (1 + inv_ret)
            # Flat: no change (holding USD)

        # Check for position change
        if new_signal != current_pos:
            # Apply transaction cost
            current_equity_usd *= (1 - tx_cost)

            # Record trade exit if we had a position
            if trade_entry_price is not None:
                exit_pnl_usd = current_equity_usd - trade_entry_usd
                exit_pnl_btc = (current_equity_usd / price) - (trade_entry_usd / trade_entry_price)
                trades.append({
                    "entry_date": trade_entry_date,
                    "exit_date": dates.iloc[i],
                    "signal": current_pos,
                    "entry_price": trade_entry_price,
                    "exit_price": price,
                    "pnl_usd": exit_pnl_usd,
                    "pnl_btc": exit_pnl_btc,
                })

            # Enter new position
            current_pos = new_signal
            trade_entry_price = price
            trade_entry_usd = current_equity_usd
            trade_entry_date = dates.iloc[i]

        equity_usd[i] = current_equity_usd
        equity_btc[i] = current_equity_usd / price
        positions.append(current_pos)

    # Close final trade
    if trade_entry_price is not None and len(trades) == 0 or (
        trades and trades[-1]["exit_date"] != dates.iloc[-1]
    ):
        trades.append({
            "entry_date": trade_entry_date,
            "exit_date": dates.iloc[-1],
            "signal": current_pos,
            "entry_price": trade_entry_price,
            "exit_price": btc_close[-1],
            "pnl_usd": current_equity_usd - trade_entry_usd,
            "pnl_btc": (current_equity_usd / btc_close[-1]) - (trade_entry_usd / trade_entry_price),
        })

    # Build output DataFrames
    equity_df = pd.DataFrame({
        "date": dates.values,
        "position": positions,
        "equity_usd": equity_usd,
        "equity_btc": equity_btc,
        "btc_close": btc_close,
    })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "signal", "entry_price", "exit_price", "pnl_usd", "pnl_btc"]
    )

    # Compute metrics
    metrics = compute_metrics(equity_df, trades_df, initial_usd, initial_btc)

    return {
        "equity_df": equity_df,
        "trades_df": trades_df,
        "metrics": metrics,
    }


def compute_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_usd: float,
    initial_btc: float,
) -> dict:
    """Compute backtest performance metrics."""
    eq_usd = equity_df["equity_usd"].values
    eq_btc = equity_df["equity_btc"].values
    btc = equity_df["btc_close"].values
    n = len(eq_usd)

    # Returns
    final_usd = eq_usd[-1]
    final_btc = eq_btc[-1]
    total_return_usd = (final_usd / initial_usd) - 1
    total_return_btc = (final_btc / initial_btc) - 1

    # Buy & hold benchmark
    buy_hold_usd = initial_btc * btc[-1]
    buy_hold_return = (buy_hold_usd / initial_usd) - 1

    # Max drawdown (USD)
    peak_usd = np.maximum.accumulate(eq_usd)
    dd_usd = (eq_usd - peak_usd) / peak_usd
    max_dd_usd = dd_usd.min()

    # Max drawdown (BTC)
    peak_btc = np.maximum.accumulate(eq_btc)
    dd_btc = (eq_btc - peak_btc) / peak_btc
    max_dd_btc = dd_btc.min()

    # Daily returns for Sharpe
    daily_returns = np.diff(eq_usd) / eq_usd[:-1]
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Hit rate (% of trades with positive PnL)
    if len(trades_df) > 0 and "pnl_usd" in trades_df.columns:
        winning = (trades_df["pnl_usd"] > 0).sum()
        hit_rate = winning / len(trades_df)
    else:
        hit_rate = 0.0

    # Number of trades (position changes)
    n_trades = len(trades_df)

    # Exposure % (days with non-Flat position)
    non_flat = (equity_df["position"] != "Flat").sum()
    exposure_pct = non_flat / n if n > 0 else 0.0

    return {
        "initial_usd": initial_usd,
        "initial_btc": initial_btc,
        "final_usd": final_usd,
        "final_btc": final_btc,
        "total_return_usd": total_return_usd,
        "total_return_btc": total_return_btc,
        "buy_hold_return_usd": buy_hold_return,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_btc": max_dd_btc,
        "sharpe_ratio": sharpe,
        "hit_rate": hit_rate,
        "n_trades": n_trades,
        "exposure_pct": exposure_pct,
        "n_days": n,
    }


def format_metrics(metrics: dict) -> str:
    """Pretty-print metrics as a formatted string."""
    m = metrics
    lines = [
        "=" * 50,
        "        BACKTEST RESULTS",
        "=" * 50,
        f"  Period:              {m['n_days']} trading days",
        f"  Initial USD:         ${m['initial_usd']:,.2f}",
        f"  Initial BTC:         {m['initial_btc']:.4f}",
        "",
        f"  Final USD:           ${m['final_usd']:,.2f}",
        f"  Final BTC:           {m['final_btc']:.4f}",
        f"  Total Return (USD):  {m['total_return_usd']:+.2%}",
        f"  Total Return (BTC):  {m['total_return_btc']:+.2%}",
        f"  Buy & Hold (USD):    {m['buy_hold_return_usd']:+.2%}",
        "",
        f"  Max Drawdown (USD):  {m['max_drawdown_usd']:.2%}",
        f"  Max Drawdown (BTC):  {m['max_drawdown_btc']:.2%}",
        f"  Sharpe Ratio:        {m['sharpe_ratio']:.3f}",
        f"  Hit Rate:            {m['hit_rate']:.1%}",
        f"  # Trades:            {m['n_trades']}",
        f"  Exposure:            {m['exposure_pct']:.1%}",
        "=" * 50,
    ]
    return "\n".join(lines)
