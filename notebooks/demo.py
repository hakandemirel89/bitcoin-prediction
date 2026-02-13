# %% [markdown]
# # BTC Prediction from IBIT ETF Flows
# Walk-forward neural network model with full backtest.
# See `plan.md` for architecture and assumptions.

# %% Cell 1 — Setup
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("demo")

print(f"PyTorch: {torch.__version__}")
print(f"Device:  cpu (Raspberry Pi)")
print(f"Project: {PROJECT_ROOT}")

# %% Cell 2 — Load Data
from src.providers import get_ibit_flows, get_btc_prices, merge_data

# Try online sources first, fall back to CSV automatically
ibit_df = get_ibit_flows(force_csv=False)
btc_df = get_btc_prices(force_csv=False)

print(f"\nIBIT flows: {len(ibit_df)} rows, {ibit_df['date'].min()} to {ibit_df['date'].max()}")
print(f"BTC prices: {len(btc_df)} rows, {btc_df['date'].min()} to {btc_df['date'].max()}")

# Merge
merged = merge_data(ibit_df, btc_df)
print(f"Merged:     {len(merged)} rows")
print(f"\nMerged columns: {list(merged.columns)}")

# %% Cell 3 — Display Flow Summary Table
summary_cols = ["date", "flow_usd", "flow_cum_30d", "flow_cum_60d", "btc_close", "btc_bought_sold"]
print("\n--- Flow Summary (last 15 rows) ---")
print(merged[summary_cols].tail(15).to_string(index=False, float_format="%.1f"))

# %% Cell 4 — Feature Engineering
from src.features import build_features, prepare_dataset, get_feature_columns

X, y, dates, feature_cols, clean_df = prepare_dataset(merged)

print(f"\nFeatures:   {len(feature_cols)}")
print(f"Samples:    {len(X)} (after dropping NaN)")
print(f"Date range: {dates.iloc[0]} to {dates.iloc[-1]}")
print(f"\nFeature list:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nTarget stats (48h log return):")
print(f"  mean:  {y.mean():.5f}")
print(f"  std:   {y.std():.5f}")
print(f"  min:   {y.min():.5f}")
print(f"  max:   {y.max():.5f}")

# %% Cell 5 — Walk-Forward Model Training
from src.model import walk_forward_train, predictions_to_signals, save_artifacts

logger.info("Starting rolling walk-forward training...")

wf_results = walk_forward_train(
    X, y, dates,
    train_size=90,      # Rolling window: last 90 days
    val_size=20,
    step_size=1,        # Daily rolling model (new model per day)
    max_epochs=200,
    batch_size=32,
    lr=1e-3,
    patience=15,
)

predictions = wf_results["predictions"]
actuals = wf_results["actuals"]
pred_dates = wf_results["pred_dates"]

print(f"\nWalk-Forward Results:")
print(f"  Folds:       {len(wf_results['fold_results'])}")
print(f"  OOS samples: {len(predictions)}")

if len(predictions) > 0:
    # Prediction quality
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    corr = np.corrcoef(actuals, predictions)[0, 1] if len(predictions) > 1 else 0
    direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))

    print(f"  MSE:             {mse:.8f}")
    print(f"  MAE:             {mae:.6f}")
    print(f"  Correlation:     {corr:.4f}")
    print(f"  Direction Acc:   {direction_acc:.2%}")

    for fold in wf_results["fold_results"]:
        print(f"  Fold {fold['fold']}: train={fold['train_size']}, "
              f"test={fold['test_size']}, val_loss={fold['best_val_loss']:.6f}, "
              f"epochs={fold['epochs_trained']}")

# %% Cell 6 — Generate Signals & Run Backtest
from src.backtest import run_backtest, format_metrics

SIGNAL_THRESHOLD = 0.005

if len(predictions) > 0:
    signals = predictions_to_signals(predictions, threshold=SIGNAL_THRESHOLD)

    # Build backtest dataframe: match prediction dates to prices
    bt_df = clean_df[clean_df["date"].isin(pred_dates)].reset_index(drop=True)
    bt_signals = signals[:len(bt_df)]

    print(f"\nSignal distribution:")
    for s in ["Long", "Short", "Flat"]:
        count = (bt_signals == s).sum()
        print(f"  {s:5s}: {count} ({count/len(bt_signals):.1%})")

    bt_result = run_backtest(
        dates=bt_df["date"],
        btc_close=bt_df["btc_close"].values,
        signals=bt_signals,
        initial_btc=1.0,
        tx_cost_bps=10.0,
    )

    print("\n" + format_metrics(bt_result["metrics"]))

    # Show trades
    print("\n--- Trade Log ---")
    if len(bt_result["trades_df"]) > 0:
        print(bt_result["trades_df"].to_string(index=False, float_format="%.2f"))
    else:
        print("  No trades executed.")
else:
    print("\nNo predictions available — skipping backtest.")
    bt_result = None

# %% Cell 7 — Charts
from src.viz import (
    plot_ibit_daily_flow,
    plot_btc_close,
    plot_btc_with_flow_overlay,
    plot_equity_curves,
    plot_equity_btc,
    plot_flow_summary_table,
)

print("\nGenerating charts...")

# Chart 1: IBIT Daily Flow
fig_flow = plot_ibit_daily_flow(merged)
fig_flow.show()

# Chart 2: BTC Close
fig_btc = plot_btc_close(merged)
fig_btc.show()

# Chart 3: BTC + Flow overlay
fig_overlay = plot_btc_with_flow_overlay(merged)
fig_overlay.show()

# Chart 4 & 5: Equity curves (if backtest ran)
if bt_result is not None:
    fig_eq_usd = plot_equity_curves(bt_result["equity_df"], bt_result["metrics"]["initial_usd"])
    fig_eq_usd.show()

    fig_eq_btc = plot_equity_btc(bt_result["equity_df"])
    fig_eq_btc.show()

# Chart 6: Flow Summary Table
fig_table = plot_flow_summary_table(merged)
fig_table.show()

print("Charts generated.")

# %% Cell 8 — Tweet Mode Prediction
from src.model import generate_tweet

if wf_results["final_model"] is not None:
    tweet = generate_tweet(
        df=merged,
        model=wf_results["final_model"],
        scaler=wf_results["final_scaler"],
        feature_cols=feature_cols,
        threshold=SIGNAL_THRESHOLD,
    )
    print("\n" + tweet)
else:
    print("\nNo model available for tweet generation.")

# %% Cell 9 — Save Artifacts
if wf_results["final_model"] is not None:
    save_dir = save_artifacts(
        model=wf_results["final_model"],
        scaler=wf_results["final_scaler"],
        feature_cols=feature_cols,
        threshold=SIGNAL_THRESHOLD,
        fold_results=wf_results["fold_results"],
        predictions_data={
            "predictions": wf_results["predictions"],
            "actuals": wf_results["actuals"],
            "pred_dates": wf_results["pred_dates"],
        },
    )
    print(f"\nArtifacts saved to: {save_dir}")
else:
    print("\nNo model to save.")

# %% Cell 10 — Summary
print("\n" + "=" * 50)
print("  PIPELINE COMPLETE")
print("=" * 50)
print(f"  Data points:     {len(merged)}")
print(f"  Features:        {len(feature_cols)}")
print(f"  Walk-fwd folds:  {len(wf_results['fold_results'])}")
print(f"  OOS predictions: {len(predictions)}")
if bt_result is not None:
    m = bt_result["metrics"]
    print(f"  Strategy return: {m['total_return_usd']:+.2%} (USD)")
    print(f"  Buy&Hold return: {m['buy_hold_return_usd']:+.2%} (USD)")
    print(f"  Sharpe ratio:    {m['sharpe_ratio']:.3f}")
print("=" * 50)
