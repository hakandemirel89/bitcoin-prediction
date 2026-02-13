"""
Streamlit dashboard for BTC-IBIT prediction system.
Reuses src/ modules — no logic duplication.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.providers import get_ibit_flows, get_btc_prices, merge_data
from src.features import build_features, get_feature_columns, prepare_dataset
from src.model import (
    walk_forward_train, predictions_to_signals, save_artifacts,
    load_latest_artifacts, generate_tweet, FlowMLP,
)
from src.backtest import run_backtest, compute_metrics, format_metrics
from src.viz import (
    plot_ibit_daily_flow, plot_btc_close, plot_btc_with_flow_overlay,
    plot_equity_curves, plot_equity_btc,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BTC-IBIT Prediction Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("₿ BTC-IBIT Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Data Explorer", "Model", "Backtest", "Daily Prediction"],
    index=0,
)

st.sidebar.markdown("---")
force_csv = st.sidebar.checkbox("Force CSV (offline mode)", value=False)
st.sidebar.caption("Uncheck to fetch live data from Farside + yfinance.")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data(use_csv: bool):
    ibit = get_ibit_flows(force_csv=use_csv)
    btc = get_btc_prices(force_csv=use_csv)
    merged = merge_data(ibit, btc)
    return ibit, btc, merged


@st.cache_data(ttl=3600, show_spinner="Building features...")
def load_features(_merged):
    X, y, dates, feature_cols, clean_df = prepare_dataset(_merged)
    return X, y, dates, feature_cols, clean_df


try:
    ibit_df, btc_df, merged = load_data(force_csv)
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()


# =========================================================================
# PAGE: Data Explorer
# =========================================================================
if page == "Data Explorer":
    st.title("IBIT Flow & BTC Price Data")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    latest = merged.iloc[-1]
    prev = merged.iloc[-2] if len(merged) > 1 else latest
    c1.metric("BTC Close", f"${latest['btc_close']:,.0f}",
              delta=f"{latest['btc_close'] - prev['btc_close']:+,.0f}")
    c2.metric("IBIT Flow (1D)", f"${latest['flow_usd']:+,.1f}M")
    c3.metric("Cum 30D Flow", f"${latest['flow_cum_30d']:,.0f}M")
    c4.metric("Data Points", f"{len(merged)}")

    # Charts
    tab1, tab2, tab3 = st.tabs(["Flow + Price Overlay", "IBIT Daily Flow", "BTC Close"])
    with tab1:
        st.plotly_chart(plot_btc_with_flow_overlay(merged), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_ibit_daily_flow(merged), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_btc_close(merged), use_container_width=True)

    # Data table
    st.subheader("Flow Summary Table")
    n_rows = st.slider("Rows to display", 10, min(200, len(merged)), 30)
    display = merged[["date", "flow_usd", "flow_cum_30d", "flow_cum_60d",
                       "btc_close", "btc_bought_sold"]].tail(n_rows).copy()
    display.columns = ["Date", "1D IBIT Flow (M$)", "Cum 30D (M$)",
                        "Cum 60D (M$)", "BTC Close ($)", "BTC Bought/Sold"]
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")

    def color_flow(val):
        color = "#2ecc71" if val >= 0 else "#e74c3c"
        return f"color: {color}"

    styled = display.style.applymap(color_flow, subset=["1D IBIT Flow (M$)"])
    styled = styled.format({
        "1D IBIT Flow (M$)": "{:+,.1f}",
        "Cum 30D (M$)": "{:,.0f}",
        "Cum 60D (M$)": "{:,.0f}",
        "BTC Close ($)": "${:,.0f}",
        "BTC Bought/Sold": "{:,.1f}",
    })
    st.dataframe(styled, use_container_width=True, height=min(n_rows * 35 + 40, 700))


# =========================================================================
# PAGE: Model
# =========================================================================
elif page == "Model":
    st.title("Model Training & Artifacts")

    # Check for existing artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    meta_files = sorted(artifacts_dir.glob("metadata_*.json")) if artifacts_dir.exists() else []

    if meta_files:
        import json
        latest_meta_path = meta_files[-1]
        with open(latest_meta_path) as f:
            meta = json.load(f)

        st.success(f"Latest artifact: `{latest_meta_path.name}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Features", meta["n_features"])
        c2.metric("Threshold", f"{meta['threshold']}")
        c3.metric("Timestamp", meta["timestamp"])

        # Feature list
        with st.expander("Feature Columns", expanded=False):
            for i, col in enumerate(meta["feature_columns"], 1):
                st.text(f"  {i:2d}. {col}")

        # Fold results
        if meta.get("fold_results"):
            st.subheader("Walk-Forward Fold Results")
            folds_df = pd.DataFrame(meta["fold_results"])
            st.dataframe(folds_df, use_container_width=True)

            # Val loss chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"Fold {r['fold']}" for r in meta["fold_results"]],
                y=[r["best_val_loss"] for r in meta["fold_results"]],
                marker_color="#3498db",
            ))
            fig.update_layout(title="Best Validation Loss per Fold",
                              yaxis_title="MSE", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trained model found. Train one below.")

    # Training button
    st.subheader("Train New Model")
    with st.expander("Training Parameters", expanded=not bool(meta_files)):
        tc1, tc2, tc3 = st.columns(3)
        min_train = tc1.number_input("Min train days", 60, 300, 90)
        val_size = tc2.number_input("Val window", 10, 60, 20)
        test_size = tc3.number_input("Test window", 10, 60, 20)
        tc4, tc5, tc6 = st.columns(3)
        max_epochs = tc4.number_input("Max epochs", 50, 500, 200)
        lr = tc5.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.5f")
        patience_val = tc6.number_input("Early stop patience", 5, 50, 15)
        threshold = st.number_input("Signal threshold", 0.001, 0.05, 0.005, format="%.4f")

    if st.button("Train Walk-Forward Model", type="primary"):
        X, y, dates, feature_cols, clean_df = load_features(merged)

        with st.spinner(f"Training walk-forward on {len(X)} samples..."):
            wf = walk_forward_train(
                X, y, dates,
                min_train=min_train, val_size=val_size, test_size=test_size,
                max_epochs=max_epochs, lr=lr, patience=patience_val,
            )

        if wf["final_model"] is not None:
            save_artifacts(
                wf["final_model"], wf["final_scaler"], feature_cols,
                threshold, wf["fold_results"],
            )
            st.success(f"Model trained: {len(wf['fold_results'])} folds, "
                        f"{len(wf['predictions'])} OOS predictions")
            st.rerun()
        else:
            st.error("Training produced no model — check data size.")


# =========================================================================
# PAGE: Backtest
# =========================================================================
elif page == "Backtest":
    st.title("Backtest")

    # Controls
    with st.sidebar:
        st.subheader("Backtest Controls")
        threshold = st.slider("Signal threshold", 0.001, 0.020, 0.005, 0.001,
                               format="%.3f")
        tx_cost = st.slider("Transaction cost (bps)", 0.0, 50.0, 10.0, 1.0)
        base_asset = st.radio("Base asset", ["USD", "BTC"], index=0)

    # We need features + model
    X, y, dates, feature_cols, clean_df = load_features(merged)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    try:
        model, scaler, meta = load_latest_artifacts(artifacts_dir)
    except FileNotFoundError:
        st.warning("No trained model found. Go to **Model** page to train first.")
        st.stop()

    # Generate predictions using the loaded model
    import torch
    model.eval()
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        predictions = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    signals = predictions_to_signals(predictions, threshold=threshold)

    # Date range filter
    min_date = dates.iloc[0].date()
    max_date = dates.iloc[-1].date()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = col2.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    # Filter
    mask = (dates.dt.date >= start_date) & (dates.dt.date <= end_date)
    bt_dates = dates[mask].reset_index(drop=True)
    bt_prices = clean_df.loc[mask.values, "btc_close"].values
    bt_signals = signals[mask.values]

    if len(bt_dates) < 2:
        st.warning("Need at least 2 data points for backtest.")
        st.stop()

    # Run
    bt = run_backtest(
        dates=bt_dates,
        btc_close=bt_prices,
        signals=bt_signals,
        initial_btc=1.0,
        tx_cost_bps=tx_cost,
    )
    m = bt["metrics"]

    # KPIs
    st.subheader("Summary Statistics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Return (USD)", f"{m['total_return_usd']:+.2%}")
    k2.metric("Return (BTC)", f"{m['total_return_btc']:+.2%}")
    k3.metric("Buy & Hold", f"{m['buy_hold_return_usd']:+.2%}")
    k4.metric("Sharpe", f"{m['sharpe_ratio']:.3f}")
    k5.metric("Max DD (USD)", f"{m['max_drawdown_usd']:.2%}")
    k6.metric("Hit Rate", f"{m['hit_rate']:.1%}")

    k7, k8, k9, k10 = st.columns(4)
    k7.metric("# Trades", m["n_trades"])
    k8.metric("Exposure", f"{m['exposure_pct']:.1%}")
    k9.metric("Final USD", f"${m['final_usd']:,.0f}")
    k10.metric("Final BTC", f"{m['final_btc']:.4f}")

    # Equity charts
    st.subheader("Equity Curves")
    eq_tab1, eq_tab2 = st.tabs(["USD Equity", "BTC Equity"])
    with eq_tab1:
        st.plotly_chart(plot_equity_curves(bt["equity_df"], m["initial_usd"]),
                        use_container_width=True)
    with eq_tab2:
        st.plotly_chart(plot_equity_btc(bt["equity_df"]), use_container_width=True)

    # Drawdown chart
    st.subheader("Drawdown")
    eq = bt["equity_df"]
    if base_asset == "USD":
        peak = np.maximum.accumulate(eq["equity_usd"])
        dd = (eq["equity_usd"] - peak) / peak
        dd_label = "Drawdown (USD)"
    else:
        peak = np.maximum.accumulate(eq["equity_btc"])
        dd = (eq["equity_btc"] - peak) / peak
        dd_label = "Drawdown (BTC)"

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=eq["date"], y=dd, fill="tozeroy",
        line=dict(color="#e74c3c", width=1), name=dd_label,
    ))
    fig_dd.update_layout(
        title=dd_label, yaxis_title="Drawdown", yaxis_tickformat=".1%",
        template="plotly_dark", height=350,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Signal distribution
    st.subheader("Signal Distribution")
    sig_counts = pd.Series(bt_signals).value_counts()
    fig_sig = go.Figure(go.Pie(
        labels=sig_counts.index, values=sig_counts.values,
        marker_colors=["#2ecc71", "#e74c3c", "#95a5a6"],
    ))
    fig_sig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_sig, use_container_width=True)

    # Trades table
    st.subheader("Trade Log")
    if len(bt["trades_df"]) > 0:
        trades_display = bt["trades_df"].copy()
        trades_display["pnl_usd"] = trades_display["pnl_usd"].round(2)
        trades_display["pnl_btc"] = trades_display["pnl_btc"].round(4)
        trades_display["entry_price"] = trades_display["entry_price"].round(2)
        trades_display["exit_price"] = trades_display["exit_price"].round(2)
        st.dataframe(trades_display, use_container_width=True, height=400)
    else:
        st.info("No trades executed in this period.")


# =========================================================================
# PAGE: Daily Prediction
# =========================================================================
elif page == "Daily Prediction":
    st.title("Daily Prediction (Tweet Mode)")

    artifacts_dir = PROJECT_ROOT / "artifacts"
    try:
        model, scaler, meta = load_latest_artifacts(artifacts_dir)
        feature_cols = meta["feature_columns"]
        threshold = meta["threshold"]
    except FileNotFoundError:
        st.warning("No trained model. Go to **Model** page to train first.")
        st.stop()

    # Latest data point
    feat_df = build_features(merged)
    latest = feat_df.iloc[-1]

    # Predict
    import torch
    x = latest[feature_cols].values.astype(np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_scaled, dtype=torch.float32)).item()

    pred_pct = pred * 100
    if pred > threshold:
        signal, signal_color = "LONG", "#2ecc71"
    elif pred < -threshold:
        signal, signal_color = "SHORT", "#e74c3c"
    else:
        signal, signal_color = "FLAT", "#95a5a6"

    date_str = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    flow = latest.get("flow_usd", 0)
    btc_close = latest.get("btc_close", 0)

    # Display
    st.markdown(f"### Signal for {date_str}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC Close", f"${btc_close:,.0f}")
    c2.metric("IBIT Flow", f"${flow:+,.1f}M")
    c3.metric("48h Forecast", f"{pred_pct:+.2f}%")
    c4.markdown(
        f"<div style='text-align:center; padding:12px; background:{signal_color}; "
        f"border-radius:8px; font-size:28px; font-weight:bold; color:white;'>"
        f"{signal}</div>",
        unsafe_allow_html=True,
    )

    # Tweet block
    st.subheader("Tweet-Ready Text")
    flow_sign = "+" if flow >= 0 else ""
    tweet_text = (
        f"BTC-IBIT Signal -- {date_str}\n"
        f"{'=' * 38}\n"
        f"Yesterday's IBIT Flow: {flow_sign}${flow:.1f}M\n"
        f"Latest BTC Close:      ${btc_close:,.0f}\n"
        f"Model Forecast (48h):  {pred_pct:+.2f}%\n"
        f"Signal:                {signal}\n"
        f"{'=' * 38}\n"
        f"Educational only, not financial advice."
    )
    st.code(tweet_text, language=None)

    # Recent predictions table
    st.subheader("Recent Data Context")
    recent = merged[["date", "flow_usd", "flow_cum_30d", "btc_close", "btc_bought_sold"]].tail(10)
    recent.columns = ["Date", "Flow (M$)", "Cum 30D (M$)", "BTC Close", "BTC Bought/Sold"]
    st.dataframe(recent, use_container_width=True)

    st.info("**Disclaimer**: This is an educational tool. Model predictions are based on "
            "limited IBIT flow data and should not be used for actual trading decisions. "
            "Past performance does not guarantee future results.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: {len(merged)} rows | "
    f"{merged['date'].min().strftime('%Y-%m-%d')} to "
    f"{merged['date'].max().strftime('%Y-%m-%d')}"
)
st.sidebar.caption("Educational only. Not financial advice.")
