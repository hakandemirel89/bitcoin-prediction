"""
Plotly visualization for IBIT flow analysis and backtest results.
All functions return plotly Figure objects for notebook display.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_ibit_daily_flow(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of IBIT daily net flows (green=inflow, red=outflow).

    Input df must have columns: date, flow_usd
    """
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["flow_usd"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["flow_usd"],
        marker_color=colors,
        name="IBIT Net Flow",
    ))
    fig.update_layout(
        title="IBIT Daily Net Flow (USD Millions)",
        xaxis_title="Date",
        yaxis_title="Flow (M USD)",
        template="plotly_dark",
        height=450,
        showlegend=False,
    )
    return fig


def plot_btc_close(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of BTC close price.

    Input df must have columns: date, btc_close
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["btc_close"],
        mode="lines",
        name="BTC Close",
        line=dict(color="#f39c12", width=2),
    ))
    fig.update_layout(
        title="BTC Daily Close Price (USD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_btc_with_flow_overlay(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis chart: BTC close (line) + IBIT flow (bars).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Flow bars
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["flow_usd"]]
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["flow_usd"],
            marker_color=colors, name="IBIT Flow", opacity=0.6,
        ),
        secondary_y=False,
    )

    # BTC line
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["btc_close"],
            mode="lines", name="BTC Close",
            line=dict(color="#f39c12", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="BTC Price vs IBIT Flows",
        template="plotly_dark",
        height=500,
    )
    fig.update_yaxes(title_text="IBIT Flow (M USD)", secondary_y=False)
    fig.update_yaxes(title_text="BTC Price (USD)", secondary_y=True)

    return fig


def plot_equity_curves(equity_df: pd.DataFrame, initial_usd: float) -> go.Figure:
    """
    Equity curves: Strategy USD vs Buy & Hold USD.
    """
    fig = go.Figure()

    # Strategy equity
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity_usd"],
        mode="lines",
        name="Strategy (USD)",
        line=dict(color="#3498db", width=2),
    ))

    # Buy & hold
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=initial_usd * (equity_df["btc_close"] / equity_df["btc_close"].iloc[0]),
        mode="lines",
        name="Buy & Hold (USD)",
        line=dict(color="#95a5a6", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Equity Curve — USD",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_equity_btc(equity_df: pd.DataFrame) -> go.Figure:
    """
    Equity curve in BTC terms: Strategy vs holding 1 BTC.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity_btc"],
        mode="lines",
        name="Strategy (BTC)",
        line=dict(color="#9b59b6", width=2),
    ))

    # Benchmark: always 1 BTC
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=[1.0] * len(equity_df),
        mode="lines",
        name="Hold 1 BTC",
        line=dict(color="#95a5a6", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Equity Curve — BTC Denominated",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (BTC)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_flow_summary_table(df: pd.DataFrame) -> go.Figure:
    """
    Plotly table showing the flow summary data.
    Columns: Date, 1D Flow, Cum 30D, Cum 60D, BTC Close, BTC Bought/Sold
    """
    # Use last 30 rows for display
    display = df.tail(30).copy()

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[
                "Date", "1D IBIT Flow (M$)", "Cum 30D Flow (M$)",
                "Cum 60D Flow (M$)", "BTC Close ($)", "BTC Bought/Sold"
            ],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[
                display["date"].dt.strftime("%Y-%m-%d"),
                display["flow_usd"].round(1),
                display["flow_cum_30d"].round(1),
                display["flow_cum_60d"].round(1),
                display["btc_close"].round(0),
                display["btc_bought_sold"].round(2),
            ],
            fill_color=[
                ["#1a1a2e"] * len(display),
                [("#2ecc71" if v >= 0 else "#e74c3c") for v in display["flow_usd"]],
                ["#1a1a2e"] * len(display),
                ["#1a1a2e"] * len(display),
                ["#1a1a2e"] * len(display),
                ["#1a1a2e"] * len(display),
            ],
            font=dict(color="white", size=11),
            align="center",
        ),
    )])

    fig.update_layout(
        title="IBIT Flow Summary (Last 30 Days)",
        template="plotly_dark",
        height=700,
    )
    return fig
